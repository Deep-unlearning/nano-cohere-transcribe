"""CUDA graph capture for the autoregressive decoder step.

Per-step kernel-launch overhead is the dominant cost at low batch sizes (the
bs=1 short-form regime). PyTorch's ``torch.cuda.CUDAGraph`` lets us replay an
entire decoder step in one GPU dispatch, but graphs require static shapes —
incompatible with the ``torch.cat``-grown KV cache used by the eager path.

This module solves that with a fixed-shape KV buffer: pre-allocate
``[B, H, max_kv, Dh]`` and write each new token's K/V at index ``pos_idx`` via
``index_copy_``. The self-attn mask is reconstructed each replay from
``pos_idx`` so positions ``> pos_idx`` are zeroed out.

Captured once per ``(B, T_enc)`` pair and cached on the model. Cross-attn KV is
populated from the prefill output before capture.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _step_layer(
    layer,
    x: torch.Tensor,
    pos_idx: torch.Tensor,
    self_k: torch.Tensor,
    self_v: torch.Tensor,
    cross_k: torch.Tensor,
    cross_v: torch.Tensor,
    self_mask: torch.Tensor,
    cross_mask: torch.Tensor,
) -> torch.Tensor:
    """Single-token forward through one ``TransformerDecoderLayer`` with fixed-shape KV.

    Mirrors :class:`TransformerDecoderLayer.forward` but writes new K/V
    in-place into pre-allocated buffers rather than concat-growing them.
    """
    B = x.size(0)
    sa = layer.first_sub_layer
    H, D, Dh = sa.num_heads, sa.hidden_size, sa.head_dim

    # ---- Self-attention with fixed-shape KV ----
    h = layer.layer_norm_1(x)
    q = sa.query_net(h).view(B, 1, H, Dh).transpose(1, 2)
    new_k = sa.key_net(h).view(B, 1, H, Dh).transpose(1, 2)
    new_v = sa.value_net(h).view(B, 1, H, Dh).transpose(1, 2)
    self_k.index_copy_(2, pos_idx, new_k)
    self_v.index_copy_(2, pos_idx, new_v)
    attn = F.scaled_dot_product_attention(
        q, self_k, self_v, attn_mask=self_mask, dropout_p=0.0, scale=sa.scale,
    )
    attn = attn.transpose(1, 2).contiguous().view(B, 1, D)
    x = x + sa.out_projection(attn)

    # ---- Cross-attention against prefilled cache ----
    h = layer.layer_norm_2(x)
    ca = layer.second_sub_layer
    q = ca.query_net(h).view(B, 1, H, Dh).transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q, cross_k, cross_v, attn_mask=cross_mask, dropout_p=0.0, scale=ca.scale,
    )
    attn = attn.transpose(1, 2).contiguous().view(B, 1, D)
    x = x + ca.out_projection(attn)

    # ---- FFN ----
    return x + layer.third_sub_layer(layer.layer_norm_3(x))


class DecoderStepGraph:
    """Captures a CUDA graph for one greedy decoder step at fixed ``(B, T_enc)``."""

    def __init__(
        self,
        model,
        B: int,
        T_enc: int,
        max_kv: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        decoder = model.transf_decoder._decoder
        embedding = model.transf_decoder._embedding
        head = model.log_softmax
        layers = decoder.layers
        L = len(layers)
        sa0 = layers[0].first_sub_layer
        H, D, Dh = sa0.num_heads, sa0.hidden_size, sa0.head_dim
        V = head.mlp.layer0.out_features

        self.B, self.T_enc, self.max_kv = B, T_enc, max_kv
        self.dtype, self.device = dtype, device

        # ---- Persistent input/output buffers ----
        self.token = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos_id = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos_idx = torch.zeros(1, dtype=torch.long, device=device)
        self.self_k = [torch.zeros(B, H, max_kv, Dh, dtype=dtype, device=device) for _ in range(L)]
        self.self_v = [torch.zeros(B, H, max_kv, Dh, dtype=dtype, device=device) for _ in range(L)]
        self.cross_k = [torch.zeros(B, H, T_enc, Dh, dtype=dtype, device=device) for _ in range(L)]
        self.cross_v = [torch.zeros(B, H, T_enc, Dh, dtype=dtype, device=device) for _ in range(L)]
        self.cross_mask = torch.zeros(B, 1, 1, T_enc, dtype=dtype, device=device)
        self.logits = torch.zeros(B, V, dtype=dtype, device=device)
        # Mask scaffolding (read-only inside the graph).
        self._arange = torch.arange(max_kv, device=device).view(1, 1, 1, -1)
        self._zero = torch.zeros((), dtype=dtype, device=device)
        self._neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)

        # Module refs (no parameters allocated; just rebind to graph buffers).
        self._layers = layers
        self._final_ln = decoder.final_layer_norm
        self._embed = embedding
        self._head = head

        # ---- Warmup + capture ----
        # 3 eager warmup runs prime the workspace allocator and any lazy
        # cuBLAS heuristics, then we capture in a fresh graph context.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run()

    def _run(self) -> None:
        """The single decoder step. Reads from / writes to persistent buffers."""
        # Self-attn mask: zero where position <= pos_idx, -inf otherwise.
        self_mask = torch.where(
            self._arange <= self.pos_idx.view(1, 1, 1, 1), self._zero, self._neg_inf
        )
        x = self._embed(self.token, self.pos_id)
        for i, layer in enumerate(self._layers):
            x = _step_layer(
                layer, x, self.pos_idx,
                self.self_k[i], self.self_v[i],
                self.cross_k[i], self.cross_v[i],
                self_mask, self.cross_mask,
            )
        x = self._final_ln(x)
        self.logits.copy_(self._head(x).squeeze(1))

    def load_prefill(
        self,
        self_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cross_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cross_mask: torch.Tensor,
    ) -> None:
        """Copy prefill state from the eager path into graph buffers.

        ``self_caches`` are concatenated KV up to ``prompt_len``; we copy them
        into the leading slice of our fixed-size buffer. The remaining slots
        are zero (and masked out) until the step loop fills them.
        """
        prompt_len = self_caches[0][0].size(2)
        for i, (sk, sv) in enumerate(self_caches):
            self.self_k[i].zero_()
            self.self_v[i].zero_()
            self.self_k[i][:, :, :prompt_len].copy_(sk)
            self.self_v[i][:, :, :prompt_len].copy_(sv)
        for i, (ck, cv) in enumerate(cross_caches):
            self.cross_k[i].copy_(ck)
            self.cross_v[i].copy_(cv)
        self.cross_mask.copy_(cross_mask)

    def step(self, token: torch.Tensor, pos: int) -> torch.Tensor:
        """Replay one decoder step. Returns logits ``[B, V]``."""
        self.token.copy_(token.view(self.B, 1))
        self.pos_id.fill_(pos)
        self.pos_idx.fill_(pos)
        self._graph.replay()
        return self.logits


def get_or_build_graph(model, B: int, T_enc: int, max_kv: int) -> DecoderStepGraph:
    """Return a cached :class:`DecoderStepGraph` for ``(B, T_enc, max_kv)``.

    Cache lives on ``model._step_graphs`` so it survives across calls. Capture
    is ~100–500 ms on A100; subsequent calls with the same shape are free.
    """
    cache = getattr(model, "_step_graphs", None)
    if cache is None:
        cache = {}
        model._step_graphs = cache
    key = (B, T_enc, max_kv)
    g = cache.get(key)
    if g is None:
        device = next(model.parameters()).device
        dtype = next(model.transf_decoder.parameters()).dtype
        g = DecoderStepGraph(model, B=B, T_enc=T_enc, max_kv=max_kv, dtype=dtype, device=device)
        cache[key] = g
    return g
