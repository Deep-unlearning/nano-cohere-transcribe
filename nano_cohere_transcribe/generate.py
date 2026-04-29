"""Greedy autoregressive decoding for the Cohere ASR decoder.

Keeps two KV caches per layer:

* self-attention cache: grows by one position per decoded token.
* cross-attention cache: computed once from the encoder output, reused every step.

The initial "prefill" pass runs the full language prompt through the decoder in
one shot to populate both caches; each subsequent step feeds a single token.
"""
from __future__ import annotations

from typing import Optional

import torch

from ._graph import get_or_build_graph
from .tokenizer import SUPPORTED_LANGUAGES, CohereTokenizer


@torch.inference_mode()
def greedy_generate(
    model,
    encoder_hidden_states: torch.Tensor,
    encoder_lengths: torch.Tensor,
    language: str,
    punctuation: bool = True,
    max_new_tokens: int = 256,
    tokenizer: Optional[CohereTokenizer] = None,
    use_cuda_graph: bool = True,
) -> list[list[int]]:
    """Greedy decode; returns per-row token id lists with prompt and trailing EOS stripped."""
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("greedy_generate needs a CohereTokenizer (on model.tokenizer or as arg).")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; supported: {sorted(SUPPORTED_LANGUAGES)}")

    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    B = encoder_hidden_states.size(0)
    enc_T = encoder_hidden_states.size(1)

    # Cross-attn mask: mask padded encoder positions. Same tensor every step.
    enc_pos = torch.arange(enc_T, device=device).unsqueeze(0).expand(B, -1)
    enc_valid = enc_pos < encoder_lengths.to(device=device).unsqueeze(1)
    neg_inf = torch.full((), float("-inf"), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    cross_mask = torch.where(enc_valid[:, None, None, :], zero, neg_inf)  # [B, 1, 1, T_enc]

    prompt_ids = tokenizer.build_prompt(language, punctuation=punctuation)
    prompt = torch.tensor([prompt_ids] * B, dtype=torch.long, device=device)  # [B, P]
    eos_id = tokenizer.eos_id

    generated = torch.zeros(B, max_new_tokens, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # --- Prefill over the prompt ---
    h, self_caches, cross_caches = _decoder_forward(
        model,
        input_ids=prompt,
        positions=torch.arange(prompt.size(1), device=device).unsqueeze(0).expand(B, -1),
        encoder_hidden_states=encoder_hidden_states,
        cross_mask=cross_mask,
        self_caches=None,
        cross_caches=None,
        use_causal_self_mask=True,
    )
    next_token = model.log_softmax(h[:, -1:, :]).squeeze(1).argmax(dim=-1)  # [B]
    generated[:, 0] = next_token
    finished |= next_token == eos_id

    # --- Step loop ---
    # CUDA graph the per-step decoder if we're on a GPU and not in fp32 (graph
    # capture works with fp32 too, but we don't keep a cuBLAS-fp32-warm path).
    # Skip graphs at large B: each new (B, T_enc) triggers a fresh capture
    # (~100-500 ms). At big batches the per-step launch overhead is already
    # amortized across the batch, so the capture cost outweighs the win
    # — measured on earnings22 short-form bs=64 (25.2 s -> 26.7 s with graphs).
    GRAPH_MAX_BATCH = 16
    use_graph = (
        use_cuda_graph
        and device.type == "cuda"
        and dtype != torch.float32
        and max_new_tokens > 1
        and B <= GRAPH_MAX_BATCH
    )
    graph = None
    if use_graph:
        # Buffer must hold prompt + generated tokens. +8 slack for safety.
        max_kv = prompt.size(1) + max_new_tokens + 8
        graph = get_or_build_graph(model, B=B, T_enc=enc_T, max_kv=max_kv)
        graph.load_prefill(self_caches, cross_caches, cross_mask)

    step = 1
    past_len = prompt.size(1)
    while step < max_new_tokens and not finished.all():
        if graph is not None:
            logits = graph.step(next_token, past_len)
            next_token = logits.argmax(dim=-1)
        else:
            tok = next_token.unsqueeze(1)  # [B, 1]
            pos = torch.full((B, 1), past_len, dtype=torch.long, device=device)
            h, self_caches, _ = _decoder_forward(
                model,
                input_ids=tok,
                positions=pos,
                encoder_hidden_states=encoder_hidden_states,
                cross_mask=cross_mask,
                self_caches=self_caches,
                cross_caches=cross_caches,
                use_causal_self_mask=False,
            )
            next_token = model.log_softmax(h[:, -1:, :]).squeeze(1).argmax(dim=-1)
        # After EOS, keep writing EOS for rows that finished so we don't overwrite.
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        generated[:, step] = next_token
        finished |= next_token == eos_id
        past_len += 1
        step += 1

    out: list[list[int]] = []
    arr = generated[:, :step].cpu().tolist()
    for row in arr:
        if eos_id in row:
            row = row[: row.index(eos_id)]
        out.append(row)
    return out


def _decoder_forward(
    model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    cross_mask: torch.Tensor,
    self_caches: Optional[list],
    cross_caches: Optional[list],
    use_causal_self_mask: bool,
):
    B, T = input_ids.shape
    past_len = 0 if self_caches is None else self_caches[0][0].size(2)
    total_kv = past_len + T

    if use_causal_self_mask and T > 1:
        q_pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(1)
        k_pos = torch.arange(total_kv, device=input_ids.device).unsqueeze(0)
        neg_inf = torch.full((), float("-inf"), device=input_ids.device, dtype=encoder_hidden_states.dtype)
        zero = torch.zeros((), device=input_ids.device, dtype=encoder_hidden_states.dtype)
        self_mask = torch.where(k_pos <= q_pos, zero, neg_inf).unsqueeze(0).unsqueeze(0)  # [1,1,T,total_kv]
    else:
        self_mask = None

    h, new_self, new_cross = model.transf_decoder(
        input_ids=input_ids,
        positions=positions,
        encoder_hidden_states=encoder_hidden_states,
        self_attn_mask=self_mask,
        cross_attn_mask=cross_mask,
        self_kv_caches=self_caches,
        cross_kv_caches=cross_caches,
    )
    return h, new_self, new_cross
