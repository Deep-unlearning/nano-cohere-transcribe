"""Static-shape wrapper around one decoder step (1 new token, prefilled caches).

Run:
    uv run profile.py \\
        --model autokernel_models/cohere_decoder_step.py \\
        --class-name DecoderStep \\
        --input-shape 8,1,1024 \\
        --dtype bfloat16

Models the autoregressive hot path: input is a single token's hidden state, the
self-attention KV cache holds `kv_len` previously generated tokens, and the
cross-attention KV cache is prefilled from a `enc_len`-frame encoder pass. All
of those are stashed as buffers in __init__ so forward() takes one tensor —
matching profile.py's contract.

Tweak DEFAULTS at top to match the regime you want to profile (decoding step
mid-sentence vs. early, single 35 s chunk vs. longer encoder context).
"""
from __future__ import annotations

import torch
from torch import nn

from nano_cohere_transcribe.model import CohereAsrConfig, TransformerDecoderLayer

# Defaults: bs=8 chunks, 128 self-attn cache tokens (mid-utterance),
# 437 cross-attn frames (35 s encoder output).
BATCH = 8
KV_LEN = 128
ENC_LEN = 437


class DecoderStep(nn.Module):
    def __init__(self, batch_size: int = BATCH, kv_len: int = KV_LEN, enc_len: int = ENC_LEN):
        super().__init__()
        cfg = CohereAsrConfig()
        D = cfg.dec_hidden_size
        H = cfg.dec_num_heads
        Dh = D // H
        self.layer = TransformerDecoderLayer(
            hidden_size=D,
            inner_size=cfg.dec_inner_size,
            num_heads=H,
            hidden_act=cfg.dec_hidden_act,
        )
        # Encoder hidden states (already projected to decoder hidden_size — that
        # projection lives outside the decoder layer in the real model).
        self.register_buffer(
            "encoder_hidden_states", torch.randn(batch_size, enc_len, D), persistent=False
        )
        # Self-attention KV cache: [B, H, kv_len, Dh] each.
        self.register_buffer("self_k", torch.randn(batch_size, H, kv_len, Dh), persistent=False)
        self.register_buffer("self_v", torch.randn(batch_size, H, kv_len, Dh), persistent=False)
        # Cross-attention KV cache: [B, H, enc_len, Dh] each (prefilled, reused).
        self.register_buffer("cross_k", torch.randn(batch_size, H, enc_len, Dh), persistent=False)
        self.register_buffer("cross_v", torch.randn(batch_size, H, enc_len, Dh), persistent=False)
        # No masks needed: causal future is empty (single new token), encoder is full.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` is the next-token hidden state of shape [B, 1, D].
        out, _, _ = self.layer(
            x,
            encoder_hidden_states=self.encoder_hidden_states,
            self_attn_mask=None,
            cross_attn_mask=None,
            self_kv_cache=(self.self_k, self.self_v),
            cross_kv_cache=(self.cross_k, self.cross_v),
        )
        return out
