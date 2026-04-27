"""Static-shape wrapper around one Conformer encoder block for autokernel's profile.py.

Run:
    uv run profile.py \\
        --model autokernel_models/cohere_encoder_block.py \\
        --class-name EncoderBlock \\
        --input-shape 8,437,1280 \\
        --dtype bfloat16

Why 437: 35 s audio @ 16 kHz, 10 ms hop, 8x conv subsampling → ~437 frames.
Why 1280: encoder d_model.
The pos_emb and masks are precomputed in __init__ so forward() takes a single
tensor — matching profile.py's contract.
"""
from __future__ import annotations

import torch
from torch import nn

from nano_cohere_transcribe.model import (
    CohereAsrConfig, ConformerLayer, RelPositionalEncoding,
)


class EncoderBlock(nn.Module):
    def __init__(self, batch_size: int = 8, seq_len: int = 437):
        super().__init__()
        cfg = CohereAsrConfig()
        d_model = cfg.enc_d_model
        d_ff = d_model * cfg.enc_ff_expansion_factor
        self.layer = ConformerLayer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=cfg.enc_n_heads,
            conv_kernel_size=cfg.enc_conv_kernel_size,
            dropout=0.0,
        )
        # Precompute relative positional embedding for the fixed seq_len.
        pos = RelPositionalEncoding(d_model, max_len=cfg.enc_pos_emb_max_len)
        # Build with a dummy input of the right shape so _pe is materialized.
        dummy = torch.zeros(1, seq_len, d_model)
        pos._build(dummy.size(1), dummy.device, dummy.dtype)
        # The block's MHA expects pos_emb of length 2L-1 sliced — match what the
        # encoder does when it calls self.pos_enc(x): RelPositionalEncoding.forward
        # returns the full [1, 2L-1, d_model] slab.
        self.register_buffer("pos_emb", pos._pe[:, : 2 * seq_len - 1, :].clone(), persistent=False)
        # No padding in our profiling scenario → both masks are None.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x, self.pos_emb, att_mask=None, pad_mask=None)
