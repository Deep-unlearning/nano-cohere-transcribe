"""Pure-PyTorch port of CohereLabs/cohere-transcribe-03-2026.

Mirrors the class hierarchy in ``modeling_cohere_asr.py`` so loading the
checkpoint is a straight ``load_state_dict(state_dict, strict=False)`` — no key
remapping. All transformers/generate/training scaffolding is dropped; the
forward pass is a single call through encoder -> proj -> decoder -> head.

Only inference is supported. Greedy autoregressive generation and its KV cache
live in ``generate.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mel import FilterbankFeatures


# ---------- Encoder ----------


class _MaskedConvSequential(nn.Sequential):
    """Sequential of Conv2d/ReLU that zeros invalid (padded) time positions.

    The channel shape here is ``(B, C, T, F)``; masking happens along the T
    dimension using a valid-length tensor that gets downsampled after every
    strided conv.
    """

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        cur = lengths.clone().long()
        mask = self._make_mask(x, cur)
        for layer in self:
            x = x * mask
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                pad = layer.padding
                k = layer.kernel_size[0]
                s = layer.stride[0]
                cur = (cur + pad[0] + pad[0] - k) // s + 1
                mask = self._make_mask(x, cur)
        return x * mask, cur

    @staticmethod
    def _make_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, _, t, f = x.shape
        m = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t) < lengths.unsqueeze(1)
        return m.unsqueeze(1).unsqueeze(-1).to(x.dtype)  # [B, 1, T, 1]


class ConvSubsampling(nn.Module):
    """3x stride-2 depthwise-striding subsampling -> linear to d_model."""

    def __init__(self, feat_in: int, conv_channels: int, d_model: int, subsampling_factor: int = 8):
        super().__init__()
        C = conv_channels
        self.conv = _MaskedConvSequential(
            nn.Conv2d(1, C, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C),
            nn.Conv2d(C, C, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C),
            nn.Conv2d(C, C, kernel_size=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(conv_channels * (feat_in // subsampling_factor), d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, n_mels, T] -> [B, 1, T, n_mels]
        x = x.transpose(1, 2).unsqueeze(1)
        x, lengths = self.conv(x, lengths)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        return self.out(x), lengths


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding materialized on demand.

    Produces a ``[1, 2L-1, d_model]`` tensor indexed by relative offset, used
    by :class:`RelPositionMultiHeadAttention` via the ``matrix_bd`` path.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._pe: Optional[torch.Tensor] = None

    def _build(self, length: int, device: torch.device, dtype: torch.dtype):
        needed = 2 * length - 1
        if self._pe is not None and self._pe.size(1) >= needed:
            if self._pe.device != device:
                self._pe = self._pe.to(device)
            if self._pe.dtype != dtype:
                self._pe = self._pe.to(dtype)
            return
        effective = max(length, self.max_len)
        positions = torch.arange(effective - 1, -effective, -1, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(positions.size(0), self.d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self._pe = pe.unsqueeze(0).to(dtype=dtype)

    def forward(self, x: torch.Tensor):
        self._build(x.size(1), x.device, x.dtype)
        L = x.size(1)
        center = self._pe.size(1) // 2 + 1
        pos_emb = self._pe[:, center - L : center + L - 1]
        return x, pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=(kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, d_model] -> [B, d_model, T]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPositionMultiHeadAttention(nn.Module):
    """Relative-position MHSA with per-layer u/v biases and rel-shift trick."""

    def __init__(self, n_head: int, n_feat: int, dropout: float = 0.0):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.scaling = self.d_k**-0.5
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x.size(0)
        q = self.linear_q(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        if pos_emb.size(0) == 1 and b > 1:
            pos_emb = pos_emb.expand(b, -1, -1)
        p = self.linear_pos(pos_emb).view(b, -1, self.h, self.d_k).transpose(1, 2)

        q_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        ac = torch.matmul(q_u, k.transpose(-1, -2))
        bd = torch.matmul(q_v, p.transpose(-1, -2))
        bd = self._rel_shift(bd)[:, :, :, : ac.size(-1)]
        scores = (ac + bd) * self.scaling

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), 0.0)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(out)


class ConformerLayer(nn.Module):
    """Macaron FFN -> MHSA -> DW-conv -> FFN -> LN block."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, att_mask=None, pad_mask=None):
        x = x + 0.5 * self.dropout(self.feed_forward1(self.norm_feed_forward1(x)))
        x = x + self.dropout(self.self_attn(self.norm_self_att(x), pos_emb, att_mask))
        x = x + self.dropout(self.conv(self.norm_conv(x), pad_mask=pad_mask))
        x = x + 0.5 * self.dropout(self.feed_forward2(self.norm_feed_forward2(x)))
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        feat_in: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_expansion_factor: int,
        conv_kernel_size: int,
        subsampling_conv_channels: int,
        subsampling_factor: int,
        pos_emb_max_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff = d_model * ff_expansion_factor
        self.pre_encode = ConvSubsampling(feat_in, subsampling_conv_channels, d_model, subsampling_factor)
        self.pos_enc = RelPositionalEncoding(d_model, pos_emb_max_len)
        self.layers = nn.ModuleList(
            [ConformerLayer(d_model, d_ff, n_heads, conv_kernel_size, dropout) for _ in range(n_layers)]
        )

    @staticmethod
    def _masks(lengths: torch.Tensor, max_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        pad = torch.arange(max_len, device=device).expand(lengths.size(0), -1) < lengths.unsqueeze(-1)  # [B, T]
        att = pad.unsqueeze(1) & pad.unsqueeze(2)  # [B, T, T]
        return ~pad, ~att  # invalid masks (True = masked)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        # The mel preprocessor runs in fp32 (STFT path); cast to whatever dtype
        # the Conv2d weights were loaded as so we don't hit a mixed-dtype conv.
        conv_dtype = self.pre_encode.conv[0].weight.dtype
        if features.dtype != conv_dtype:
            features = features.to(dtype=conv_dtype)
        x, lengths = self.pre_encode(features, lengths)
        x, pos_emb = self.pos_enc(x)
        pad_mask, att_mask = self._masks(lengths, x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, pos_emb, att_mask=att_mask, pad_mask=pad_mask)
        return x, lengths


# ---------- Decoder ----------


class FixedPositionalEncoding(nn.Module):
    """Sinusoidal PE, scaled by 1/sqrt(hidden). Stored as a persistent buffer."""

    def __init__(self, hidden_size: int, max_sequence_length: int = 1024):
        super().__init__()
        pe = torch.zeros(max_sequence_length, hidden_size)
        pos = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) / hidden_size * torch.arange(0.0, hidden_size, 2))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe.div_(math.sqrt(hidden_size))
        self.register_buffer("pos_enc", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pos_enc.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1)


class DecoderAttention(nn.Module):
    """MHSA / cross-attn primitive. KV cache is passed in explicitly by generate()."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_source: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q = self._heads(self.query_net(hidden_states))
        if cached_kv is not None and kv_source is None:
            # Self-attn with cache: compute new k/v for current tokens, concat.
            new_k = self._heads(self.key_net(hidden_states))
            new_v = self._heads(self.value_net(hidden_states))
            k = torch.cat([cached_kv[0], new_k], dim=2)
            v = torch.cat([cached_kv[1], new_v], dim=2)
        elif cached_kv is not None and kv_source is not None:
            # Cross-attn with cache: cache is computed once from encoder output, reuse.
            k, v = cached_kv
        else:
            src = hidden_states if kv_source is None else kv_source
            k = self._heads(self.key_net(src))
            v = self._heads(self.value_net(src))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)
        return self.out_projection(attn), (k, v)


class DecoderFeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, hidden_act: str = "relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        act = hidden_act.lower().replace("swish", "silu")
        if act == "relu":
            self.activation = F.relu
        elif act == "silu":
            self.activation = F.silu
        elif act == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported decoder hidden_act: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_out(self.activation(self.dense_in(x)))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, num_heads: int, hidden_act: str = "relu"):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = DecoderAttention(hidden_size, num_heads)  # self-attn
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = DecoderAttention(hidden_size, num_heads)  # cross-attn
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = DecoderFeedForward(hidden_size, inner_size, hidden_act=hidden_act)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        cross_attn_mask: Optional[torch.Tensor],
        self_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
        cross_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
    ):
        residual = x
        h = self.layer_norm_1(x)
        self_out, new_self_kv = self.first_sub_layer(
            h, kv_source=None, attn_mask=self_attn_mask, cached_kv=self_kv_cache
        )
        x = residual + self_out

        residual = x
        h = self.layer_norm_2(x)
        cross_out, new_cross_kv = self.second_sub_layer(
            h, kv_source=encoder_hidden_states, attn_mask=cross_attn_mask, cached_kv=cross_kv_cache
        )
        x = residual + cross_out

        residual = x
        x = residual + self.third_sub_layer(self.layer_norm_3(x))
        return x, new_self_kv, new_cross_kv


class TransformerDecoderCore(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, num_heads: int, num_layers: int, hidden_act: str = "relu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, inner_size, num_heads, hidden_act=hidden_act) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        cross_attn_mask: Optional[torch.Tensor],
        self_kv_caches: Optional[list],
        cross_kv_caches: Optional[list],
    ):
        new_self = []
        new_cross = []
        for i, layer in enumerate(self.layers):
            self_c = self_kv_caches[i] if self_kv_caches is not None else None
            cross_c = cross_kv_caches[i] if cross_kv_caches is not None else None
            hidden_states, sk, ck = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                self_kv_cache=self_c,
                cross_kv_cache=cross_c,
            )
            new_self.append(sk)
            new_cross.append(ck)
        return self.final_layer_norm(hidden_states), new_self, new_cross


class TransformerDecoderEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_sequence_length: int, padding_idx: int = 2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.token_embedding(input_ids) + self.position_embedding(positions))


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, inner_size: int, num_heads: int, num_layers: int,
                 max_sequence_length: int, hidden_act: str = "relu", padding_idx: int = 2):
        super().__init__()
        self._embedding = TransformerDecoderEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            padding_idx=padding_idx,
        )
        self._decoder = TransformerDecoderCore(
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_act=hidden_act,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        cross_attn_mask: Optional[torch.Tensor],
        self_kv_caches: Optional[list],
        cross_kv_caches: Optional[list],
    ):
        h = self._embedding(input_ids, positions)
        return self._decoder(
            h,
            encoder_hidden_states=encoder_hidden_states,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            self_kv_caches=self_kv_caches,
            cross_kv_caches=cross_kv_caches,
        )


# ---------- Head + top-level ----------


class TokenClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, log_softmax: bool = True):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer0 = nn.Linear(hidden_size, num_classes)
        self.use_log_softmax = log_softmax

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.mlp.layer0(hidden_states)
        if self.use_log_softmax:
            return torch.log_softmax(logits, dim=-1)
        return logits


class _PreprocessorContainer(nn.Module):
    """Named container so state_dict key ``preprocessor.featurizer.*`` matches upstream."""

    def __init__(self, featurizer: FilterbankFeatures):
        super().__init__()
        self.featurizer = featurizer


@dataclass
class CohereAsrConfig:
    vocab_size: int = 16384
    max_audio_clip_s: int = 35
    sample_rate: int = 16000
    # encoder
    enc_feat_in: int = 128
    enc_d_model: int = 1280
    enc_n_heads: int = 8
    enc_n_layers: int = 48
    enc_ff_expansion_factor: int = 4
    enc_conv_kernel_size: int = 9
    enc_subsampling_conv_channels: int = 256
    enc_subsampling_factor: int = 8
    enc_pos_emb_max_len: int = 5000
    # decoder
    dec_hidden_size: int = 1024
    dec_inner_size: int = 4096
    dec_num_heads: int = 8
    dec_num_layers: int = 8
    dec_max_sequence_length: int = 1024
    dec_hidden_act: str = "relu"
    dec_padding_idx: int = 2
    # head
    head_num_classes: int = 16384
    head_log_softmax: bool = True

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "CohereAsrConfig":
        enc = cfg["encoder"]
        dec = cfg["transf_decoder"]["config_dict"]
        head = cfg["head"]
        return cls(
            vocab_size=cfg["vocab_size"],
            max_audio_clip_s=cfg.get("max_audio_clip_s", 35),
            sample_rate=cfg.get("sample_rate", 16000),
            enc_feat_in=enc["feat_in"],
            enc_d_model=enc["d_model"],
            enc_n_heads=enc["n_heads"],
            enc_n_layers=enc["n_layers"],
            enc_ff_expansion_factor=enc["ff_expansion_factor"],
            enc_conv_kernel_size=enc["conv_kernel_size"],
            enc_subsampling_conv_channels=enc["subsampling_conv_channels"],
            enc_subsampling_factor=enc["subsampling_factor"],
            enc_pos_emb_max_len=enc["pos_emb_max_len"],
            dec_hidden_size=dec["hidden_size"],
            dec_inner_size=dec["inner_size"],
            dec_num_heads=dec["num_attention_heads"],
            dec_num_layers=dec["num_layers"],
            dec_max_sequence_length=dec["max_sequence_length"],
            dec_hidden_act=dec.get("hidden_act", "relu"),
            head_num_classes=head["num_classes"],
            head_log_softmax=bool(head.get("log_softmax", False)),
        )


class CohereAsr(nn.Module):
    """Top-level pure-PyTorch Cohere ASR model.

    Field names match the upstream checkpoint so `load_state_dict(..., strict=False)`
    loads every weight without key remapping.
    """

    def __init__(self, config: CohereAsrConfig):
        super().__init__()
        self.config = config

        self.preprocessor = _PreprocessorContainer(
            FilterbankFeatures(
                sample_rate=config.sample_rate,
                n_window_size=400,
                n_window_stride=160,
                n_fft=512,
                nfilt=config.enc_feat_in,
            )
        )
        self.encoder = ConformerEncoder(
            feat_in=config.enc_feat_in,
            d_model=config.enc_d_model,
            n_heads=config.enc_n_heads,
            n_layers=config.enc_n_layers,
            ff_expansion_factor=config.enc_ff_expansion_factor,
            conv_kernel_size=config.enc_conv_kernel_size,
            subsampling_conv_channels=config.enc_subsampling_conv_channels,
            subsampling_factor=config.enc_subsampling_factor,
            pos_emb_max_len=config.enc_pos_emb_max_len,
        )
        if config.enc_d_model != config.dec_hidden_size:
            self.encoder_decoder_proj = nn.Linear(config.enc_d_model, config.dec_hidden_size)
        else:
            self.encoder_decoder_proj = None

        self.transf_decoder = TransformerDecoderWrapper(
            vocab_size=config.head_num_classes,
            hidden_size=config.dec_hidden_size,
            inner_size=config.dec_inner_size,
            num_heads=config.dec_num_heads,
            num_layers=config.dec_num_layers,
            max_sequence_length=config.dec_max_sequence_length,
            hidden_act=config.dec_hidden_act,
            padding_idx=config.dec_padding_idx,
        )

        self.log_softmax = TokenClassifierHead(
            hidden_size=config.dec_hidden_size,
            num_classes=config.head_num_classes,
            log_softmax=config.head_log_softmax,
        )
        # Tie head weight to decoder token embedding (upstream does the same).
        self.log_softmax.mlp.layer0.weight = self.transf_decoder._embedding.token_embedding.weight

    # ---- High-level helpers ----

    @torch.no_grad()
    def warmup(self, duration_s: float = 1.0, batch_size: int = 1) -> None:
        """Run a dummy transcription so first real call doesn't pay graph capture cost.

        Pre-builds the CUDA graph for ``batch_size`` and the encoder output
        length corresponding to ``duration_s`` of audio. Subsequent calls
        with the same ``(B, T_enc)`` reuse the cached graph for free.
        """
        device = next(self.parameters()).device
        if device.type != "cuda":
            return
        n = int(self.config.sample_rate * duration_s)
        wavs = [torch.zeros(n, dtype=torch.float32, device=device) for _ in range(batch_size)]
        if batch_size == 1:
            self.transcribe(wavs[0], language="en", max_new_tokens=2)
        else:
            self.transcribe_batch(wavs, language="en", max_new_tokens=2, batch_size=batch_size)

    @torch.no_grad()
    def compute_features(self, waveform: torch.Tensor):
        """Return ``(mel_features, frame_lengths)`` for a mono waveform.

        ``waveform`` is ``[num_samples]`` or ``[B, num_samples]`` float.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        seq_len = torch.full((waveform.size(0),), waveform.size(1), dtype=torch.long, device=device)
        return self.preprocessor.featurizer(waveform, seq_len)

    @torch.no_grad()
    def encode(self, features: torch.Tensor, feat_lengths: torch.Tensor):
        """Run features through encoder + projection, return ``(enc_out, enc_lengths)``."""
        enc, enc_len = self.encoder(features, feat_lengths)
        if self.encoder_decoder_proj is not None:
            enc = self.encoder_decoder_proj(enc)
        return enc, enc_len

    @torch.no_grad()
    def transcribe(
        self,
        waveform: torch.Tensor,
        language: str = "en",
        punctuation: bool = True,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        long_form_threshold_s: float | None = None,
    ) -> str:
        """Transcribe a mono waveform. Auto-chunks audio longer than the model's budget.

        ``waveform``: 1-D torch tensor at ``config.sample_rate``.
        ``batch_size``: how many chunks to process together during long-form decode
        (no effect for short audio — there's only one chunk). Bigger values speed up
        long audio at the cost of GPU memory.
        ``long_form_threshold_s``: clips longer than this are split into chunks.
        Defaults to ``max_audio_clip_s - 5`` (matches the reference fast-path threshold).
        """
        return self.transcribe_batch(
            [waveform],
            language=language,
            punctuation=punctuation,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            long_form_threshold_s=long_form_threshold_s,
        )[0]

    @torch.no_grad()
    def transcribe_batch(
        self,
        waveforms: list,
        language: str = "en",
        punctuation: bool = True,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        long_form_threshold_s: float | None = None,
    ) -> list[str]:
        """Transcribe a list of mono waveforms.

        Each waveform is a 1-D ``torch.Tensor`` or ``np.ndarray`` at
        ``config.sample_rate``. Long clips are split into chunks; all chunks
        across all input waveforms are sorted by duration (descending) and
        processed in batches of ``batch_size`` for efficient padding.
        """
        import numpy as np
        from collections import defaultdict

        from .chunk import (
            get_chunk_separator,
            join_chunk_texts,
            split_audio_chunks_energy,
        )

        if not waveforms:
            return []

        sr = self.config.sample_rate
        if long_form_threshold_s is None:
            long_form_threshold_s = float(self.config.max_audio_clip_s) - 5.0

        # Flatten (sample_idx, chunk_idx, chunk_tensor) triples.
        all_chunks: list[torch.Tensor] = []
        owners: list[tuple[int, int]] = []
        for si, w in enumerate(waveforms):
            t = w if isinstance(w, torch.Tensor) else torch.as_tensor(np.asarray(w))
            t = t.reshape(-1).to(dtype=torch.float32)
            duration_s = t.shape[0] / sr
            if duration_s <= long_form_threshold_s:
                all_chunks.append(t)
                owners.append((si, 0))
            else:
                arr = t.detach().to("cpu").numpy()
                pieces = split_audio_chunks_energy(
                    waveform=arr,
                    sample_rate=sr,
                    max_audio_clip_s=float(self.config.max_audio_clip_s),
                    overlap_chunk_second=5.0,
                    min_energy_window_samples=1600,
                )
                for ci, c in enumerate(pieces):
                    all_chunks.append(torch.from_numpy(c))
                    owners.append((si, ci))

        chunk_texts = self._transcribe_chunks_batched(
            all_chunks, language=language, punctuation=punctuation,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )

        # Regroup per input waveform and join.
        groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for (si, ci), txt in zip(owners, chunk_texts):
            groups[si].append((ci, txt))
        sep = get_chunk_separator(language)
        out = [""] * len(waveforms)
        for si, items in groups.items():
            items.sort(key=lambda x: x[0])
            out[si] = join_chunk_texts([t for _, t in items], separator=sep)
        return out

    @torch.no_grad()
    def _transcribe_chunks_batched(
        self,
        chunks: list[torch.Tensor],
        language: str,
        punctuation: bool,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        """Transcribe a flat list of <=max_audio_clip_s chunks, returning texts in input order."""
        from .generate import greedy_generate

        if not chunks:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Process longest chunks first so every batch has similarly-sized tensors
        # (minimizes padding waste).
        order = sorted(range(len(chunks)), key=lambda i: int(chunks[i].shape[0]), reverse=True)
        texts: list[str] = [""] * len(chunks)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch_waves = [chunks[i] for i in batch_idx]
            enc, enc_len = self._encode_padded_batch(batch_waves)
            ids_list = greedy_generate(
                self,
                encoder_hidden_states=enc,
                encoder_lengths=enc_len,
                language=language,
                punctuation=punctuation,
                max_new_tokens=max_new_tokens,
            )
            for i, ids in zip(batch_idx, ids_list):
                texts[i] = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return texts

    @torch.no_grad()
    def _encode_padded_batch(self, waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad waveforms to the longest in the batch and run features + encoder."""
        device = next(self.parameters()).device
        max_len = max(int(w.shape[0]) for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len, dtype=torch.float32, device=device)
        lens = torch.zeros(len(waveforms), dtype=torch.long, device=device)
        for i, w in enumerate(waveforms):
            n = int(w.shape[0])
            padded[i, :n] = w.to(device=device, dtype=torch.float32)
            lens[i] = n
        feats, feat_len = self.preprocessor.featurizer(padded, lens)
        enc, enc_len = self.encoder(feats, feat_len)
        if self.encoder_decoder_proj is not None:
            enc = self.encoder_decoder_proj(enc)
        return enc, enc_len
