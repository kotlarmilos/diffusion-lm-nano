"""Bidirectional Transformer denoiser with continuous timestep conditioning."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, t: Tensor) -> Tensor:
        half = self.dimension // 2
        if half == 0:
            return t[:, None]
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = t[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class MaskedDiffusionTransformer(nn.Module):
    """Predict clean tokens from a partially masked sequence without causal masking."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        t: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.config.max_length:
            raise ValueError("sequence exceeds model.max_length")
        if t.ndim == 0:
            t = t.expand(batch)
        if t.shape != (batch,):
            raise ValueError("t must be scalar or have shape [batch]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = hidden + self.time_embedding(t.float())[:, None, :]
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.output(hidden)

