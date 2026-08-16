"""Forward corruption for masked discrete diffusion."""

from __future__ import annotations

import torch
from torch import Tensor

from .schedules import MaskSchedule

IGNORE_INDEX = -100


def corrupt_tokens(
    input_ids: Tensor,
    t: Tensor,
    schedule: MaskSchedule,
    mask_token_id: int,
    attention_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replace tokens with an absorbing mask and return sparse targets."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if t.ndim == 0:
        t = t.expand(input_ids.shape[0])
    if t.shape != (input_ids.shape[0],):
        raise ValueError("t must be scalar or have shape [batch]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    else:
        attention_mask = attention_mask.bool()

    probabilities = schedule.probability(t).to(
        device=input_ids.device, dtype=torch.float32
    )
    probabilities = probabilities[:, None].expand_as(input_ids)
    draws = torch.rand(
        input_ids.shape,
        device=input_ids.device,
        generator=generator,
        dtype=torch.float32,
    )
    selected = (draws < probabilities) & attention_mask
    selected = torch.where(probabilities <= 0, False, selected)
    selected = torch.where(probabilities >= 1, attention_mask, selected)
    corrupted = input_ids.masked_fill(selected, mask_token_id)
    targets = input_ids.masked_fill(~selected, IGNORE_INDEX)
    return corrupted, targets, selected

