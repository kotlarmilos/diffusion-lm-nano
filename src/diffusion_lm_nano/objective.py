"""Masked-token denoising objective."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .corruption import IGNORE_INDEX


def masked_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Average cross entropy only over tokens selected by forward corruption."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets shapes are incompatible")
    selected = targets.ne(IGNORE_INDEX)
    if not selected.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[selected], targets[selected])


def masked_accuracy(logits: Tensor, targets: Tensor) -> tuple[int, int]:
    selected = targets.ne(IGNORE_INDEX)
    total = int(selected.sum().item())
    if total == 0:
        return 0, 0
    correct = int((logits.argmax(dim=-1)[selected] == targets[selected]).sum().item())
    return correct, total

