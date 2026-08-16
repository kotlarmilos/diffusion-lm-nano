"""Mask-probability schedules for absorbing-state discrete diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class MaskSchedule(Protocol):
    def probability(self, t: "torch.Tensor") -> "torch.Tensor":
        """Return the marginal mask probability at normalized time t."""


@dataclass(frozen=True, slots=True)
class LinearSchedule:
    def probability(self, t: "torch.Tensor") -> "torch.Tensor":
        return t.clamp(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CosineSchedule:
    def probability(self, t: "torch.Tensor") -> "torch.Tensor":
        return (1.0 - torch_cos(t.clamp(0.0, 1.0) * (math.pi / 2.0))).clamp(
            0.0, 1.0
        )


@dataclass(frozen=True, slots=True)
class PowerSchedule:
    power: float = 2.0

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError("power must be positive")

    def probability(self, t: "torch.Tensor") -> "torch.Tensor":
        return t.clamp(0.0, 1.0).pow(self.power)


def torch_cos(value: "torch.Tensor") -> "torch.Tensor":
    return value.cos()


def make_schedule(name: str, power: float = 2.0) -> MaskSchedule:
    if name == "linear":
        return LinearSchedule()
    if name == "cosine":
        return CosineSchedule()
    if name == "power":
        return PowerSchedule(power)
    raise ValueError(f"unknown schedule {name!r}")

