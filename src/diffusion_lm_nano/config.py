"""Typed TOML configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Literal

ScheduleKind = Literal["linear", "cosine", "power"]
DecodeStrategy = Literal["fixed_fraction", "confidence", "fixed_order", "adaptive"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 258
    max_length: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    mask_token_id: int = 257
    pad_token_id: int = 256

    def validate(self) -> None:
        if self.vocab_size < 3:
            raise ValueError("model.vocab_size must be at least 3")
        if self.max_length < 1 or self.d_model < 1 or self.n_layers < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.d_ff < self.d_model:
            raise ValueError("model.d_ff must be at least model.d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        for name, value in (
            ("mask_token_id", self.mask_token_id),
            ("pad_token_id", self.pad_token_id),
        ):
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"model.{name} must be within the vocabulary")
        if self.mask_token_id == self.pad_token_id:
            raise ValueError("mask and pad token ids must differ")


@dataclass(frozen=True, slots=True)
class AdaptiveDecodeConfig:
    min_passes: int = 1
    entropy_threshold: float = 0.0
    confidence_delta: float = 0.0
    confidence_patience: int = 0

    @property
    def entropy_rule_enabled(self) -> bool:
        return self.entropy_threshold > 0.0

    @property
    def confidence_rule_enabled(self) -> bool:
        return self.confidence_patience > 0

    @property
    def early_stopping_enabled(self) -> bool:
        return self.entropy_rule_enabled or self.confidence_rule_enabled

    def validate(self, max_passes: int | None = None) -> None:
        if isinstance(self.min_passes, bool) or not isinstance(self.min_passes, int):
            raise ValueError("adaptive min_passes must be an integer")
        if self.min_passes < 1:
            raise ValueError("adaptive min_passes must be positive")
        if isinstance(self.entropy_threshold, bool) or not isinstance(
            self.entropy_threshold, (int, float)
        ):
            raise ValueError("adaptive entropy_threshold must be numeric")
        if not math.isfinite(self.entropy_threshold):
            raise ValueError("adaptive entropy_threshold must be finite")
        if not 0.0 <= self.entropy_threshold <= 1.0:
            raise ValueError("adaptive entropy_threshold must be in [0, 1]")
        if isinstance(self.confidence_delta, bool) or not isinstance(
            self.confidence_delta, (int, float)
        ):
            raise ValueError("adaptive confidence_delta must be numeric")
        if not math.isfinite(self.confidence_delta):
            raise ValueError("adaptive confidence_delta must be finite")
        if not 0.0 <= self.confidence_delta <= 1.0:
            raise ValueError("adaptive confidence_delta must be in [0, 1]")
        if isinstance(self.confidence_patience, bool) or not isinstance(
            self.confidence_patience, int
        ):
            raise ValueError("adaptive confidence_patience must be an integer")
        if self.confidence_patience < 0:
            raise ValueError("adaptive confidence_patience must be non-negative")
        if max_passes is None:
            return
        if self.min_passes > max_passes:
            raise ValueError("adaptive min_passes cannot exceed sampling_steps")
        if self.entropy_rule_enabled and self.min_passes >= max_passes:
            raise ValueError(
                "adaptive entropy_threshold cannot trigger before sampling_steps"
            )
        available_improvements = max_passes - max(2, self.min_passes)
        if (
            self.confidence_rule_enabled
            and self.confidence_patience > available_improvements
        ):
            raise ValueError(
                "adaptive confidence_patience cannot trigger within sampling_steps"
            )


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    schedule: ScheduleKind = "cosine"
    schedule_power: float = 2.0
    train_min_t: float = 0.001
    sampling_steps: int = 16
    strategy: DecodeStrategy = "confidence"
    temperature: float = 1.0
    adaptive_min_passes: int = 1
    adaptive_entropy_threshold: float = 0.0
    adaptive_confidence_delta: float = 0.0
    adaptive_confidence_patience: int = 0

    def adaptive_decode_config(self) -> AdaptiveDecodeConfig:
        return AdaptiveDecodeConfig(
            min_passes=self.adaptive_min_passes,
            entropy_threshold=self.adaptive_entropy_threshold,
            confidence_delta=self.adaptive_confidence_delta,
            confidence_patience=self.adaptive_confidence_patience,
        )

    def validate(self) -> None:
        if self.schedule not in {"linear", "cosine", "power"}:
            raise ValueError(f"unknown diffusion.schedule {self.schedule!r}")
        if self.schedule_power <= 0:
            raise ValueError("diffusion.schedule_power must be positive")
        if not 0.0 <= self.train_min_t < 1.0:
            raise ValueError("diffusion.train_min_t must be in [0, 1)")
        if self.sampling_steps < 1:
            raise ValueError("diffusion.sampling_steps must be positive")
        if self.strategy not in {
            "fixed_fraction",
            "confidence",
            "fixed_order",
            "adaptive",
        }:
            raise ValueError(f"unknown diffusion.strategy {self.strategy!r}")
        if self.temperature < 0:
            raise ValueError("diffusion.temperature must be non-negative")
        self.adaptive_decode_config().validate(self.sampling_steps)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 7
    steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    log_every: int = 50
    device: str = "auto"
    corpus_path: str | None = None
    validation_fraction: float = 0.1

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.log_every < 1:
            raise ValueError("train counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("train optimizer values are invalid")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("train.device must be auto, cpu, cuda, or mps")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("train.validation_fraction must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig

    def validate(self) -> None:
        self.model.validate()
        self.diffusion.validate()
        self.train.validate()

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        _reject_unknown(raw, {"model", "diffusion", "train"}, "root")
        config = cls(
            model=_load_section(ModelConfig, raw.get("model", {}), "model"),
            diffusion=_load_section(
                DiffusionConfig, raw.get("diffusion", {}), "diffusion"
            ),
            train=_load_section(TrainConfig, raw.get("train", {}), "train"),
        )
        config.validate()
        return config


def _load_section(cls: type[Any], values: Any, section: str) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a TOML table")
    allowed = set(cls.__dataclass_fields__)
    _reject_unknown(values, allowed, section)
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"invalid {section} configuration: {error}") from error


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} key: {unknown[0]}")
