"""Iterative parallel unmasking for masked diffusion language models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn

from .config import AdaptiveDecodeConfig, DecodeStrategy
from .schedules import MaskSchedule

StepCallback = Callable[[int, Tensor, Tensor], None]
StoppingReason = Literal[
    "all_revealed",
    "entropy_threshold",
    "confidence_stall",
    "maximum_passes",
]


@dataclass(frozen=True, slots=True)
class SamplingStep:
    pass_index: int
    timestep: float
    unresolved_count: int
    mean_confidence: float | None
    normalized_entropy: float | None
    confidence_improvement: float | None
    reveal_count: int
    remaining_count: int
    adaptive_finalization: bool
    stopping_reason: StoppingReason | None


@dataclass(frozen=True, slots=True)
class SamplingResult:
    tokens: Tensor
    forward_passes: int
    stopping_reason: StoppingReason
    trajectory: tuple[SamplingStep, ...]


@torch.no_grad()
def sample(
    model: nn.Module,
    *,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    steps: int,
    strategy: DecodeStrategy = "confidence",
    temperature: float = 1.0,
    schedule: MaskSchedule | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    callback: StepCallback | None = None,
    adaptive_config: AdaptiveDecodeConfig | None = None,
) -> Tensor:
    """Generate fixed-length sequences by progressively revealing masked positions."""
    return sample_with_metadata(
        model,
        batch_size=batch_size,
        length=length,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        steps=steps,
        strategy=strategy,
        temperature=temperature,
        schedule=schedule,
        device=device,
        generator=generator,
        callback=callback,
        adaptive_config=adaptive_config,
        collect_diagnostics=False,
    ).tokens


@torch.no_grad()
def sample_with_metadata(
    model: nn.Module,
    *,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    steps: int,
    strategy: DecodeStrategy = "confidence",
    temperature: float = 1.0,
    schedule: MaskSchedule | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    callback: StepCallback | None = None,
    adaptive_config: AdaptiveDecodeConfig | None = None,
    collect_diagnostics: bool = True,
) -> SamplingResult:
    """Generate a sequence and return pass accounting and audit trajectory."""
    if batch_size < 1 or length < 1 or steps < 1:
        raise ValueError("batch_size, length, and steps must be positive")
    initial_tokens = torch.full(
        (batch_size, length), mask_token_id, dtype=torch.long, device=device
    )
    return sample_from_tokens_with_metadata(
        model,
        initial_tokens,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        steps=steps,
        strategy=strategy,
        temperature=temperature,
        schedule=schedule,
        device=device,
        generator=generator,
        callback=callback,
        adaptive_config=adaptive_config,
        collect_diagnostics=collect_diagnostics,
    )


@torch.no_grad()
def sample_from_tokens(
    model: nn.Module,
    initial_tokens: Tensor,
    *,
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    steps: int,
    strategy: DecodeStrategy = "confidence",
    temperature: float = 1.0,
    schedule: MaskSchedule | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    callback: StepCallback | None = None,
    start_timestep: float = 1.0,
    adaptive_config: AdaptiveDecodeConfig | None = None,
) -> Tensor:
    """Reveal mask tokens in an existing sequence with the sampler reveal loop."""
    return sample_from_tokens_with_metadata(
        model,
        initial_tokens,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        steps=steps,
        strategy=strategy,
        temperature=temperature,
        schedule=schedule,
        device=device,
        generator=generator,
        callback=callback,
        start_timestep=start_timestep,
        adaptive_config=adaptive_config,
        collect_diagnostics=False,
    ).tokens


@torch.no_grad()
def sample_from_tokens_with_metadata(
    model: nn.Module,
    initial_tokens: Tensor,
    *,
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    steps: int,
    strategy: DecodeStrategy = "confidence",
    temperature: float = 1.0,
    schedule: MaskSchedule | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    callback: StepCallback | None = None,
    start_timestep: float = 1.0,
    adaptive_config: AdaptiveDecodeConfig | None = None,
    collect_diagnostics: bool = True,
) -> SamplingResult:
    """Reveal masks and return the complete sequence with adaptive diagnostics."""
    if initial_tokens.ndim != 2 or steps < 1:
        raise ValueError("initial_tokens must be rank two and steps must be positive")
    if strategy not in {"fixed_fraction", "confidence", "fixed_order", "adaptive"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < start_timestep <= 1.0:
        raise ValueError("start_timestep must be in (0, 1]")
    if vocab_size < 3:
        raise ValueError("vocab_size must be at least three")
    if not 0 <= mask_token_id < vocab_size or not 0 <= pad_token_id < vocab_size:
        raise ValueError("special token ids must be within the vocabulary")
    if mask_token_id == pad_token_id:
        raise ValueError("mask and pad token ids must differ")

    resolved_adaptive_config = adaptive_config or AdaptiveDecodeConfig()
    if strategy == "adaptive":
        resolved_adaptive_config.validate(steps)
    tokens = initial_tokens.to(device=device, dtype=torch.long).clone()
    batch_size, length = tokens.shape
    trajectory: list[SamplingStep] = []
    previous_mean_confidence: float | None = None
    stalled_passes = 0
    stopping_reason: StoppingReason | None = None

    if not tokens.eq(mask_token_id).any():
        return SamplingResult(
            tokens=tokens,
            forward_passes=0,
            stopping_reason="all_revealed",
            trajectory=(),
        )

    for step in range(steps):
        masked = tokens.eq(mask_token_id)
        unresolved_count = int(masked.sum().item())
        if not unresolved_count:
            stopping_reason = "all_revealed"
            break
        time_value = start_timestep * (1.0 - step / steps)
        t = torch.full((batch_size,), time_value, device=tokens.device)
        logits = model(tokens, t)
        if logits.shape != (batch_size, length, vocab_size):
            raise ValueError("model returned unexpected logits shape")
        logits = logits.clone()
        logits[..., mask_token_id] = -torch.inf
        logits[..., pad_token_id] = -torch.inf
        predictions, confidence, probabilities = _draw_predictions(
            logits, temperature, generator
        )
        diagnostics_required = strategy == "adaptive" or collect_diagnostics
        mean_confidence: float | None = None
        mean_entropy: float | None = None
        confidence_improvement: float | None = None
        if diagnostics_required:
            mean_confidence = float(confidence[masked].float().mean().item())
            entropy = normalized_entropy(probabilities, vocab_size - 2)
            mean_entropy = float(entropy[masked].mean().item())
            confidence_improvement = (
                None
                if previous_mean_confidence is None
                else mean_confidence - previous_mean_confidence
            )

        adaptive_reason: StoppingReason | None = None
        pass_index = step + 1
        if (
            strategy == "adaptive"
            and pass_index >= resolved_adaptive_config.min_passes
            and pass_index < steps
        ):
            if (
                mean_entropy is not None
                and mean_entropy < resolved_adaptive_config.entropy_threshold
            ):
                adaptive_reason = "entropy_threshold"
            if (
                resolved_adaptive_config.confidence_patience
                and confidence_improvement is not None
            ):
                if (
                    confidence_improvement
                    < resolved_adaptive_config.confidence_delta
                ):
                    stalled_passes += 1
                else:
                    stalled_passes = 0
                if (
                    adaptive_reason is None
                    and stalled_passes
                    >= resolved_adaptive_config.confidence_patience
                ):
                    adaptive_reason = "confidence_stall"

        reveal = torch.zeros_like(masked)
        if adaptive_reason is not None:
            reveal = masked
        else:
            target_mask_fraction = (
                schedule.probability(t) if schedule is not None else t.clamp(0.0, 1.0)
            )
            steps_left = steps - step
            for batch_index in range(batch_size):
                positions = masked[batch_index].nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                target_remaining = int(
                    math.ceil(float(target_mask_fraction[batch_index].item() * length))
                )
                schedule_count = positions.numel() - target_remaining
                termination_count = math.ceil(positions.numel() / steps_left)
                count = max(schedule_count, termination_count)
                count = min(count, positions.numel())
                if count == positions.numel():
                    reveal[batch_index, positions] = True
                    continue
                if strategy in {"confidence", "adaptive"}:
                    scores = confidence[batch_index, positions]
                    chosen = positions[scores.topk(count).indices]
                elif strategy == "fixed_order":
                    chosen = positions[:count]
                else:
                    order = torch.randperm(
                        positions.numel(),
                        device=tokens.device,
                        generator=generator,
                    )
                    chosen = positions[order[:count]]
                reveal[batch_index, chosen] = True

        reveal_count = int(reveal.sum().item())
        tokens = torch.where(reveal, predictions, tokens)
        remaining_count = int(tokens.eq(mask_token_id).sum().item())
        step_stopping_reason: StoppingReason | None = adaptive_reason
        if not remaining_count and step_stopping_reason is None:
            step_stopping_reason = (
                "maximum_passes" if pass_index == steps else "all_revealed"
            )
        trajectory.append(
            SamplingStep(
                pass_index=pass_index,
                timestep=time_value,
                unresolved_count=unresolved_count,
                mean_confidence=mean_confidence,
                normalized_entropy=mean_entropy,
                confidence_improvement=confidence_improvement,
                reveal_count=reveal_count,
                remaining_count=remaining_count,
                adaptive_finalization=adaptive_reason is not None,
                stopping_reason=step_stopping_reason,
            )
        )
        if callback is not None:
            callback(
                pass_index,
                tokens.detach().clone(),
                confidence.detach().clone(),
            )
        if mean_confidence is not None:
            previous_mean_confidence = mean_confidence
        if step_stopping_reason is not None:
            stopping_reason = step_stopping_reason
            break

    if tokens.eq(mask_token_id).any():
        raise RuntimeError("sampler failed to reveal all positions")
    if stopping_reason is None:
        raise RuntimeError("sampler completed without a stopping reason")
    return SamplingResult(
        tokens=tokens,
        forward_passes=len(trajectory),
        stopping_reason=stopping_reason,
        trajectory=tuple(trajectory),
    )


def normalized_entropy(probabilities: Tensor, support_size: int) -> Tensor:
    """Return categorical entropy normalized to the configured token support."""
    if probabilities.ndim < 1:
        raise ValueError("probabilities must have a vocabulary dimension")
    if support_size < 1 or support_size > probabilities.shape[-1]:
        raise ValueError("support_size must fit the probability vocabulary")
    if support_size == 1:
        return torch.zeros(
            probabilities.shape[:-1],
            dtype=torch.float32,
            device=probabilities.device,
        )
    stable = probabilities.float()
    log_probabilities = torch.where(
        stable > 0,
        stable.log(),
        torch.zeros_like(stable),
    )
    entropy = -(stable * log_probabilities).sum(dim=-1)
    return (entropy / math.log(support_size)).clamp(0.0, 1.0)


def _draw_predictions(
    logits: Tensor,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if temperature == 0:
        probabilities = logits.softmax(dim=-1)
        return (
            logits.argmax(dim=-1),
            probabilities.max(dim=-1).values,
            probabilities,
        )
    probabilities = (logits / temperature).softmax(dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    predictions = torch.multinomial(flat, 1, generator=generator).reshape(
        logits.shape[:-1]
    )
    confidence = probabilities.gather(-1, predictions[..., None]).squeeze(-1)
    return predictions, confidence, probabilities
