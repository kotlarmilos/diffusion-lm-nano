"""Iterative parallel unmasking for masked diffusion language models."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor, nn

from .config import DecodeStrategy
from .schedules import MaskSchedule

StepCallback = Callable[[int, Tensor, Tensor], None]


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
) -> Tensor:
    """Generate fixed-length sequences by progressively revealing masked positions."""
    if batch_size < 1 or length < 1 or steps < 1:
        raise ValueError("batch_size, length, and steps must be positive")
    initial_tokens = torch.full(
        (batch_size, length), mask_token_id, dtype=torch.long, device=device
    )
    return sample_from_tokens(
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
) -> Tensor:
    """Reveal mask tokens in an existing sequence with the sampler reveal loop."""
    if initial_tokens.ndim != 2 or steps < 1:
        raise ValueError("initial_tokens must be rank two and steps must be positive")
    if strategy not in {"fixed_fraction", "confidence", "fixed_order"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < start_timestep <= 1.0:
        raise ValueError("start_timestep must be in (0, 1]")

    tokens = initial_tokens.to(device=device, dtype=torch.long).clone()
    batch_size, length = tokens.shape
    for step in range(steps):
        masked = tokens.eq(mask_token_id)
        if not masked.any():
            break
        time_value = start_timestep * (1.0 - step / steps)
        t = torch.full((batch_size,), time_value, device=device)
        logits = model(tokens, t)
        if logits.shape != (batch_size, length, vocab_size):
            raise ValueError("model returned unexpected logits shape")
        logits = logits.clone()
        logits[..., mask_token_id] = -torch.inf
        logits[..., pad_token_id] = -torch.inf
        predictions, confidence = _draw_predictions(logits, temperature, generator)
        target_mask_fraction = (
            schedule.probability(t) if schedule is not None else t.clamp(0.0, 1.0)
        )

        reveal = torch.zeros_like(masked)
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
            if strategy == "confidence":
                scores = confidence[batch_index, positions]
                chosen = positions[scores.topk(count).indices]
            elif strategy == "fixed_order":
                chosen = positions[:count]
            else:
                order = torch.randperm(
                    positions.numel(), device=device, generator=generator
                )
                chosen = positions[order[:count]]
            reveal[batch_index, chosen] = True
        tokens = torch.where(reveal, predictions, tokens)
        if callback is not None:
            callback(step + 1, tokens.detach().clone(), confidence.detach().clone())

    if tokens.eq(mask_token_id).any():
        raise RuntimeError("sampler failed to reveal all positions")
    return tokens


def _draw_predictions(
    logits: Tensor,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    if temperature == 0:
        probabilities = logits.softmax(dim=-1)
        return logits.argmax(dim=-1), probabilities.max(dim=-1).values
    probabilities = (logits / temperature).softmax(dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    predictions = torch.multinomial(flat, 1, generator=generator).reshape(
        logits.shape[:-1]
    )
    confidence = probabilities.gather(-1, predictions[..., None]).squeeze(-1)
    return predictions, confidence
