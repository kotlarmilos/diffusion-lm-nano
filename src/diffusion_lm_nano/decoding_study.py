"""Run reveal budget and ordering studies for masked diffusion decoding."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .artifacts import record_artifact
from .config import DiffusionConfig, ModelConfig, ProjectConfig
from .corruption import corrupt_tokens
from .model import MaskedDiffusionTransformer
from .sampler import sample_from_tokens
from .schedules import MaskSchedule, make_schedule
from .tokenizer_data import (
    BinaryTokenDataset,
    ByteTokenizer,
    GPT2NanoTokenizerAdapter,
    JsonlTokenDataset,
    Tokenizer,
)
from .train import select_device

OrderingStrategy = Literal["confidence", "fixed_order", "random"]

DEFAULT_STEP_BUDGETS = (1, 2, 3, 4, 6, 12)
DEFAULT_ORDERING_STRATEGIES: tuple[OrderingStrategy, ...] = (
    "confidence",
    "fixed_order",
)
DEFAULT_TEXT = "parallel token updates"
DEFAULT_INITIAL_TIMESTEP = 0.75

_ORDERING_ALIASES = {
    "confidence": "confidence",
    "fixed_order": "fixed_order",
    "fixed-order": "fixed_order",
    "fixed": "fixed_order",
    "random": "random",
    "fixed_fraction": "random",
}
_SAMPLER_STRATEGY = {
    "confidence": "confidence",
    "fixed_order": "fixed_order",
    "random": "fixed_fraction",
}


@dataclass(frozen=True, slots=True)
class TokenMetricCounts:
    correct: int
    total: int
    sequences: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    def merge(self, other: "TokenMetricCounts") -> "TokenMetricCounts":
        return TokenMetricCounts(
            correct=self.correct + other.correct,
            total=self.total + other.total,
            sequences=self.sequences + other.sequences,
        )


def parse_step_budgets(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
        try:
            values = [int(part) for part in parts]
        except ValueError as error:
            raise ValueError("step budgets must be integers") from error
    else:
        values = [int(value) for value in raw]
    if not values:
        raise ValueError("at least one step budget is required")
    if any(value < 1 for value in values):
        raise ValueError("step budgets must be positive")
    if len(set(values)) != len(values):
        raise ValueError("step budgets must be unique")
    return tuple(sorted(values))


def parse_ordering_strategies(raw: str | Iterable[str]) -> tuple[OrderingStrategy, ...]:
    values = raw.replace(",", " ").split() if isinstance(raw, str) else list(raw)
    if not values:
        raise ValueError("at least one ordering strategy is required")
    parsed: list[OrderingStrategy] = []
    for value in values:
        key = value.strip().lower()
        try:
            strategy = cast(OrderingStrategy, _ORDERING_ALIASES[key])
        except KeyError as error:
            raise ValueError(f"unknown ordering strategy {value!r}") from error
        if strategy not in parsed:
            parsed.append(strategy)
    return tuple(parsed)


def accumulate_token_metrics(
    predictions: Tensor,
    references: Tensor,
    pad_token_id: int,
    evaluation_mask: Tensor | None = None,
) -> TokenMetricCounts:
    if predictions.shape != references.shape:
        raise ValueError("predictions and references must have the same shape")
    active = references.ne(pad_token_id)
    if evaluation_mask is not None:
        if evaluation_mask.shape != references.shape:
            raise ValueError("evaluation_mask must match references")
        active = active & evaluation_mask.bool()
    total = int(active.sum().item())
    correct = int((predictions[active] == references[active]).sum().item())
    return TokenMetricCounts(correct=correct, total=total, sequences=references.shape[0])


def merge_token_metrics(counts: Iterable[TokenMetricCounts]) -> TokenMetricCounts:
    merged = TokenMetricCounts(correct=0, total=0, sequences=0)
    for item in counts:
        merged = merged.merge(item)
    return merged


def best_by_steps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        accuracy = row.get("token_accuracy")
        if accuracy is None:
            continue
        steps = int(row["steps"])
        current = best.get(steps)
        if current is None or float(accuracy) > float(current["token_accuracy"]):
            best[steps] = {
                "steps": steps,
                "ordering_strategy": row["ordering_strategy"],
                "token_accuracy": accuracy,
                "network_forward_passes": row["network_forward_passes"],
            }
    return [best[steps] for steps in sorted(best)]


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("token_accuracy") is not None
    ]
    candidates.sort(
        key=lambda row: (
            int(row["network_forward_passes"]),
            -float(row["token_accuracy"]),
        )
    )
    frontier: list[dict[str, Any]] = []
    best_accuracy = float("-inf")
    for row in candidates:
        accuracy = float(row["token_accuracy"])
        if accuracy > best_accuracy:
            frontier.append(
                {
                    "steps": row["steps"],
                    "ordering_strategy": row["ordering_strategy"],
                    "token_accuracy": row["token_accuracy"],
                    "network_forward_passes": row["network_forward_passes"],
                }
            )
            best_accuracy = accuracy
    return frontier


def compare_orderings(
    rows: Sequence[dict[str, Any]],
    *,
    primary: OrderingStrategy = "confidence",
    baseline: OrderingStrategy = "fixed_order",
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["steps"]), {})[str(row["ordering_strategy"])] = row
    comparison = []
    for steps in sorted(by_step):
        primary_row = by_step[steps].get(primary)
        baseline_row = by_step[steps].get(baseline)
        if primary_row is None or baseline_row is None:
            continue
        primary_accuracy = primary_row.get("token_accuracy")
        baseline_accuracy = baseline_row.get("token_accuracy")
        delta = (
            None
            if primary_accuracy is None or baseline_accuracy is None
            else float(primary_accuracy) - float(baseline_accuracy)
        )
        comparison.append(
            {
                "steps": steps,
                "primary": primary,
                "baseline": baseline,
                "primary_token_accuracy": primary_accuracy,
                "baseline_token_accuracy": baseline_accuracy,
                "accuracy_delta": delta,
            }
        )
    return comparison


def summarize_step_budget_hypothesis(
    rows: Sequence[dict[str, Any]],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    best = best_by_steps(rows)
    if not best:
        return {"status": "no_active_tokens"}
    accuracies = [float(row["token_accuracy"]) for row in best]
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    knee_index = next(
        index
        for index, row in enumerate(best)
        if float(row["token_accuracy"]) >= max_accuracy - tolerance
    )
    knee = best[knee_index]["steps"]
    monotonic = all(
        accuracies[index] + 1e-12 >= accuracies[index - 1]
        for index in range(1, len(accuracies))
    )
    monotonic_to_knee = all(
        accuracies[index] + 1e-12 >= accuracies[index - 1]
        for index in range(1, knee_index + 1)
    )
    if len(best) < 2:
        status = "insufficient_budgets"
    elif max_accuracy <= min_accuracy + 1e-12:
        status = "flat_development_scale"
    elif monotonic_to_knee and int(knee) < int(best[-1]["steps"]):
        status = "knee_observed_development_scale"
    elif monotonic and int(knee) < int(best[-1]["steps"]):
        status = "supported_development_scale"
    elif monotonic:
        status = "no_saturation_observed_development_scale"
    elif max_accuracy > accuracies[0] + 1e-12:
        status = "nonmonotonic_improvement_development_scale"
    else:
        status = "not_supported_development_scale"
    return {
        "status": status,
        "monotonic_non_decreasing": monotonic,
        "monotonic_to_knee": monotonic_to_knee,
        "max_token_accuracy": max_accuracy,
        "knee_steps": knee,
        "tolerance": tolerance,
    }


def summarize_ordering_hypothesis(
    comparison: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    deltas = [
        float(row["accuracy_delta"])
        for row in comparison
        if row.get("accuracy_delta") is not None
    ]
    if not deltas:
        status = "insufficient_orderings"
    elif all(delta >= 0.0 for delta in deltas) and any(delta > 0.0 for delta in deltas):
        status = "supported_development_scale"
    elif all(delta == 0.0 for delta in deltas):
        status = "tied_development_scale"
    else:
        status = "not_supported_development_scale"
    return {
        "status": status,
        "min_accuracy_delta": min(deltas) if deltas else None,
        "max_accuracy_delta": max(deltas) if deltas else None,
    }


@torch.no_grad()
def run_budget_sweep(
    model: nn.Module,
    reference_batches: Sequence[Tensor],
    *,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    step_budgets: Iterable[int],
    ordering_strategies: Iterable[OrderingStrategy],
    seed: int,
    device: torch.device,
    schedule: MaskSchedule | None = None,
    initial_timestep: float = DEFAULT_INITIAL_TIMESTEP,
) -> list[dict[str, Any]]:
    budgets = parse_step_budgets(step_budgets)
    strategies = parse_ordering_strategies(ordering_strategies)
    if not reference_batches:
        raise ValueError("at least one reference batch is required")
    if not 0.0 < initial_timestep <= 1.0:
        raise ValueError("initial timestep must be in (0, 1]")
    schedule = schedule or make_schedule(
        diffusion_config.schedule,
        diffusion_config.schedule_power,
    )
    prepared_batches = []
    for batch_index, reference in enumerate(reference_batches):
        if reference.ndim != 2:
            raise ValueError("reference batches must have shape [batch, sequence]")
        reference = reference.to(device)
        attention = reference.ne(model_config.pad_token_id)
        t = torch.full((reference.shape[0],), initial_timestep, device=device)
        corruption_generator = _make_generator(device, seed + batch_index)
        corrupted, _targets, selected = corrupt_tokens(
            reference,
            t,
            schedule,
            model_config.mask_token_id,
            attention,
            corruption_generator,
        )
        if not selected.any():
            raise ValueError("initial corruption selected no tokens")
        prepared_batches.append((reference, corrupted, selected))
    rows = []
    for strategy in strategies:
        sampler_strategy = _SAMPLER_STRATEGY[strategy]
        for budget_index, steps in enumerate(budgets):
            batch_counts: list[TokenMetricCounts] = []
            network_forward_passes = 0
            for batch_index, (reference, corrupted, selected) in enumerate(prepared_batches):
                generator_seed = seed + 100_000 + budget_index * 10_000 + batch_index
                generator = _make_generator(device, generator_seed)
                calls = 0

                def capture(*_: object) -> None:
                    nonlocal calls
                    calls += 1

                generated = sample_from_tokens(
                    model,
                    corrupted,
                    mask_token_id=model_config.mask_token_id,
                    pad_token_id=model_config.pad_token_id,
                    vocab_size=model_config.vocab_size,
                    steps=steps,
                    strategy=sampler_strategy,
                    temperature=diffusion_config.temperature,
                    schedule=schedule,
                    device=device,
                    generator=generator,
                    callback=capture,
                    start_timestep=initial_timestep,
                )
                network_forward_passes += calls
                batch_counts.append(
                    accumulate_token_metrics(
                        generated.cpu(),
                        reference.cpu(),
                        model_config.pad_token_id,
                        selected.cpu(),
                    )
                )
            counts = merge_token_metrics(batch_counts)
            rows.append(
                {
                    "ordering_strategy": strategy,
                    "sampler_strategy": sampler_strategy,
                    "steps": steps,
                    "network_forward_passes": network_forward_passes,
                    "evaluated_sequences": counts.sequences,
                    "evaluated_tokens": counts.total,
                    "correct_tokens": counts.correct,
                    "token_accuracy": counts.accuracy,
                }
            )
    return rows


def build_reference_batches(
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    *,
    text: str,
    corpus_path: Path | None,
    batch_size: int,
    max_sequences: int,
    validation_fraction: float,
) -> tuple[list[Tensor], str]:
    if batch_size < 1 or max_sequences < 1:
        raise ValueError("batch size and max sequences must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be in the open unit interval")
    if corpus_path is None:
        ids = tokenizer.encode(text)[: model_config.max_length]
        ids += [model_config.pad_token_id] * (model_config.max_length - len(ids))
        return [torch.tensor([ids], dtype=torch.long)], "command-line text"
    dataset = _load_token_dataset(
        corpus_path,
        model_config.max_length,
        model_config.pad_token_id,
    )
    validation_start = int(len(dataset) * (1.0 - validation_fraction))
    selected = [
        dataset[index]
        for index in range(
            validation_start,
            min(len(dataset), validation_start + max_sequences),
        )
    ]
    if not selected:
        raise ValueError("corpus contains no held-out validation sequences")
    batches = [
        torch.stack(selected[offset : offset + batch_size])
        for offset in range(0, len(selected), batch_size)
    ]
    return batches, f"contiguous tail holdout from {corpus_path}"


def run_study(
    *,
    checkpoint_path: Path,
    config_path: Path | None,
    step_budgets: Iterable[int],
    ordering_strategies: Iterable[OrderingStrategy],
    text: str,
    corpus_path: Path | None,
    batch_size: int,
    max_sequences: int,
    validation_fraction: float | None,
    seed: int,
    device_request: str,
    tokenizer_path: Path | None = None,
    initial_timestep: float = DEFAULT_INITIAL_TIMESTEP,
) -> dict[str, Any]:
    project_config = ProjectConfig.from_toml(config_path) if config_path else None
    device = select_device(device_request)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    tokenizer = _load_tokenizer(model_config, checkpoint, tokenizer_path)
    if validation_fraction is None:
        train_config = checkpoint.get("train_config", {})
        validation_fraction = float(train_config.get("validation_fraction", 0.1))
    batches, source = build_reference_batches(
        tokenizer,
        model_config,
        text=text,
        corpus_path=corpus_path,
        batch_size=batch_size,
        max_sequences=max_sequences,
        validation_fraction=validation_fraction,
    )
    model = MaskedDiffusionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = run_budget_sweep(
        model,
        batches,
        model_config=model_config,
        diffusion_config=diffusion_config,
        step_budgets=step_budgets,
        ordering_strategies=ordering_strategies,
        seed=seed,
        device=device,
        initial_timestep=initial_timestep,
    )
    ordering_comparison = compare_orderings(rows)
    return {
        "status": "measured",
        "scale": "development-scale local run",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": _device_name(device),
        },
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "config": asdict(project_config) if project_config else None,
        "checkpoint_config": {
            "model": asdict(model_config),
            "diffusion": asdict(diffusion_config),
            "train": checkpoint.get("train_config"),
        },
        "seed": seed,
        "step_budgets": list(parse_step_budgets(step_budgets)),
        "ordering_strategies": list(parse_ordering_strategies(ordering_strategies)),
        "initial_timestep": initial_timestep,
        "metric": {
            "name": "initial_masked_token_accuracy",
            "definition": "final decoded token match rate over positions selected by initial corruption",
        },
        "source": source,
        "batch_size": batch_size,
        "max_sequences": max_sequences,
        "validation_fraction": validation_fraction,
        "results": rows,
        "best_by_steps": best_by_steps(rows),
        "pareto_frontier": pareto_frontier(rows),
        "ordering_comparison": ordering_comparison,
        "hypotheses": {
            "more_reveal_steps_help_and_saturate": summarize_step_budget_hypothesis(rows),
            "confidence_ordering_dominates_fixed_order": summarize_ordering_hypothesis(
                ordering_comparison
            ),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/smoke-checkpoint.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/decoding_study.toml"),
    )
    parser.add_argument(
        "--step-budgets",
        default=",".join(str(value) for value in DEFAULT_STEP_BUDGETS),
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_ORDERING_STRATEGIES),
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--initial-timestep",
        type=float,
        default=DEFAULT_INITIAL_TIMESTEP,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/decoding-extension.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.json"),
    )
    args = parser.parse_args(argv)
    try:
        budgets = parse_step_budgets(args.step_budgets)
        strategies = parse_ordering_strategies(args.strategies)
        result = run_study(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            step_budgets=budgets,
            ordering_strategies=strategies,
            text=args.text,
            corpus_path=args.corpus_path,
            batch_size=args.batch_size,
            max_sequences=args.max_sequences,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device_request=args.device,
            tokenizer_path=args.tokenizer_path,
            initial_timestep=args.initial_timestep,
        )
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.manifest:
        record_artifact(
            args.manifest,
            args.output,
            kind="decoding-extension",
            status="measured",
            provenance={
                "command": "python -m diffusion_lm_nano.decoding_study",
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "seed": args.seed,
                "step_budgets": list(budgets),
                "ordering_strategies": list(strategies),
                "initial_timestep": args.initial_timestep,
                "device": result["environment"]["device"],
            },
        )
    print(payload, end="")


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _load_tokenizer(
    model_config: ModelConfig,
    checkpoint: dict[str, Any],
    tokenizer_path: Path | None,
) -> Tokenizer:
    tokenizer_info = checkpoint.get("tokenizer")
    tokenizer_kind = None if tokenizer_info is None else tokenizer_info.get("kind")
    if tokenizer_kind == "GPT2NanoTokenizerAdapter":
        if tokenizer_path is None and tokenizer_info.get("path"):
            tokenizer_path = Path(str(tokenizer_info["path"]))
        if tokenizer_path is None:
            raise ValueError("checkpoint requires --tokenizer-path for GPT2 token ids")
        tokenizer: Tokenizer = GPT2NanoTokenizerAdapter(tokenizer_path)
    elif tokenizer_kind in {None, "ByteTokenizer"}:
        tokenizer = ByteTokenizer()
    else:
        raise ValueError(f"unknown tokenizer kind in checkpoint: {tokenizer_kind!r}")
    if (
        tokenizer.vocab_size != model_config.vocab_size
        or tokenizer.pad_token_id != model_config.pad_token_id
        or tokenizer.mask_token_id != model_config.mask_token_id
    ):
        raise ValueError("checkpoint and tokenizer vocabulary settings do not match")
    return tokenizer


def _load_token_dataset(
    corpus_path: Path,
    sequence_length: int,
    token_limit: int,
) -> BinaryTokenDataset | JsonlTokenDataset:
    if corpus_path.suffix == ".jsonl":
        return JsonlTokenDataset(corpus_path, sequence_length)
    return BinaryTokenDataset(
        corpus_path,
        sequence_length,
        token_limit=token_limit,
    )


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()
