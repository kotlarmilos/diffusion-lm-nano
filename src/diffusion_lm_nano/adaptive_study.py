"""Run a calibrated, matched study of fixed and adaptive reveal policies."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import time
import tomllib
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .artifacts import record_artifact, sha256
from .config import AdaptiveDecodeConfig, DiffusionConfig, ModelConfig, ProjectConfig
from .corruption import corrupt_tokens
from .model import MaskedDiffusionTransformer
from .sampler import sample_from_tokens_with_metadata
from .schedules import MaskSchedule, make_schedule
from .tokenizer_data import BinaryTokenDataset, JsonlTokenDataset
from .train import select_device

StudyPolicy = Literal["fixed_order", "fixed_fraction", "confidence", "adaptive"]
Precision = Literal["float32", "float16", "bfloat16"]

REQUIRED_POLICIES = frozenset(
    {"fixed_order", "fixed_fraction", "confidence", "adaptive"}
)
SELECTION_RULE = (
    "maximize calibration initially masked token accuracy, then exact sequence "
    "recovery, then minimize mean realized forward passes; remaining ties use "
    "the lexicographically smallest preregistered configuration"
)


@dataclass(frozen=True, slots=True)
class AdaptiveStudyConfig:
    checkpoint_path: Path
    training_config_path: Path
    corpus_path: Path
    output_path: Path
    manifest_path: Path | None
    device: str
    precision: Precision
    seed: int
    initial_timestep: float
    max_passes: int
    validation_fraction: float | None
    calibration_sequences: int
    heldout_sequences: int
    warmup_runs: int
    policies: tuple[StudyPolicy, ...]
    adaptive_min_passes: tuple[int, ...]
    adaptive_entropy_thresholds: tuple[float, ...]
    adaptive_confidence_deltas: tuple[float, ...]
    adaptive_confidence_patiences: tuple[int, ...]

    @classmethod
    def from_toml(cls, path: str | Path) -> "AdaptiveStudyConfig":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        if set(raw) != {"study"}:
            unknown = sorted(set(raw) - {"study"})
            if unknown:
                raise ValueError(f"unknown adaptive study root key: {unknown[0]}")
            raise ValueError("adaptive study config must contain [study]")
        values = raw["study"]
        if not isinstance(values, dict):
            raise ValueError("study must be a TOML table")
        allowed = {
            "checkpoint_path",
            "training_config_path",
            "corpus_path",
            "output_path",
            "manifest_path",
            "device",
            "precision",
            "seed",
            "initial_timestep",
            "max_passes",
            "validation_fraction",
            "calibration_sequences",
            "heldout_sequences",
            "warmup_runs",
            "policies",
            "adaptive_min_passes",
            "adaptive_entropy_thresholds",
            "adaptive_confidence_deltas",
            "adaptive_confidence_patiences",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown study key: {unknown[0]}")
        required = {
            "checkpoint_path",
            "training_config_path",
            "corpus_path",
            "output_path",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"missing study key: {missing[0]}")

        config = cls(
            checkpoint_path=_path_value(values["checkpoint_path"], "checkpoint_path"),
            training_config_path=_path_value(
                values["training_config_path"], "training_config_path"
            ),
            corpus_path=_path_value(values["corpus_path"], "corpus_path"),
            output_path=_path_value(values["output_path"], "output_path"),
            manifest_path=(
                _path_value(values["manifest_path"], "manifest_path")
                if "manifest_path" in values
                else None
            ),
            device=_string_value(values.get("device", "auto"), "device"),
            precision=cast(
                Precision,
                _string_value(values.get("precision", "float32"), "precision"),
            ),
            seed=_integer_value(values.get("seed", 2026), "seed"),
            initial_timestep=_float_value(
                values.get("initial_timestep", 0.75), "initial_timestep"
            ),
            max_passes=_integer_value(values.get("max_passes", 16), "max_passes"),
            validation_fraction=(
                _float_value(values["validation_fraction"], "validation_fraction")
                if "validation_fraction" in values
                else None
            ),
            calibration_sequences=_integer_value(
                values.get("calibration_sequences", 32), "calibration_sequences"
            ),
            heldout_sequences=_integer_value(
                values.get("heldout_sequences", 128), "heldout_sequences"
            ),
            warmup_runs=_integer_value(
                values.get("warmup_runs", 1), "warmup_runs"
            ),
            policies=_parse_policies(
                values.get(
                    "policies",
                    ["fixed_order", "fixed_fraction", "confidence", "adaptive"],
                )
            ),
            adaptive_min_passes=_number_tuple(
                values.get("adaptive_min_passes", [4]), int, "adaptive_min_passes"
            ),
            adaptive_entropy_thresholds=_number_tuple(
                values.get("adaptive_entropy_thresholds", [0.0]),
                float,
                "adaptive_entropy_thresholds",
            ),
            adaptive_confidence_deltas=_number_tuple(
                values.get("adaptive_confidence_deltas", [0.0]),
                float,
                "adaptive_confidence_deltas",
            ),
            adaptive_confidence_patiences=_number_tuple(
                values.get("adaptive_confidence_patiences", [0]),
                int,
                "adaptive_confidence_patiences",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("study.device must be auto, cpu, cuda, or mps")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("study.precision must be float32, float16, or bfloat16")
        if self.max_passes < 1:
            raise ValueError("study.max_passes must be positive")
        if not 0.0 < self.initial_timestep <= 1.0:
            raise ValueError("study.initial_timestep must be in (0, 1]")
        if self.validation_fraction is not None and not (
            0.0 < self.validation_fraction < 1.0
        ):
            raise ValueError("study.validation_fraction must be in (0, 1)")
        if self.calibration_sequences < 1 or self.heldout_sequences < 1:
            raise ValueError("study sequence counts must be positive")
        if self.warmup_runs < 0:
            raise ValueError("study.warmup_runs must be non-negative")
        if set(self.policies) != REQUIRED_POLICIES or len(self.policies) != 4:
            raise ValueError(
                "study.policies must contain fixed_order, fixed_fraction, "
                "confidence, and adaptive exactly once"
            )
        self.adaptive_candidates()

    def adaptive_candidates(self) -> tuple[AdaptiveDecodeConfig, ...]:
        candidates = tuple(
            AdaptiveDecodeConfig(
                min_passes=min_passes,
                entropy_threshold=entropy_threshold,
                confidence_delta=confidence_delta,
                confidence_patience=confidence_patience,
            )
            for (
                min_passes,
                entropy_threshold,
                confidence_delta,
                confidence_patience,
            ) in product(
                self.adaptive_min_passes,
                self.adaptive_entropy_thresholds,
                self.adaptive_confidence_deltas,
                self.adaptive_confidence_patiences,
            )
        )
        if not candidates:
            raise ValueError("adaptive candidate grid cannot be empty")
        for candidate in candidates:
            candidate.validate(self.max_passes)
        return candidates


@dataclass(frozen=True, slots=True)
class PreparedExample:
    dataset_index: int
    reference: Tensor
    corrupted: Tensor
    selected: Tensor
    corruption_seed: int
    decoding_seed: int


def split_holdout_windows(
    dataset_length: int,
    validation_fraction: float,
    calibration_sequences: int,
    heldout_sequences: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if dataset_length < 1:
        raise ValueError("dataset must contain at least one sequence")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if calibration_sequences < 1 or heldout_sequences < 1:
        raise ValueError("window sizes must be positive")
    validation_start = int(dataset_length * (1.0 - validation_fraction))
    end = validation_start + calibration_sequences + heldout_sequences
    if end > dataset_length:
        raise ValueError(
            "contiguous tail holdout is too small for calibration and held-out windows"
        )
    calibration = tuple(
        range(validation_start, validation_start + calibration_sequences)
    )
    heldout = tuple(range(validation_start + calibration_sequences, end))
    return calibration, heldout


@torch.no_grad()
def prepare_examples(
    references: Sequence[tuple[int, Tensor]],
    *,
    model_config: ModelConfig,
    schedule: MaskSchedule,
    initial_timestep: float,
    seed: int,
    device: torch.device,
) -> tuple[PreparedExample, ...]:
    if not references:
        raise ValueError("at least one reference sequence is required")
    prepared = []
    for dataset_index, reference in references:
        if reference.ndim != 1:
            raise ValueError("reference sequences must be rank one")
        clean = reference[None, :].to(device=device, dtype=torch.long)
        attention = clean.ne(model_config.pad_token_id)
        t = torch.full((1,), initial_timestep, device=device)
        corruption_seed = seed + dataset_index
        corrupted, _targets, selected = corrupt_tokens(
            clean,
            t,
            schedule,
            model_config.mask_token_id,
            attention,
            _make_generator(device, corruption_seed),
        )
        if not selected.any():
            raise ValueError(
                f"initial corruption selected no tokens for dataset index {dataset_index}"
            )
        prepared.append(
            PreparedExample(
                dataset_index=dataset_index,
                reference=clean,
                corrupted=corrupted,
                selected=selected,
                corruption_seed=corruption_seed,
                decoding_seed=seed + 1_000_000 + dataset_index,
            )
        )
    return tuple(prepared)


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    examples: Sequence[PreparedExample],
    *,
    policy: StudyPolicy,
    adaptive_config: AdaptiveDecodeConfig | None,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    schedule: MaskSchedule,
    max_passes: int,
    initial_timestep: float,
    device: torch.device,
    precision: Precision,
    include_trajectories: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not examples:
        raise ValueError("at least one prepared example is required")
    records = [
        _evaluate_example(
            model,
            example,
            policy=policy,
            adaptive_config=adaptive_config,
            model_config=model_config,
            diffusion_config=diffusion_config,
            schedule=schedule,
            max_passes=max_passes,
            initial_timestep=initial_timestep,
            device=device,
            precision=precision,
            include_trajectory=include_trajectories,
        )
        for example in examples
    ]
    return summarize_policy_records(policy, records, max_passes), records


def summarize_policy_records(
    policy: StudyPolicy,
    records: Sequence[dict[str, Any]],
    declared_max_passes: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("policy records cannot be empty")
    correct = sum(int(record["correct_masked_tokens"]) for record in records)
    total = sum(int(record["initially_masked_tokens"]) for record in records)
    exact = sum(bool(record["exact_sequence_recovery"]) for record in records)
    passes = [int(record["realized_forward_passes"]) for record in records]
    elapsed = [float(record["wall_time_seconds"]) for record in records]
    stopping_reasons = Counter(str(record["stopping_reason"]) for record in records)
    return {
        "policy": policy,
        "declared_max_passes": declared_max_passes,
        "evaluated_sequences": len(records),
        "initially_masked_tokens": total,
        "correct_masked_tokens": correct,
        "initially_masked_token_accuracy": correct / total if total else None,
        "exact_sequence_recoveries": exact,
        "exact_sequence_recovery_rate": exact / len(records),
        "realized_forward_passes_total": sum(passes),
        "mean_realized_forward_passes": sum(passes) / len(passes),
        "min_realized_forward_passes": min(passes),
        "max_realized_forward_passes": max(passes),
        "wall_time_seconds_total": sum(elapsed),
        "mean_wall_time_seconds": sum(elapsed) / len(elapsed),
        "stopping_reasons": dict(sorted(stopping_reasons.items())),
    }


def select_adaptive_candidate(
    candidate_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_results:
        raise ValueError("candidate results cannot be empty")

    def sort_key(row: dict[str, Any]) -> tuple[float, ...]:
        config = row["adaptive_config"]
        accuracy = row["initially_masked_token_accuracy"]
        exact = row["exact_sequence_recovery_rate"]
        if accuracy is None:
            raise ValueError("candidate accuracy is unavailable")
        return (
            -float(accuracy),
            -float(exact),
            float(row["mean_realized_forward_passes"]),
            float(config["min_passes"]),
            float(config["entropy_threshold"]),
            float(config["confidence_delta"]),
            float(config["confidence_patience"]),
        )

    return min(candidate_results, key=sort_key)


@torch.no_grad()
def calibrate_adaptive_policy(
    model: nn.Module,
    examples: Sequence[PreparedExample],
    *,
    candidates: Sequence[AdaptiveDecodeConfig],
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    schedule: MaskSchedule,
    max_passes: int,
    initial_timestep: float,
    device: torch.device,
    precision: Precision,
) -> tuple[AdaptiveDecodeConfig, dict[str, Any]]:
    candidate_results = []
    by_config: dict[tuple[int, float, float, int], AdaptiveDecodeConfig] = {}
    for candidate in candidates:
        summary, _records = evaluate_policy(
            model,
            examples,
            policy="adaptive",
            adaptive_config=candidate,
            model_config=model_config,
            diffusion_config=diffusion_config,
            schedule=schedule,
            max_passes=max_passes,
            initial_timestep=initial_timestep,
            device=device,
            precision=precision,
            include_trajectories=False,
        )
        config_values = asdict(candidate)
        summary["adaptive_config"] = config_values
        candidate_results.append(summary)
        by_config[_adaptive_key(candidate)] = candidate
    selected_metrics = select_adaptive_candidate(candidate_results)
    selected_config = AdaptiveDecodeConfig(**selected_metrics["adaptive_config"])
    selected_config = by_config[_adaptive_key(selected_config)]
    return selected_config, {
        "selection_rule": SELECTION_RULE,
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
        "selected_config": asdict(selected_config),
        "selected_metrics": selected_metrics,
    }


@torch.no_grad()
def run_matched_evaluation(
    model: nn.Module,
    examples: Sequence[PreparedExample],
    *,
    policies: Sequence[StudyPolicy],
    adaptive_config: AdaptiveDecodeConfig,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    schedule: MaskSchedule,
    max_passes: int,
    initial_timestep: float,
    device: torch.device,
    precision: Precision,
    warmup_runs: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one held-out example is required")
    if set(policies) != REQUIRED_POLICIES or len(policies) != 4:
        raise ValueError("matched evaluation requires all four policies")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    by_example = {
        example.dataset_index: {
            "dataset_index": example.dataset_index,
            "corruption_seed": example.corruption_seed,
            "decoding_seed": example.decoding_seed,
            "initially_masked_tokens": int(example.selected.sum().item()),
            "policies": {},
        }
        for example in examples
    }
    policy_results = []
    for policy in policies:
        policy_adaptive_config = adaptive_config if policy == "adaptive" else None
        for _ in range(warmup_runs):
            _evaluate_example(
                model,
                examples[0],
                policy=policy,
                adaptive_config=policy_adaptive_config,
                model_config=model_config,
                diffusion_config=diffusion_config,
                schedule=schedule,
                max_passes=max_passes,
                initial_timestep=initial_timestep,
                device=device,
                precision=precision,
                include_trajectory=False,
            )
        summary, records = evaluate_policy(
            model,
            examples,
            policy=policy,
            adaptive_config=policy_adaptive_config,
            model_config=model_config,
            diffusion_config=diffusion_config,
            schedule=schedule,
            max_passes=max_passes,
            initial_timestep=initial_timestep,
            device=device,
            precision=precision,
            include_trajectories=True,
        )
        if policy == "adaptive":
            summary["adaptive_config"] = asdict(adaptive_config)
        policy_results.append(summary)
        for example, record in zip(examples, records, strict=True):
            by_example[example.dataset_index]["policies"][policy] = record
    return {
        "policy_results": policy_results,
        "paired_examples": [
            by_example[example.dataset_index] for example in examples
        ],
    }


def _verify_manifest_expectations(
    config: AdaptiveStudyConfig,
    study_config_path: Path,
) -> dict[str, Any] | None:
    if config.manifest_path is None:
        return None
    manifest_path = config.manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    root = manifest_path.parent.parent.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")

    def relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"publication input is outside the repository: {path}") from error

    output_relative = relative(config.output_path)
    matching = [
        entry
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("path") == output_relative
    ]
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError("manifest contains duplicate adaptive study entries")
    provenance = matching[0].get("provenance")
    expectations = (
        provenance.get("immutable_input_expectations")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(expectations, dict):
        raise ValueError("adaptive study manifest entry lacks immutable expectations")

    actual_paths = {
        "checkpoint": config.checkpoint_path,
        "corpus": config.corpus_path,
        "study_config": study_config_path,
        "training_config": config.training_config_path,
    }
    verified: dict[str, Any] = {}
    for name, actual_path in actual_paths.items():
        expectation = expectations.get(name)
        if not isinstance(expectation, dict):
            raise ValueError(f"manifest lacks {name} input expectations")
        expected_path = expectation.get("path")
        actual_relative = relative(actual_path)
        if expected_path != actual_relative:
            raise ValueError(f"{name} path does not match the manifest expectation")
        expected_hash = expectation.get("sha256")
        expected_status = expectation.get("status")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            source_entries = [
                entry
                for entry in artifacts
                if isinstance(entry, dict) and entry.get("path") == expected_path
            ]
            if len(source_entries) != 1:
                raise ValueError(f"manifest cannot resolve {name} expected hash")
            source_entry = source_entries[0]
            expected_hash = source_entry.get("sha256")
            if (
                expected_status is not None
                and source_entry.get("status") != expected_status
            ):
                raise ValueError(f"{name} does not have expected manifest status")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"manifest has no immutable {name} hash")
        observed_hash = sha256(actual_path)
        if observed_hash != expected_hash:
            raise ValueError(f"{name} hash does not match the manifest expectation")
        verified[name] = {
            **expectation,
            "path": expected_path,
            "sha256": expected_hash,
            "observed_sha256": observed_hash,
        }
    return verified


def run_study(
    config: AdaptiveStudyConfig,
    *,
    study_config_path: Path,
) -> dict[str, Any]:
    config.validate()
    for name, path in (
        ("checkpoint", config.checkpoint_path),
        ("training config", config.training_config_path),
        ("corpus", config.corpus_path),
        ("study config", study_config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    immutable_expectations = _verify_manifest_expectations(
        config,
        study_config_path,
    )
    device = select_device(config.device)
    if config.precision != "float32" and device.type != "cuda":
        raise ValueError("reduced precision study runs require CUDA")
    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    source = checkpoint.get("source")
    if (
        not isinstance(source, dict)
        or source.get("git_dirty") is not False
        or source.get("git_state_captured") != "training_start"
        or not isinstance(source.get("git_revision"), str)
        or not source["git_revision"]
    ):
        raise ValueError("checkpoint does not record clean training source provenance")
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    model_config.validate()
    diffusion_config.validate()
    training_config = ProjectConfig.from_toml(config.training_config_path)
    if training_config.model != model_config:
        raise ValueError("training config model does not match checkpoint")
    if training_config.diffusion != diffusion_config:
        raise ValueError("training config diffusion settings do not match checkpoint")
    if asdict(training_config.train) != checkpoint.get("train_config"):
        raise ValueError("training config train settings do not match checkpoint")
    data_split = checkpoint.get("data_split")
    if (
        not isinstance(data_split, dict)
        or data_split.get("method") != "contiguous tail holdout"
        or not isinstance(data_split.get("validation_fraction"), (int, float))
    ):
        raise ValueError("checkpoint does not record its contiguous validation split")
    validation_fraction = float(data_split["validation_fraction"])
    if (
        config.validation_fraction is not None
        and not math.isclose(config.validation_fraction, validation_fraction)
    ):
        raise ValueError("study validation fraction does not match checkpoint split")
    training_inputs = checkpoint.get("training_inputs")
    if not isinstance(training_inputs, dict):
        raise ValueError(
            "checkpoint does not record immutable training inputs"
        )
    corpus_hash = sha256(config.corpus_path)
    training_config_hash = sha256(config.training_config_path)
    if training_inputs.get("corpus_sha256") != corpus_hash:
        raise ValueError("study corpus does not match checkpoint training corpus")
    if training_inputs.get("training_config_sha256") != training_config_hash:
        raise ValueError(
            "study training config does not match checkpoint training input"
        )
    dataset = _load_token_dataset(config.corpus_path, model_config.max_length)
    calibration_indices, heldout_indices = split_holdout_windows(
        len(dataset),
        validation_fraction,
        config.calibration_sequences,
        config.heldout_sequences,
    )
    schedule = make_schedule(
        diffusion_config.schedule,
        diffusion_config.schedule_power,
    )
    calibration_examples = prepare_examples(
        [(index, dataset[index]) for index in calibration_indices],
        model_config=model_config,
        schedule=schedule,
        initial_timestep=config.initial_timestep,
        seed=config.seed,
        device=device,
    )
    heldout_examples = prepare_examples(
        [(index, dataset[index]) for index in heldout_indices],
        model_config=model_config,
        schedule=schedule,
        initial_timestep=config.initial_timestep,
        seed=config.seed,
        device=device,
    )
    model = MaskedDiffusionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    selected_config, calibration = calibrate_adaptive_policy(
        model,
        calibration_examples,
        candidates=config.adaptive_candidates(),
        model_config=model_config,
        diffusion_config=diffusion_config,
        schedule=schedule,
        max_passes=config.max_passes,
        initial_timestep=config.initial_timestep,
        device=device,
        precision=config.precision,
    )
    heldout = run_matched_evaluation(
        model,
        heldout_examples,
        policies=config.policies,
        adaptive_config=selected_config,
        model_config=model_config,
        diffusion_config=diffusion_config,
        schedule=schedule,
        max_passes=config.max_passes,
        initial_timestep=config.initial_timestep,
        device=device,
        precision=config.precision,
        warmup_runs=config.warmup_runs,
    )
    input_hashes = {
        "checkpoint": sha256(config.checkpoint_path),
        "training_config": training_config_hash,
        "study_config": sha256(study_config_path),
        "corpus": corpus_hash,
    }
    return {
        "status": "measured",
        "study": "confidence-calibrated adaptive reveal scheduling",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": _device_name(device),
            "precision": config.precision,
        },
        "source": source,
        "inputs": {
            "checkpoint": str(config.checkpoint_path),
            "training_config": str(config.training_config_path),
            "study_config": str(study_config_path),
            "corpus": str(config.corpus_path),
            "sha256": input_hashes,
            "immutable_input_expectations": immutable_expectations,
        },
        "protocol": {
            "seed": config.seed,
            "initial_timestep": config.initial_timestep,
            "declared_max_passes": config.max_passes,
            "policies": list(config.policies),
            "temperature": diffusion_config.temperature,
            "schedule": diffusion_config.schedule,
            "precision": config.precision,
            "hardware": _device_name(device),
            "corruption": (
                "prepared once per example and reused by every policy; decoding "
                "seeds are also identical across policies"
            ),
            "metrics": {
                "reconstruction": "accuracy only over initially masked tokens",
                "exact_recovery": "all non-padding reference tokens match",
                "passes": "actual denoiser forward calls",
                "wall_time": (
                    "accelerator-synchronized per-example sampler wall time; "
                    "includes Python overhead and does not establish a speed claim"
                ),
            },
            "fixed_policy_budget": (
                "fixed_order, fixed_fraction, and confidence use the declared "
                "maximum-pass budget"
            ),
            "adaptive_budget": "adaptive reports realized forward passes",
            "warmup_runs_per_policy": config.warmup_runs,
        },
        "windows": {
            "method": "disjoint contiguous windows within the training tail holdout",
            "validation_fraction": validation_fraction,
            "calibration": {
                "start_dataset_index": calibration_indices[0],
                "end_dataset_index_exclusive": calibration_indices[-1] + 1,
                "sequences": len(calibration_indices),
            },
            "heldout": {
                "start_dataset_index": heldout_indices[0],
                "end_dataset_index_exclusive": heldout_indices[-1] + 1,
                "sequences": len(heldout_indices),
            },
        },
        "calibration": calibration,
        "heldout": heldout,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adaptive_study_smoke.toml"),
    )
    args = parser.parse_args(argv)
    try:
        config = AdaptiveStudyConfig.from_toml(args.config)
        result = run_study(config, study_config_path=args.config)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(payload, encoding="utf-8")
    if config.manifest_path is not None:
        record_artifact(
            config.manifest_path,
            config.output_path,
            kind="adaptive-decoding-study",
            status="measured",
            provenance={
                "command": (
                    "python -m diffusion_lm_nano.adaptive_study "
                    f"--config {args.config}"
                ),
                "checkpoint": str(config.checkpoint_path),
                "corpus": str(config.corpus_path),
                "input_sha256": result["inputs"]["sha256"],
                "immutable_input_expectations": result["inputs"][
                    "immutable_input_expectations"
                ],
                "selected_adaptive_config": result["calibration"]["selected_config"],
                "seed": config.seed,
                "device": result["environment"]["device"],
                "precision": config.precision,
                "calibration_window": result["windows"]["calibration"],
                "heldout_window": result["windows"]["heldout"],
            },
        )
    print(payload, end="")


def _evaluate_example(
    model: nn.Module,
    example: PreparedExample,
    *,
    policy: StudyPolicy,
    adaptive_config: AdaptiveDecodeConfig | None,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    schedule: MaskSchedule,
    max_passes: int,
    initial_timestep: float,
    device: torch.device,
    precision: Precision,
    include_trajectory: bool,
) -> dict[str, Any]:
    generator = _make_generator(device, example.decoding_seed)
    _synchronize(device)
    start = time.perf_counter()
    with _autocast_context(device, precision):
        result = sample_from_tokens_with_metadata(
            model,
            example.corrupted,
            mask_token_id=model_config.mask_token_id,
            pad_token_id=model_config.pad_token_id,
            vocab_size=model_config.vocab_size,
            steps=max_passes,
            strategy=policy,
            temperature=diffusion_config.temperature,
            schedule=schedule,
            device=device,
            generator=generator,
            start_timestep=initial_timestep,
            adaptive_config=adaptive_config,
            collect_diagnostics=policy == "adaptive",
        )
    _synchronize(device)
    elapsed = time.perf_counter() - start
    active = example.reference.ne(model_config.pad_token_id)
    correct_masked = int(
        (
            result.tokens[example.selected]
            == example.reference[example.selected]
        )
        .sum()
        .item()
    )
    initially_masked = int(example.selected.sum().item())
    exact_recovery = bool(
        (result.tokens[active] == example.reference[active]).all().item()
    )
    record = {
        "correct_masked_tokens": correct_masked,
        "initially_masked_tokens": initially_masked,
        "initially_masked_token_accuracy": correct_masked / initially_masked,
        "exact_sequence_recovery": exact_recovery,
        "declared_max_passes": max_passes,
        "realized_forward_passes": result.forward_passes,
        "wall_time_seconds": elapsed,
        "stopping_reason": result.stopping_reason,
        "complete_output": not bool(
            result.tokens.eq(model_config.mask_token_id).any().item()
        ),
        "output_token_ids_sha256": _token_sha256(result.tokens),
    }
    if include_trajectory:
        record["trajectory"] = [asdict(step) for step in result.trajectory]
    return record


def _parse_policies(values: Any) -> tuple[StudyPolicy, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("study.policies must be an array of strings")
    parsed = tuple(cast(StudyPolicy, value) for value in values)
    unknown = sorted(set(parsed) - REQUIRED_POLICIES)
    if unknown:
        raise ValueError(f"unknown study policy: {unknown[0]}")
    return parsed


def _number_tuple(
    values: Any,
    conversion: type[int] | type[float],
    name: str,
) -> tuple[int, ...] | tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"study.{name} must be a non-empty array")
    if conversion is int:
        converted = tuple(_integer_value(value, name) for value in values)
    else:
        converted = tuple(_float_value(value, name) for value in values)
    if len(set(converted)) != len(converted):
        raise ValueError(f"study.{name} values must be unique")
    return converted


def _path_value(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"study.{name} must be a non-empty path string")
    return Path(value)


def _string_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"study.{name} must be a string")
    return value


def _integer_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"study.{name} must contain integers")
    return value


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"study.{name} must contain numeric values")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"study.{name} values must be finite")
    return converted


def _adaptive_key(config: AdaptiveDecodeConfig) -> tuple[int, float, float, int]:
    return (
        config.min_passes,
        config.entropy_threshold,
        config.confidence_delta,
        config.confidence_patience,
    )


def _load_token_dataset(
    corpus_path: Path,
    sequence_length: int,
) -> BinaryTokenDataset | JsonlTokenDataset:
    if corpus_path.suffix == ".jsonl":
        return JsonlTokenDataset(corpus_path, sequence_length)
    if corpus_path.suffix == ".bin":
        return BinaryTokenDataset(corpus_path, sequence_length)
    raise ValueError("study corpus must be a .bin or .jsonl token dataset")


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _autocast_context(device: torch.device, precision: Precision) -> Any:
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("reduced precision study runs require CUDA")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine()


def _token_sha256(tokens: Tensor) -> str:
    payload = json.dumps(
        tokens.detach().cpu().tolist(),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
