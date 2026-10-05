"""Measure majority-token collapse against the publication checkpoint holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from diffusion_lm_nano.artifacts import record_artifact, sha256
from diffusion_lm_nano.config import ProjectConfig
from diffusion_lm_nano.tokenizer_data import BinaryTokenDataset


def analyze_majority_control(
    config_path: Path,
    checkpoint_path: Path,
    corpus_path: Path,
    vocab_path: Path,
    evaluation_path: Path,
    sample_path: Path,
) -> dict[str, object]:
    config = ProjectConfig.from_toml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    train_config = checkpoint["train_config"]
    source = checkpoint.get("source")
    training_inputs = checkpoint.get("training_inputs")
    if (
        model_config["max_length"] != config.model.max_length
        or model_config["pad_token_id"] != config.model.pad_token_id
        or train_config["validation_fraction"] != config.train.validation_fraction
    ):
        raise ValueError("checkpoint holdout settings do not match the configuration")
    if (
        not isinstance(source, dict)
        or source.get("git_dirty") is not False
        or source.get("git_state_captured") != "training_start"
    ):
        raise ValueError("checkpoint does not record clean training source provenance")
    if not isinstance(training_inputs, dict):
        raise ValueError("checkpoint does not record immutable training inputs")
    if training_inputs.get("training_config_sha256") != sha256(config_path):
        raise ValueError("configuration hash does not match the checkpoint")
    if training_inputs.get("corpus_sha256") != sha256(corpus_path):
        raise ValueError("corpus hash does not match the checkpoint")
    dataset = BinaryTokenDataset(
        corpus_path,
        model_config["max_length"],
        token_limit=model_config["pad_token_id"],
    )
    validation_start = int(
        len(dataset) * (1.0 - train_config["validation_fraction"])
    )
    max_sequences = 256
    end = min(len(dataset), validation_start + max_sequences)
    tokens = [
        token
        for index in range(validation_start, end)
        for token in dataset[index].tolist()
    ]
    counts = Counter(tokens)
    majority_token_id, majority_count = counts.most_common(1)[0]
    majority_fraction = majority_count / len(tokens)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    if majority_token_id not in inverse_vocab:
        raise ValueError("holdout majority token is missing from the vocabulary")
    majority_token_piece = inverse_vocab[majority_token_id]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_ids = sample["token_ids"]
    evaluated_sequences = end - validation_start
    if evaluation.get("evaluated_sequences") != evaluated_sequences:
        raise ValueError("evaluation count does not match the checkpoint holdout")
    if len(sample_ids) != model_config["max_length"]:
        raise ValueError("sample length does not match the checkpoint geometry")
    return {
        "status": "measured",
        "method": "constant-token control over the complete evaluated holdout sequences",
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256(config_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
            },
            "corpus": {
                "path": str(corpus_path),
                "sha256": sha256(corpus_path),
            },
            "vocabulary": {
                "path": str(vocab_path),
                "sha256": sha256(vocab_path),
            },
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256(evaluation_path),
            },
            "sample": {
                "path": str(sample_path),
                "sha256": sha256(sample_path),
            },
        },
        "holdout": {
            "start_dataset_index": validation_start,
            "end_dataset_index_exclusive": end,
            "sequences": evaluated_sequences,
            "tokens": len(tokens),
            "majority_token_id": majority_token_id,
            "majority_token_piece": majority_token_piece,
            "majority_token_count": majority_count,
            "majority_token_fraction": majority_fraction,
        },
        "evaluation": {
            "masked_token_accuracy": evaluation["masked_accuracy"],
            "accuracy_minus_holdout_majority_fraction": (
                evaluation["masked_accuracy"] - majority_fraction
            ),
            "comparison_limit": (
                "The constant-token control uses every token in the evaluated "
                "sequences; masked accuracy uses the randomly corrupted positions."
            ),
        },
        "unconditional_sample": {
            "tokens": len(sample_ids),
            "unique_token_ids": len(set(sample_ids)),
            "all_majority_token": set(sample_ids) == {majority_token_id},
            "stopping_reason": sample["stopping_reason"],
            "forward_passes": sample["forward_passes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    recorded_output = Path("artifacts/publication-majority-control.json")
    parser.add_argument("--config", type=Path, default=Path("configs/publication.toml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/publication-checkpoint.pt"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/train.bin"))
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/bpe-tokenizer/vocab.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("artifacts/publication-evaluation.json"),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("artifacts/publication-sample.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=recorded_output,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.json"),
    )
    args = parser.parse_args()
    result = analyze_majority_control(
        args.config,
        args.checkpoint,
        args.corpus,
        args.vocab,
        args.evaluation,
        args.sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output == recorded_output:
        record_artifact(
            args.manifest,
            args.output,
            kind="majority-token-control",
            status="measured",
            provenance={
                "command": "python scripts/analyze_publication_baseline.py",
                "checkpoint": str(args.checkpoint),
                "corpus": str(args.corpus),
                "evaluation": str(args.evaluation),
                "sample": str(args.sample),
            },
        )
    print(args.output)


if __name__ == "__main__":
    main()
