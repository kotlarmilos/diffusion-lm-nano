"""Evaluate denoising loss and masked-token accuracy on deterministic text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import record_artifact
from .config import DiffusionConfig, ModelConfig
from .corruption import corrupt_tokens
from .model import MaskedDiffusionTransformer
from .objective import masked_accuracy, masked_cross_entropy
from .runtime import select_device
from .schedules import make_schedule
from .tokenizer_data import BinaryTokenDataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if not 0.0 <= args.timestep <= 1.0:
        parser.error("--timestep must be in [0, 1]")
    if args.batch_size < 1 or args.max_sequences < 1:
        parser.error("--batch-size and --max-sequences must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    model = MaskedDiffusionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = BinaryTokenDataset(
        args.corpus_path,
        model_config.max_length,
        token_limit=model_config.pad_token_id,
    )
    train_config = checkpoint.get("train_config", {})
    validation_fraction = (
        float(args.validation_fraction)
        if args.validation_fraction is not None
        else float(train_config.get("validation_fraction", 0.1))
    )
    if not 0.0 < validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")
    validation_start = int(len(dataset) * (1.0 - validation_fraction))
    selected = [
        dataset[index]
        for index in range(
            validation_start,
            min(len(dataset), validation_start + args.max_sequences),
        )
    ]
    if not selected:
        raise ValueError("corpus contains no held-out validation sequences")
    batches = [
        torch.stack(selected[offset : offset + args.batch_size]).to(device)
        for offset in range(0, len(selected), args.batch_size)
    ]
    evaluation_source = f"contiguous tail holdout from {args.corpus_path}"
    generator = (
        None
        if device.type == "mps"
        else torch.Generator(device=device).manual_seed(args.seed)
    )
    if device.type == "mps":
        torch.manual_seed(args.seed)
    schedule = make_schedule(
        diffusion_config.schedule,
        diffusion_config.schedule_power,
    )
    loss_sum = 0.0
    correct = 0
    total = 0
    evaluated_sequences = 0
    with torch.no_grad():
        for clean in batches:
            attention = clean.ne(model_config.pad_token_id)
            t = torch.full(
                (clean.shape[0],),
                args.timestep,
                device=device,
            )
            corrupted, targets, _selected = corrupt_tokens(
                clean,
                t,
                schedule,
                model_config.mask_token_id,
                attention,
                generator,
            )
            logits = model(corrupted, t, attention)
            batch_correct, batch_total = masked_accuracy(logits, targets)
            if batch_total:
                loss_sum += (
                    float(masked_cross_entropy(logits, targets))
                    * batch_total
                )
            correct += batch_correct
            total += batch_total
            evaluated_sequences += clean.shape[0]
    result = {
        "status": "measured",
        "device": str(device),
        "seed": args.seed,
        "timestep": args.timestep,
        "source": evaluation_source,
        "evaluated_sequences": evaluated_sequences,
        "masked_tokens": total,
        "masked_accuracy": correct / total if total else None,
        "masked_cross_entropy": loss_sum / total if total else None,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        if args.manifest:
            record_artifact(
                args.manifest,
                args.output,
                kind="evaluation",
                status="measured",
                provenance={
                    "command": "python -m diffusion_lm_nano.evaluate",
                    "checkpoint": str(args.checkpoint),
                    "seed": args.seed,
                    "device": str(device),
                },
            )
    print(payload, end="")


if __name__ == "__main__":
    main()
