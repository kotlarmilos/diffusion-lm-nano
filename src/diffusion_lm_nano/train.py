"""Training CLI for masked discrete diffusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import torch
from torch import Tensor

from .artifacts import record_artifact
from .config import ProjectConfig
from .corruption import corrupt_tokens
from .model import MaskedDiffusionTransformer
from .objective import masked_cross_entropy
from .sampler import sample
from .schedules import make_schedule
from .tokenizer_data import (
    BinaryTokenDataset,
    ByteTokenizer,
    GPT2NanoTokenizerAdapter,
    JsonlTokenDataset,
)

DEFAULT_TEXT = (
    "Language diffusion learns to restore masked tokens using both left and right "
    "context. A small model makes each assumption visible. "
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_from_bytes(
    corpus: list[int],
    batch_size: int,
    length: int,
    rng: random.Random,
) -> Tensor:
    if len(corpus) < length:
        repeats = length // max(len(corpus), 1) + 1
        corpus = (corpus or [32]) * repeats
    rows = []
    for _ in range(batch_size):
        start = rng.randrange(len(corpus) - length + 1)
        rows.append(corpus[start : start + length])
    return torch.tensor(rows, dtype=torch.long)


def _load_corpus(
    config: ProjectConfig,
    tokenizer: ByteTokenizer | GPT2NanoTokenizerAdapter,
) -> list[int] | BinaryTokenDataset | JsonlTokenDataset:
    path = config.train.corpus_path
    if path is None:
        return tokenizer.encode(DEFAULT_TEXT * 8)
    corpus_path = Path(path)
    if corpus_path.suffix == ".bin":
        return BinaryTokenDataset(corpus_path, config.model.max_length)
    if corpus_path.suffix == ".jsonl":
        return JsonlTokenDataset(corpus_path, config.model.max_length)
    return tokenizer.encode(corpus_path.read_text(encoding="utf-8"))


def train(
    config: ProjectConfig,
    config_path: Path,
    checkpoint_path: Path,
    sample_path: Path,
    manifest_path: Path,
    tokenizer_path: Path | None = None,
) -> dict[str, float]:
    torch.manual_seed(config.train.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.train.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)
    rng = random.Random(config.train.seed)
    device = select_device(config.train.device)
    model = MaskedDiffusionTransformer(config.model).to(device)
    tokenizer = (
        GPT2NanoTokenizerAdapter(tokenizer_path)
        if tokenizer_path is not None
        else ByteTokenizer()
    )
    if (
        tokenizer.vocab_size != config.model.vocab_size
        or tokenizer.pad_token_id != config.model.pad_token_id
        or tokenizer.mask_token_id != config.model.mask_token_id
    ):
        raise ValueError(
            "tokenizer vocabulary and special token IDs must match model config"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    schedule = make_schedule(
        config.diffusion.schedule, config.diffusion.schedule_power
    )
    corpus = _load_corpus(config, tokenizer)
    training_size: int | None = None
    training_tokens: list[int] | None = None
    if isinstance(corpus, (BinaryTokenDataset, JsonlTokenDataset)):
        training_size = int(len(corpus) * (1.0 - config.train.validation_fraction))
        if training_size < 1 or training_size >= len(corpus):
            raise ValueError(
                "token dataset must contain at least one train and validation sequence"
            )
    else:
        split = int(len(corpus) * (1.0 - config.train.validation_fraction))
        training_tokens = (
            corpus[:split]
            if split >= config.model.max_length
            else corpus
        )
    final_loss = float("nan")
    model.train()
    for step in range(1, config.train.steps + 1):
        if isinstance(corpus, (BinaryTokenDataset, JsonlTokenDataset)):
            assert training_size is not None
            indices = [
                rng.randrange(training_size)
                for _ in range(config.train.batch_size)
            ]
            clean = torch.stack([corpus[index] for index in indices])
        else:
            assert training_tokens is not None
            clean = _batch_from_bytes(
                training_tokens,
                config.train.batch_size,
                config.model.max_length,
                rng,
            )
        clean = clean.to(device)
        t = torch.rand(
            config.train.batch_size, generator=generator, device="cpu"
        ).to(device)
        t = t * (1.0 - config.diffusion.train_min_t) + config.diffusion.train_min_t
        corrupted, targets, _ = corrupt_tokens(
            clean,
            t,
            schedule,
            config.model.mask_token_id,
            generator=None if device.type != "cpu" else generator,
        )
        loss = masked_cross_entropy(model(corrupted, t), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % config.train.log_every == 0 or step == config.train.steps:
            print(json.dumps({"step": step, "loss": final_loss}))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_config": asdict(config.model),
        "diffusion_config": asdict(config.diffusion),
        "train_config": asdict(config.train),
        "data_split": {
            "validation_fraction": config.train.validation_fraction,
            "method": "contiguous tail holdout",
        },
        "tokenizer": {
            "kind": type(tokenizer).__name__,
            "path": str(tokenizer_path) if tokenizer_path is not None else None,
        },
        "model_state": model.state_dict(),
        "train_steps": config.train.steps,
        "final_train_loss": final_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    model.eval()
    if device.type == "mps":
        torch.manual_seed(config.train.seed)
        sample_generator = None
    else:
        sample_generator = torch.Generator(device=device).manual_seed(
            config.train.seed
        )
    trajectory: list[dict[str, object]] = []

    def capture(step: int, tokens: Tensor, confidence: Tensor) -> None:
        trajectory.append(
            {
                "step": step,
                "token_ids": tokens[0].cpu().tolist(),
                "mean_confidence": float(confidence[0].mean().cpu()),
            }
        )

    generated = sample(
        model,
        batch_size=1,
        length=config.model.max_length,
        mask_token_id=config.model.mask_token_id,
        pad_token_id=config.model.pad_token_id,
        vocab_size=config.model.vocab_size,
        steps=config.diffusion.sampling_steps,
        strategy=config.diffusion.strategy,
        temperature=config.diffusion.temperature,
        schedule=schedule,
        device=device,
        generator=sample_generator,
        callback=capture,
    )
    decoded = (
        tokenizer.decode(generated[0].cpu().tolist())
        if config.model.vocab_size == tokenizer.vocab_size
        else None
    )
    sample_payload = {
        "status": "measured",
        "seed": config.train.seed,
        "strategy": config.diffusion.strategy,
        "tokenizer": type(tokenizer).__name__,
        "token_ids": generated[0].cpu().tolist(),
        "text": decoded,
        "trajectory": trajectory,
    }
    sample_path.write_text(
        json.dumps(sample_payload, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "command": (
            "python -m diffusion_lm_nano.train "
            f"--config {config_path} --checkpoint {checkpoint_path} "
            f"--sample {sample_path} --manifest {manifest_path}"
        ),
        "config": asdict(config),
        "seed": config.train.seed,
    }
    record_artifact(
        manifest_path,
        checkpoint_path,
        kind="model-checkpoint",
        status="measured",
        provenance=provenance,
    )
    record_artifact(
        manifest_path,
        sample_path,
        kind="iterative-sample",
        status="measured",
        provenance=provenance,
    )
    return {"final_train_loss": final_loss}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/checkpoint.pt")
    )
    parser.add_argument(
        "--sample", type=Path, default=Path("artifacts/sample.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.json")
    )
    args = parser.parse_args(argv)
    config = ProjectConfig.from_toml(args.config)
    train(
        config,
        args.config,
        args.checkpoint,
        args.sample,
        args.manifest,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
