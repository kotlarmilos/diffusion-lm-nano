"""Benchmark diffusion decoding under an explicit, result-neutral protocol."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .artifacts import record_artifact
from .config import DiffusionConfig, ModelConfig
from .model import MaskedDiffusionTransformer
from .sampler import sample_with_metadata
from .schedules import make_schedule
from .train import select_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_autoregressive_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    """Load the exact gpt2-nano implementation from its staged wheel."""
    try:
        from src.model import GPT, GPTConfig
    except ImportError as error:
        raise RuntimeError(
            "Install the staged gpt2-nano wheel before requesting the "
            "autoregressive benchmark."
        ) from error
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--length", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--autoregressive-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch size and repeats must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion = DiffusionConfig(**checkpoint["diffusion_config"])
    length = args.length if args.length is not None else model_config.max_length
    if length < 1:
        parser.error("--length must be positive")
    if length > model_config.max_length:
        parser.error("--length exceeds checkpoint maximum")
    model = MaskedDiffusionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    diffusion_parameters = sum(parameter.numel() for parameter in model.parameters())
    measured_calls: list[int] = []
    measured_stopping_reasons: list[str] = []

    def run(seed: int) -> tuple[int, str]:
        if device.type == "mps":
            torch.manual_seed(seed)
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        sampling_result = sample_with_metadata(
            model,
            batch_size=args.batch_size,
            length=length,
            mask_token_id=model_config.mask_token_id,
            pad_token_id=model_config.pad_token_id,
            vocab_size=model_config.vocab_size,
            steps=diffusion.sampling_steps,
            strategy=diffusion.strategy,
            temperature=diffusion.temperature,
            schedule=make_schedule(diffusion.schedule, diffusion.schedule_power),
            device=device,
            generator=generator,
            adaptive_config=diffusion.adaptive_decode_config(),
            collect_diagnostics=diffusion.strategy == "adaptive",
        )
        return sampling_result.forward_passes, sampling_result.stopping_reason

    for index in range(args.warmup):
        run(args.seed + index)
    elapsed = []
    for index in range(args.repeats):
        synchronize(device)
        start = time.perf_counter()
        calls, stopping_reason = run(args.seed + args.warmup + index)
        synchronize(device)
        measured_calls.append(calls)
        measured_stopping_reasons.append(stopping_reason)
        elapsed.append(time.perf_counter() - start)
    autoregressive_result: dict[str, object]
    if args.autoregressive_checkpoint is None:
        autoregressive_result = {
            "status": "pending",
            "reason": "No gpt2-nano checkpoint was supplied.",
        }
    else:
        if args.batch_size != 1:
            parser.error(
                "the staged gpt2-nano generator supports batch size 1 for comparison"
            )
        autoregressive = load_autoregressive_model(
            args.autoregressive_checkpoint,
            device,
        )
        if autoregressive.config.vocab_size + 2 != model_config.vocab_size:
            raise ValueError(
                "diffusion and autoregressive checkpoints do not share the "
                "gpt2-nano vocabulary"
            )
        if length + 1 > autoregressive.config.context_len:
            raise ValueError("autoregressive checkpoint context is too short")
        try:
            from src.model import generate as generate_autoregressive
        except ImportError as error:
            raise RuntimeError("gpt2-nano generation code is unavailable") from error
        for _ in range(args.warmup):
            generate_autoregressive(
                autoregressive,
                [0],
                length,
                use_cache=True,
                temperature=1.0,
                do_sample=False,
            )
        autoregressive_elapsed = []
        for _ in range(args.repeats):
            synchronize(device)
            start = time.perf_counter()
            generate_autoregressive(
                autoregressive,
                [0],
                length,
                use_cache=True,
                temperature=1.0,
                do_sample=False,
            )
            synchronize(device)
            autoregressive_elapsed.append(time.perf_counter() - start)
        autoregressive_result = {
            "status": "measured",
            "checkpoint": str(args.autoregressive_checkpoint),
            "median_seconds": statistics.median(autoregressive_elapsed),
            "repeats": args.repeats,
            "batch_size": 1,
            "generated_tokens_per_sample": length,
            "network_forward_passes": length,
            "parameter_count": sum(
                parameter.numel() for parameter in autoregressive.parameters()
            ),
            "decoding": "greedy with KV cache from a one-token prompt",
        }
    result = {
        "status": "measured",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "Apple Metal Performance Shaders"
                if device.type == "mps"
                else platform.processor() or platform.machine()
            ),
        },
        "diffusion": {
            "median_seconds": statistics.median(elapsed),
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "generated_tokens_per_sample": length,
            "network_forward_passes": statistics.median(measured_calls),
            "stopping_reasons": measured_stopping_reasons,
            "parameter_count": diffusion_parameters,
            "decoding": "fixed-length unconditional masked diffusion",
        },
        "autoregressive_gpt2_nano": autoregressive_result,
        "comparison_protocol": {
            "hardware": "same machine and precision",
            "tokenizer": "same existing gpt2-nano tokenizer through the local adapter",
            "workload": (
                "same generated token count and batch size; diffusion is "
                "unconditional while gpt2-nano starts from one token"
            ),
            "timing": "synchronize accelerators, warm up, then report median wall time",
            "quality": "held-out token metrics plus blinded samples; do not compare training loss directly",
        },
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        if args.manifest:
            record_artifact(
                args.manifest,
                args.output,
                kind="benchmark",
                status="measured",
                provenance={
                    "command": "python -m diffusion_lm_nano.benchmark",
                    "checkpoint": str(args.checkpoint),
                    "seed": args.seed,
                    "device": str(device),
                },
            )
    print(payload, end="")


if __name__ == "__main__":
    main()
