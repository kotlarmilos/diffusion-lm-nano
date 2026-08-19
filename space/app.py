"""Gradio visualization for iterative unmasking from a required checkpoint."""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr
import torch

from diffusion_lm_nano.artifacts import sha256
from diffusion_lm_nano.config import DiffusionConfig, ModelConfig
from diffusion_lm_nano.model import MaskedDiffusionTransformer
from diffusion_lm_nano.sampler import sample
from diffusion_lm_nano.schedules import make_schedule
from diffusion_lm_nano.tokenizer_data import (
    ByteTokenizer,
    GPT2NanoTokenizerAdapter,
    Tokenizer,
)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")


def load_model() -> tuple[
    MaskedDiffusionTransformer,
    ModelConfig,
    DiffusionConfig,
    Tokenizer,
]:
    if not CHECKPOINT_PATH:
        raise RuntimeError("Set CHECKPOINT_PATH to a trusted diffusion-lm-nano checkpoint.")
    path = Path(CHECKPOINT_PATH)
    if not path.is_file():
        raise RuntimeError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    tokenizer_kind = checkpoint.get("tokenizer", {}).get("kind")
    if tokenizer_kind == "ByteTokenizer":
        tokenizer: Tokenizer = ByteTokenizer()
    elif tokenizer_kind == "GPT2NanoTokenizerAdapter":
        if not TOKENIZER_PATH:
            raise RuntimeError(
                "Set TOKENIZER_PATH to a directory containing vocab.json and "
                "merges.json for this checkpoint."
            )
        training_inputs = checkpoint.get("training_inputs")
        expected_hash = (
            training_inputs.get("tokenizer_sha256")
            if isinstance(training_inputs, dict)
            else None
        )
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(
                "Checkpoint does not record an immutable tokenizer hash."
            )
        if sha256(TOKENIZER_PATH) != expected_hash:
            raise RuntimeError("Tokenizer files do not match the checkpoint.")
        tokenizer = GPT2NanoTokenizerAdapter(TOKENIZER_PATH)
    else:
        raise RuntimeError(f"Unsupported checkpoint tokenizer: {tokenizer_kind!r}")
    if (
        tokenizer.vocab_size != model_config.vocab_size
        or tokenizer.pad_token_id != model_config.pad_token_id
        or tokenizer.mask_token_id != model_config.mask_token_id
    ):
        raise RuntimeError("Checkpoint and tokenizer vocabulary settings do not match")
    model = MaskedDiffusionTransformer(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_config, diffusion_config, tokenizer


def render_tokens(
    token_ids: list[int],
    mask_token_id: int,
    tokenizer: Tokenizer,
) -> str:
    pieces = []
    for token_id in token_ids:
        if token_id == mask_token_id:
            pieces.append('<span class="mask">■</span>')
        else:
            try:
                pieces.append(html.escape(tokenizer.decode([token_id])))
            except ValueError:
                pieces.append(f'<span class="special">[{token_id}]</span>')
    return "".join(pieces)


def generate(length: int, steps: int, strategy: str, seed: int) -> str:
    model, model_config, diffusion_config, tokenizer = load_model()
    length = int(length)
    steps = int(steps)
    if not 1 <= length <= model_config.max_length:
        raise gr.Error(f"Length must be between 1 and {model_config.max_length}.")
    if (
        strategy == "adaptive"
        and steps < diffusion_config.adaptive_min_passes
    ):
        raise gr.Error(
            "Adaptive steps must be at least the checkpoint adaptive minimum."
        )
    frames = [
        "<li><strong>Start</strong><pre>"
        + render_tokens(
            [model_config.mask_token_id] * length,
            model_config.mask_token_id,
            tokenizer,
        )
        + "</pre></li>"
    ]

    def capture(step: int, tokens: torch.Tensor, confidence: torch.Tensor) -> None:
        rendered = render_tokens(
            tokens[0].tolist(),
            model_config.mask_token_id,
            tokenizer,
        )
        remaining = int(tokens[0].eq(model_config.mask_token_id).sum())
        frames.append(
            f"<li><strong>Step {step}</strong> · {remaining} masks remain"
            f"<pre>{rendered}</pre></li>"
        )

    sample(
        model,
        batch_size=1,
        length=length,
        mask_token_id=model_config.mask_token_id,
        pad_token_id=model_config.pad_token_id,
        vocab_size=model_config.vocab_size,
        steps=steps,
        strategy=strategy,  # type: ignore[arg-type]
        temperature=diffusion_config.temperature,
        schedule=make_schedule(
            diffusion_config.schedule, diffusion_config.schedule_power
        ),
        generator=torch.Generator().manual_seed(int(seed)),
        callback=capture,
        adaptive_config=diffusion_config.adaptive_decode_config(),
    )
    return "<ol class='trajectory'>" + "".join(frames) + "</ol>"


with gr.Blocks(css=".mask{color:#999}.special{color:#a33}pre{white-space:pre-wrap}") as demo:
    gr.Markdown(
        "# diffusion-lm-nano\n"
        "This Space requires a compatible checkpoint through `CHECKPOINT_PATH`. "
        "Publication checkpoints also require the matching tokenizer through "
        "`TOKENIZER_PATH`. It displays every parallel reveal step.\n\n"
        "**Measured result:** The 10,000-step publication checkpoint collapsed "
        "onto the end-of-word token and produces blank output. This interface "
        "visualizes that failure rather than presenting a usable language model."
    )
    with gr.Row():
        length = gr.Slider(1, 256, value=24, step=1, label="Token length")
        steps = gr.Slider(1, 128, value=6, step=1, label="Denoising steps")
    with gr.Row():
        strategy = gr.Radio(
            ["fixed_fraction", "fixed_order", "confidence"],
            value="confidence",
            label="Reveal strategy",
        )
        seed = gr.Number(value=7, precision=0, label="Seed")
    button = gr.Button("Generate")
    output = gr.HTML()
    button.click(generate, [length, steps, strategy, seed], output)


if __name__ == "__main__":
    demo.launch()
