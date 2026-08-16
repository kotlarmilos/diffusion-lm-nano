#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e .
"$PYTHON" -m pip install vendor/gpt2_nano-0.2.0-py3-none-any.whl
"$PYTHON" scripts/prepare_publication_inputs.py
export TOKENIZER_PATH="${TOKENIZER_PATH:-data/bpe-tokenizer}"

./scripts/train_publication.sh

TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  TOKENIZER_ARGS=(--tokenizer-path "$TOKENIZER_PATH")
fi

"$PYTHON" -m diffusion_lm_nano.evaluate \
  --checkpoint artifacts/publication-checkpoint.pt \
  "${TOKENIZER_ARGS[@]}" \
  --corpus-path data/train.bin \
  --output artifacts/publication-evaluation.json \
  --manifest artifacts/manifest.json

"$PYTHON" -m diffusion_lm_nano.benchmark \
  --checkpoint artifacts/publication-checkpoint.pt \
  --autoregressive-checkpoint checkpoints/gpt2-nano.pt \
  --output artifacts/publication-benchmark.json \
  --manifest artifacts/manifest.json
