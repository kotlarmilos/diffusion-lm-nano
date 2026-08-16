#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

export PYTHONHASHSEED=2026
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  TOKENIZER_ARGS=(--tokenizer-path "$TOKENIZER_PATH")
fi

"$PYTHON" -m diffusion_lm_nano.train \
  --config configs/publication.toml \
  "${TOKENIZER_ARGS[@]}" \
  --checkpoint artifacts/publication-checkpoint.pt \
  --sample artifacts/publication-sample.json \
  --manifest artifacts/manifest.json
