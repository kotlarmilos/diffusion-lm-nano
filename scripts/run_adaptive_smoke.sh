#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

"$PYTHON" -m diffusion_lm_nano.train \
  --config configs/decoding_signal.toml \
  --checkpoint artifacts/decoding-signal-checkpoint.pt \
  --sample artifacts/decoding-signal-sample.json \
  --manifest artifacts/manifest.json

"$PYTHON" -m diffusion_lm_nano.adaptive_study \
  --config configs/adaptive_study_smoke.toml
