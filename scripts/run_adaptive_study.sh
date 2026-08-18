#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-configs/adaptive_study_publication.toml}"
cd "$ROOT"

"$PYTHON" -m diffusion_lm_nano.adaptive_study --config "$CONFIG"
