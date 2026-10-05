# diffusion-lm-nano

This repository preserves one measured masked diffusion experiment from release
`b38dbab`.

The 10,000-step checkpoint reached held-out masked-token accuracy `0.36029` and
cross entropy `5.14059`. The majority `<|EOW|>` token occupies `0.36205` of the
complete held-out sequences. The saved unconditional sample contains only that
token and decodes to spaces.

Adaptive calibration selected thresholds that disable both early-stop rules.
Every held-out example used all 32 passes. The recorded `5.62x` timing ratio
applies only to the exact batch-one, 256-token workload. It does not compare
matched output quality.

The checkpoint does not preserve the training hardware or runtime. The NVIDIA
RTX A6000 and PyTorch `2.4.1+cu124` records apply only to later evaluation,
adaptive decoding, and timing.

Read `article.md` for the method, results, and limits.

## Retained evidence

The release keeps these measured files.

- `artifacts/publication-evaluation.json`
- `artifacts/publication-majority-control.json`
- `artifacts/publication-sample.json`
- `artifacts/adaptive-decoding-publication.json`
- `artifacts/publication-benchmark.json`
- `configs/publication.toml`
- `configs/adaptive_study_publication.toml`

`artifacts/manifest.json` records their hashes and input provenance. The 179 MB
checkpoint remains outside Git at `artifacts/publication-checkpoint.pt`. Its
SHA-256 is
`96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`.

## Prepare and verify inputs

Use Python 3.11 or newer.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Place `publication-checkpoint.pt` at
`artifacts/publication-checkpoint.pt`. Use the release checkpoint file or the
matching Hugging Face artifact repository. Then download the pinned corpus,
tokenizer, and autoregressive checkpoint and verify every input.

```sh
python scripts/prepare_publication_inputs.py
python scripts/validate_manifest.py --require-all
```

Use `--skip-checkpoint` to prepare the smaller external inputs before the
diffusion checkpoint is available. Use `--verify-only` to prevent downloads.

## Run the focused checks

```sh
python -m unittest -v tests.test_release
python -m diffusion_lm_nano.evaluate \
  --checkpoint artifacts/publication-checkpoint.pt \
  --corpus-path data/train.bin \
  --device cuda \
  --output artifacts/reproduced-evaluation.json
python -m diffusion_lm_nano.adaptive_study \
  --config configs/adaptive_study_publication.toml \
  --output artifacts/reproduced-adaptive-study.json
python scripts/analyze_publication_baseline.py \
  --output artifacts/reproduced-majority-control.json
```

The timing comparison also needs the retained `gpt2-nano` wheel.

```sh
python -m pip install vendor/gpt2_nano-0.2.0-py3-none-any.whl
python -m diffusion_lm_nano.benchmark \
  --checkpoint artifacts/publication-checkpoint.pt \
  --autoregressive-checkpoint checkpoints/gpt2-nano.pt \
  --length 256 \
  --batch-size 1 \
  --repeats 5 \
  --warmup 1 \
  --device cuda \
  --output artifacts/reproduced-benchmark.json
```

New timing values depend on the machine and runtime. The retained JSON files
remain the evidence for the published claim.

The checkpoint and measured artifacts are published at
`https://huggingface.co/kotlarmilos/diffusion-lm-nano`.
