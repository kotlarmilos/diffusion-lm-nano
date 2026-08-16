---
license: mit
library_name: pytorch
pipeline_tag: fill-mask
tags:
  - diffusion
  - language-modeling
  - masked-diffusion
  - research
---

# diffusion-lm-nano

This model card describes the architecture and expected checkpoint format in the repository. A tiny local smoke checkpoint verifies wiring only. No publication model weights are bundled with the release candidate.

## Model description

The model is a small bidirectional Transformer denoiser for absorbing-state masked discrete diffusion. It receives token ids and normalized diffusion time, then predicts clean-token logits at every position. Training loss includes only positions selected by forward corruption. Sampling starts from masks and reveals positions iteratively with random fixed-fraction selection, fixed-order selection, or confidence ranking.

## Intended use

The implementation supports education, architecture experiments, and controlled comparison with a similarly sized autoregressive baseline. It is not intended for deployment or factual question answering.

## Training data

The smoke configuration uses a short public string embedded in the source only to validate the pipeline. The publication configuration expects operator-provided local token data. The repository suggests FineWeb-Edu as one possible source but does not include or derive a dataset.

FineWeb-Edu is published by HuggingFaceFW. Its dataset card identifies ODC-By 1.0 for the collection. Original web documents can retain additional rights and conditions. A model publisher must replace this section with the exact dataset revision, subset, filters, and attribution record used for the actual checkpoint.

## Evaluation

No publication-scale comparative quality or latency result is reported.
`diffusion_lm_nano.evaluate` measures masked-token accuracy and denoising cross
entropy across supplied held-out corpus sequences. `diffusion_lm_nano.benchmark`
measures local decoding wall time and can load the exact staged `gpt2-nano`
implementation for a same-process comparison. The artifact records that the
diffusion workload is fixed-length and unconditional while the autoregressive
workload starts from one prompt token.

A development-scale local extension run in `artifacts/decoding-extension.json`
uses a synthetic copy-pattern corpus, `artifacts/decoding-signal-checkpoint.pt`,
seed `15`, and initial timestep `0.75`. Confidence decoding records `156/476`,
`164/476`, `173/476`, `160/476`, `154/476`, and `154/476` correct initially
masked tokens at reveal budgets `1`, `2`, `3`, `4`, `6`, and `12`. Fixed-order
decoding records `156/476`, `151/476`, `152/476`, `156/476`, `150/476`, and
`151/476`. The result shows a development-scale knee at three reveal steps and
an equal-budget confidence ordering advantage after the one-step case.

## Limitations

Masked generation can produce incoherent, biased, memorized, or unsafe text. Byte tokenization is only a smoke-test convenience. Fixed-length unconditional sampling does not represent a production generation interface. Results depend heavily on data preparation, schedule, number of denoising steps, and reveal strategy. The development-scale extension uses a synthetic local corpus and does not estimate open-domain language quality.

## Checkpoint use

Create a checkpoint with the training CLI, then set `CHECKPOINT_PATH` for the included Space. Only load checkpoints from trusted sources because PyTorch checkpoint files can contain unsafe serialized data.
