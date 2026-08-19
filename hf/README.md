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

This model card describes a measured 44.75M-parameter masked discrete diffusion
checkpoint. The training run completed cleanly, but the model collapsed onto
the end-of-word token. The checkpoint is prepared as negative experimental
evidence and a reproducible systems artifact, not as a usable language model.

## Model description

The model is a small bidirectional Transformer denoiser for absorbing-state masked discrete diffusion. It receives token ids and normalized diffusion time, then predicts clean-token logits at every position. Training loss includes only positions selected by forward corruption. Sampling starts from masks and reveals positions iteratively with random fixed-fraction selection, fixed-order selection, or confidence ranking.

## Intended use

The implementation supports education, architecture experiments, failure
analysis, and controlled systems measurements against a similarly sized
autoregressive baseline. The measured checkpoint is not intended for text
generation, deployment, or factual question answering.

## Training data

The checkpoint used the first 10-million-token `gpt2-nano` shard at revision
`ee547b4112bce2d2557dd0975038035bab6c86eb`. The shard SHA-256 is
`f08410ea036cc56b78ffc3f6edbf94f5a1d469aeb7c708bb56e4ce544623a69e`.
It uses the 9,157-token custom BPE vocabulary from the same revision.
`gpt2-nano` prepared the tokens from a FineWeb-Edu subset. FineWeb-Edu
identifies ODC-By 1.0 as the collection license. Original web documents can
retain additional rights and conditions.

## Evaluation

The clean source-bound checkpoint used 10,000 optimizer steps on an NVIDIA RTX
A6000 with PyTorch `2.4.1+cu124`.

| Measurement | Result |
| --- | ---: |
| Held-out sequences | 256 |
| Randomly masked tokens | 19,390 |
| Masked-token accuracy | 0.36029 |
| Masked cross entropy | 5.14059 |
| Majority `<\|EOW\|>` fraction over the same sequences | 0.36205 |
| Exact sequences recovered in the adaptive study | 0 of 128 |

The unconditional sample contains token `9156` at all 256 positions. This token
is `<|EOW|>` and decodes as a space. The sample is therefore blank. The
majority-token fraction uses every token in the evaluated sequences while
masked accuracy uses randomly corrupted positions, so it is a control rather
than an exact paired baseline. Together with the one-token sample support, it
shows that the checkpoint is collapsed.

The adaptive study calibrated 12 configurations on 32 sequences and evaluated
the selected configuration on a disjoint 128-sequence window. Calibration
selected zero entropy and confidence thresholds, which disable early stopping.
Adaptive and ordinary confidence decoding both reached `0.36157` masked-token
accuracy and used all 32 passes for every example. Adaptive stopping provided
no quality gain or pass reduction.

The same-host batch-one timing benchmark measured `0.25060` seconds for 256
diffusion tokens using 62 full-sequence passes and `1.40847` seconds for 256
`gpt2-nano` tokens using 256 cached passes. The nominal ratio is `5.62x`.
Diffusion is unconditional and fixed-length, `gpt2-nano` starts from one token,
the objectives differ, and diffusion quality failed. This timing is only a
systems measurement for the recorded workload.

A development-scale local extension run in `artifacts/decoding-extension.json`
uses a synthetic copy-pattern corpus, `artifacts/decoding-signal-checkpoint.pt`,
seed `15`, and initial timestep `0.75`. Confidence decoding records `156/476`,
`164/476`, `173/476`, `160/476`, `154/476`, and `154/476` correct initially
masked tokens at reveal budgets `1`, `2`, `3`, `4`, `6`, and `12`. Fixed-order
decoding records `156/476`, `151/476`, `152/476`, `156/476`, `150/476`, and
`151/476`. The result shows a development-scale knee at three reveal steps and
an equal-budget confidence ordering advantage after the one-step case.

## Limitations

This checkpoint produces blank output and should not be used as a language
model. The 10,000-step recipe was not sufficient to avoid majority-token
collapse. Fixed-length unconditional sampling does not represent a production
generation interface. Results depend heavily on data preparation, schedule,
training budget, corruption weighting, denoising steps, and reveal strategy.
The development-scale extension uses a synthetic local corpus and does not
estimate open-domain language quality.

## Checkpoint use

The checkpoint file is `publication-checkpoint.pt`, with SHA-256
`96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`.
It records source commit `2bb34f0516d531acfbcd5df8ee66c5e8cd80d1b6`
with a clean Git state captured at training start. Set `CHECKPOINT_PATH` and the
matching tokenizer path for the included Space. Only load checkpoints from
trusted sources because PyTorch checkpoint files can contain unsafe serialized
data.
