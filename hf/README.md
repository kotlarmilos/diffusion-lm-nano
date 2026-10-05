---
library_name: pytorch
pipeline_tag: fill-mask
tags:
  - diffusion
  - language-modeling
  - masked-diffusion
  - research
---

# diffusion-lm-nano

This card describes the measured checkpoint from release `b38dbab`. The model
is a 44.75M-parameter bidirectional Transformer trained to predict masked
tokens. It is preserved as a collapse and stopping-study result, not as a
usable language model.

## Result

The 10,000-step checkpoint has held-out masked-token accuracy `0.36029` and
cross entropy `5.14059`. The majority `<|EOW|>` token occupies `0.36205` of the
same complete held-out sequences. The saved 256-token unconditional sample
contains only `<|EOW|>` and decodes to spaces.

Adaptive calibration tested 12 settings on 32 sequences and evaluated the
selected setting on a separate 128-sequence window. It selected entropy
threshold zero and confidence patience zero. Both stopping rules were disabled.
Every held-out example used all 32 passes.

The recorded timing comparison measured `0.25060` seconds for the diffusion
path and `1.40847` seconds for the autoregressive path. The `5.62x` ratio is
limited to batch size one, 256 output tokens, and the exact recorded execution
paths. Output quality was not matched and the diffusion sample was blank.

## Provenance

The checkpoint SHA-256 is
`96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`.
It records source commit `2bb34f0516d531acfbcd5df8ee66c5e8cd80d1b6`
and a clean Git state at training start.

The checkpoint does not record the training hardware or PyTorch runtime. Later
held-out evaluation, adaptive decoding, and timing ran on one NVIDIA RTX A6000
with PyTorch `2.4.1+cu124`.

The input shard and tokenizer come from `kotlarmilos/gpt2-nano` at revision
`ee547b4112bce2d2557dd0975038035bab6c86eb`. Exact hashes are in
`artifacts/manifest.json`.

## Intended use

Use this checkpoint to inspect majority-token collapse, reproduce the held-out
controls, and study the adaptive stopping implementation. Do not use it for
text generation, factual tasks, or deployment.

Only load checkpoint files from trusted sources. PyTorch checkpoint files can
contain unsafe serialized data.

The implementation is available at
`https://github.com/kotlarmilos/diffusion-lm-nano`.
