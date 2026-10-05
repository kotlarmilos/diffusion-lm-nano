---
title: "Testing early stopping in a diffusion language model"
thumbnail: /blog/assets/diffusion-lm-nano/thumbnail.png
authors:
- user: kotlarmilos
---

# Testing early stopping in a diffusion language model

## TL;DR

I tested adaptive stopping in a small masked diffusion language model. The
10,000-step checkpoint failed its basic quality check. Its held-out
masked-token accuracy is `0.36029` with cross entropy `5.14059`. The majority
`<|EOW|>` token occupies `0.36205` of the complete held-out sequences. The
saved unconditional sample contains only that token and decodes to spaces.

Calibration selected the setting that disables both early-stop rules. Every
held-out example used all 32 passes. The stopping method saved no work on this
checkpoint.

## What I tested

An autoregressive language model predicts one token at a time. A masked
diffusion model can predict every unresolved position in one pass, reveal some
positions, and repeat with more visible context.

A fixed pass budget is simple but can waste work when an example is already
resolved. I added two stopping signals to confidence-ranked decoding.

- Stop when the mean normalized entropy is low.
- Stop when mean confidence stops improving.

The adaptive policy can commit all remaining predictions after a minimum pass
count. It never reads the correct answer. Low uncertainty can still mean
confidently wrong, so the thresholds need calibration.

The study selects thresholds on one held-out window and evaluates the selected
setting on a later non-overlapping window. Selection prefers masked-token
accuracy, then exact sequence recovery, then fewer passes.

## Model and loss

The denoiser is a bidirectional Transformer. It receives a damaged token
sequence and a diffusion time between zero and one. A schedule chooses how many
tokens to replace with a dedicated mask token.

Masked positions can use visible context on both sides. The model adds token,
position, and time embeddings before the Transformer stack. It predicts a
clean-token distribution at every position.

Training averages cross entropy over corrupted positions.

$$
\mathcal{L}
= -\frac{1}{|S|}\sum_{i \in S}\log p_\theta(x_i \mid z_t,t)
$$

`S` is the set of masked positions and `z_t` is the damaged sequence. Visible
tokens supply context but do not contribute to the loss.

This is a simple masked-token objective. It does not reproduce the complete
MDLM likelihood objective or training recipe.

## Recorded scope

The source-bound run used a 12-layer, 44.75M-parameter model for 10,000
optimizer steps. Batch size was 32 and sequence length was 256. The checkpoint
records clean source commit `2bb34f0`.

The checkpoint does not preserve the training hardware or PyTorch runtime. The
later evaluation, adaptive study, and timing benchmark ran on one NVIDIA RTX
A6000 with PyTorch `2.4.1+cu124`. Those records do not describe training.

The input is a 10-million-token shard from the existing `gpt2-nano`
FineWeb-Edu preparation. Training uses the first 90 percent. A contiguous tail
is reserved for evaluation. The vocabulary has 9,157 content tokens plus
separate padding and mask IDs.

The exact training and adaptive study configurations are
`configs/publication.toml` and
`configs/adaptive_study_publication.toml`.

## The checkpoint collapsed

Evaluation at diffusion time `0.5` uses 256 held-out sequences and 19,390
randomly masked positions.

| Measurement | Result |
| --- | ---: |
| Masked-token accuracy | 0.36029 |
| Masked cross entropy | 5.14059 |
| Majority `<\|EOW\|>` fraction over complete sequences | 0.36205 |
| Unique token IDs in the unconditional sample | 1 |

The majority token is ID `9156`, the `<|EOW|>` token. The tokenizer decodes it
as a space. It occupies 23,727 of the 65,536 positions in the complete
evaluated sequences.

The accuracy and majority frequency use different position sets. Accuracy uses
the random masks. The frequency count uses every position. Their comparison is
descriptive rather than a paired test.

The saved sample gives direct collapse evidence. All 256 generated positions
are token `9156`. The output is 256 spaces. The masked accuracy does not
establish useful generation.

## Calibration disabled early stopping

The adaptive study compares 12 settings on 32 calibration sequences and tests
the selected setting on 128 later sequences. The initial diffusion time is
`0.75`.

The selected setting is

```text
min_passes = 4
entropy_threshold = 0
confidence_delta = 0
confidence_patience = 0
```

Entropy threshold zero disables the entropy rule. Confidence patience zero
disables the confidence-stall rule. The adaptive policy therefore follows
ordinary confidence decoding.

| Policy | Masked-token accuracy | Mean passes | Exact recovery |
| --- | ---: | ---: | ---: |
| Fixed order | 0.25597 | 32 | 0 of 128 |
| Fixed fraction | 0.25419 | 32 | 0 of 128 |
| Confidence | 0.36157 | 32 | 0 of 128 |
| Adaptive | 0.36157 | 32 | 0 of 128 |

Confidence and adaptive decoding both recover 7,334 of 20,284 initially masked
tokens. Every example uses all 32 passes. Mean time per example is `0.12537`
seconds for confidence and `0.13782` seconds for adaptive. These values include
Python overhead.

This result is limited by the weak checkpoint. It shows that the tested rules
did not help under this calibration procedure. It does not show that adaptive
stopping cannot work with a stronger denoiser.

## The timing claim is narrow

I also timed diffusion and the existing `gpt2-nano` checkpoint on the same
later evaluation GPU. Both paths produce 256 tokens at batch size one. The
table reports medians from five timed repeats.

| Decoder | Parameters | Forward passes | Median seconds |
| --- | ---: | ---: | ---: |
| Diffusion | 44,749,824 | 62 full-sequence passes | 0.25060 |
| `gpt2-nano` with KV cache | 44,062,661 | 256 cached passes | 1.40847 |

The measured ratio is `5.62x` for this exact fixed-length workload. The
diffusion path starts from an empty masked sequence. The autoregressive path
starts from one token. Their objectives differ. Output quality is not matched,
and the diffusion sample is blank.

This is a bounded timing result. It is not evidence that the collapsed
diffusion model is a faster useful language model.

## Evidence and reproduction

Release `b38dbab` preserves the measured JSON files, their exact configurations,
and a manifest with hashes. The raw files are unchanged.

The 179 MB diffusion checkpoint stays outside Git. Its SHA-256 is
`96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`.

Prepare and verify the recorded inputs after placing the checkpoint at
`artifacts/publication-checkpoint.pt`.

```sh
python scripts/prepare_publication_inputs.py
python scripts/validate_manifest.py --require-all
```

Run a new held-out evaluation without replacing the recorded evidence.

```sh
python -m diffusion_lm_nano.evaluate \
  --checkpoint artifacts/publication-checkpoint.pt \
  --corpus-path data/train.bin \
  --device cuda \
  --output artifacts/reproduced-evaluation.json
```

Run the adaptive study with the exact publication configuration and a separate
output path.

```sh
python -m diffusion_lm_nano.adaptive_study \
  --config configs/adaptive_study_publication.toml \
  --output artifacts/reproduced-adaptive-study.json
```

Recompute the majority-token control.

```sh
python scripts/analyze_publication_baseline.py \
  --output artifacts/reproduced-majority-control.json
```

The raw measured evidence remains in these files.

- `artifacts/publication-evaluation.json`
- `artifacts/publication-majority-control.json`
- `artifacts/publication-sample.json`
- `artifacts/adaptive-decoding-publication.json`
- `artifacts/publication-benchmark.json`

## Conclusion

The stopping experiment is implemented and measured, but the checkpoint fails
the more basic generation test. It collapses to the majority end-of-word token.
Calibration responds by disabling both early-stop rules and using the full pass
budget for every held-out example.

The useful result is the failure boundary. A stopping study should come after
the denoiser beats simple baselines and produces non-collapsed samples.

## References

1. Austin and collaborators. *Structured Denoising Diffusion Models in Discrete
   State-Spaces*. arXiv:2107.03006.
2. Sahoo and collaborators. *Simple and Effective Masked Diffusion Language
   Models*. arXiv:2406.07524.

The code and measured evidence are available in
[diffusion-lm-nano on GitHub](https://github.com/kotlarmilos/diffusion-lm-nano)
and
[diffusion-lm-nano on Hugging Face](https://huggingface.co/kotlarmilos/diffusion-lm-nano).
