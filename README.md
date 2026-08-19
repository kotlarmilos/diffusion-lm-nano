# Building a diffusion language model from scratch

This repository is the runnable companion to `article.md`. It implements a small masked discrete diffusion language model in the D3PM and MDLM family. The denoiser is a bidirectional Transformer conditioned on continuous diffusion time. Training masks tokens under a configurable schedule and computes cross entropy only at corrupted positions. Generation starts from masks and reveals several positions per network pass, with optional confidence-calibrated early finalization.

The repository is a measured release candidate. A clean 10,000-step RTX A6000
run completed, but the publication checkpoint collapsed onto the end-of-word
token. The adaptive stopping study selected no early-stop rule. A matched-host
timing benchmark is included as a narrow systems result, not as evidence of
language quality or overall model superiority.

## Run the deterministic smoke path

Python 3.11 and PyTorch 2.2 or newer are required.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m diffusion_lm_nano.train \
  --config configs/smoke.toml \
  --checkpoint artifacts/smoke-checkpoint.pt \
  --sample artifacts/smoke-sample.json
```

The smoke configuration fixes the seed, uses CPU execution, disables dropout, selects greedy decoding, and runs four optimizer steps. It is a pipeline check, not an experiment.

Evaluation and timing are separate so measured files can retain their provenance.

```sh
python -m diffusion_lm_nano.evaluate \
  --checkpoint artifacts/smoke-checkpoint.pt \
  --output artifacts/smoke-evaluation.json

python -m diffusion_lm_nano.benchmark \
  --checkpoint artifacts/smoke-checkpoint.pt \
  --output artifacts/smoke-benchmark.json
```

## Core layout

`src/diffusion_lm_nano/schedules.py` defines linear, cosine, and power mask schedules. `corruption.py` applies absorbing mask corruption. `model.py` contains the noncausal Transformer and timestep embedding. `objective.py` excludes clean tokens from loss. `sampler.py` implements fixed-fraction, fixed-order, confidence-ranked, and adaptive confidence-ranked reveal. `adaptive_study.py` implements the calibrated matched study.

## Adaptive reveal scheduling

The `adaptive` strategy uses the same confidence-ranked quota as `confidence` unless an early-finalization rule fires. After `adaptive_min_passes`, it commits all current predictions when mean normalized entropy over unresolved positions is below `adaptive_entropy_threshold`, or when mean unresolved confidence improves by less than `adaptive_confidence_delta` for `adaptive_confidence_patience` consecutive checks. Threshold zero and patience zero disable the corresponding rule. The sampler returns actual forward passes, a stopping reason, and per-pass confidence, entropy, improvement, reveal, and remaining-mask fields through `sample_with_metadata` and `sample_from_tokens_with_metadata`.

Run the focused synthetic smoke protocol. The wrapper first regenerates the
ignored development checkpoint so the command works from a clean checkout:

```sh
./scripts/run_adaptive_smoke.sh
```

The publication protocol uses disjoint calibration and held-out windows from the same contiguous corpus tail. It preregisters an adaptive grid, selects only on calibration masked-token reconstruction accuracy and exact recovery with realized passes as a later tie-breaker, then evaluates fixed-order, fixed-fraction, confidence, and the selected adaptive policy on identical corruptions and decoding seeds.

```sh
./scripts/run_adaptive_study.sh configs/adaptive_study_publication.toml
```

Calibration selected zero entropy and confidence thresholds, which disable early
stopping. Confidence and adaptive decoding both reached `0.36157` held-out
masked-token accuracy, recovered zero complete sequences, and used all 32 passes
on every example. Adaptive added Python overhead without reducing work.

## Existing gpt2-nano data

The adapter consumes references to existing local artifacts. It never copies them into this repository.

`GPT2NanoTokenizerAdapter` loads the base project's custom `vocab.json` and
`merges.json`, then reserves two IDs after its 9,157-token vocabulary for
padding and masking. `BinaryTokenDataset` validates the common shard header,
which contains 256 little-endian signed 32-bit integers. The header records
magic `20250320`, version `1`, and the payload token count. The payload then
stores exactly that many little-endian `uint16` token IDs. `JsonlTokenDataset`
accepts records with an `input_ids` integer list.

`configs/publication.toml` uses padding ID 9,157, mask ID 9,158, and model
vocabulary size 9,159. `scripts/prepare_publication_inputs.py` downloads the
base tokenizer and token shard from their recorded Hugging Face locations and
verifies their SHA-256 digests before training.

No private `gpt2-nano` artifact is required for the smoke path.

## Publication results

The clean checkpoint is bound to source commit `2bb34f0`, has SHA-256
`96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`,
and records `git_dirty: false` at training start.

| Measurement | Result |
| --- | ---: |
| Held-out sequences | 256 |
| Randomly masked tokens | 19,390 |
| Masked-token accuracy | 0.36029 |
| Masked cross entropy | 5.14059 |
| Majority `<\|EOW\|>` fraction over the same sequences | 0.36205 |
| Unique token IDs in the 256-token unconditional sample | 1 |

All generated positions are token `9156`, the `<|EOW|>` token that decodes as a
space. The checkpoint is therefore a documented collapse result rather than a
usable language model. Reproduce the majority-token control after downloading
the verified inputs.

```sh
PYTHONPATH=src python scripts/analyze_publication_baseline.py
```

On the same RTX A6000, the 44.75M-parameter diffusion decoder took a median
`0.25060` seconds for 256 tokens using 62 full-sequence passes. The
44.06M-parameter `gpt2-nano` decoder took `1.40847` seconds using 256 cached
passes. The `5.62x` ratio applies only to this batch-one fixed-length workload.
The diffusion output is blank, the objectives differ, and the autoregressive
path starts from one token, so this result does not establish model superiority.

## FineWeb-Edu provenance

The publication checkpoint uses the first 10-million-token shard from the
existing `kotlarmilos/gpt2-nano` model repository at pinned revision
`ee547b4112bce2d2557dd0975038035bab6c86eb`. That project prepared its custom
BPE tokens from a FineWeb-Edu subset. This repository does not fetch raw
FineWeb-Edu documents, transform them, or redistribute them.

The FineWeb-Edu dataset card identifies the collection license as ODC-By 1.0. Web documents can also carry rights and usage conditions from their original sources. Anyone preparing a local training split must review the current dataset card, preserve required attribution, respect source restrictions, and record the exact dataset revision and filtering procedure. There is no dataset card here because this project creates no derived dataset.

## Fair autoregressive comparison

A comparison with autoregressive gpt2-nano must hold tokenizer, data split, parameter budget, training-token budget, batch size, precision, hardware, prompt set, generated token count, and random seeds constant. Both systems need the same warmup and accelerator synchronization. Report median wall time and peak memory. Report the number of network forward passes because diffusion spends one pass per denoising step while autoregression generally spends one cached step per generated token.

Quality needs held-out metrics appropriate to each objective and blinded sample
review. Diffusion masked-token cross entropy is not directly comparable with
next-token cross entropy. The measured benchmark loads the exact staged
`gpt2-nano` implementation and records that diffusion is fixed-length and
unconditional while the autoregressive path starts from one prompt token.

## Publication run

Review the external data path and hardware settings first, then run the script.

```sh
./scripts/train_publication.sh
```

To run the full publication workflow on Colab with verified input preparation,
training, evaluation, adaptive decoding study, and benchmarking in one command:

```sh
./scripts/colab_publication.sh
```

`artifacts/manifest.json` distinguishes measured outputs from simulations and
pending work. The repository tracks the measured JSON evidence and the small
development checkpoint so a clean checkout can verify the reported results.
The 179 MB publication checkpoint remains ignored because it belongs in the
Hugging Face model repository, not in Git history.

## Space

`space/app.py` visualizes every reveal step. It does not bundle a model. Set `CHECKPOINT_PATH` to a compatible checkpoint and install `space/requirements.txt` before launching the app.

## License

Code and original documentation are available under the MIT License. External model, tokenizer, and dataset artifacts retain their own terms.
