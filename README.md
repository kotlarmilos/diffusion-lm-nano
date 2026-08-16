# Building a diffusion language model from scratch

This repository is the runnable companion to `article.md`. It implements a small masked discrete diffusion language model in the D3PM and MDLM family. The denoiser is a bidirectional Transformer conditioned on continuous diffusion time. Training masks tokens under a configurable schedule and computes cross entropy only at corrupted positions. Generation starts from masks and reveals several positions per network pass.

The repository is a release candidate. No external tokenizer, dataset, or checkpoint is bundled, and no comparative benchmark is claimed. A measured smoke checkpoint can be generated locally to verify the pipeline.

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

`src/diffusion_lm_nano/schedules.py` defines linear, cosine, and power mask schedules. `corruption.py` applies absorbing mask corruption. `model.py` contains the noncausal Transformer and timestep embedding. `objective.py` excludes clean tokens from loss. `sampler.py` implements fixed-fraction random reveal and confidence-ranked reveal. The three CLI modules train, evaluate, and benchmark.

## Existing gpt2-nano data

The adapter consumes references to existing local artifacts. It never copies them into this repository.

`GPT2NanoTokenizerAdapter` loads the base project's custom `vocab.json` and
`merges.json`, then reserves two IDs after its 9,157-token vocabulary for
padding and masking. `BinaryTokenDataset` reads contiguous little-endian
`uint16` token IDs from the common `train.bin` layout. `JsonlTokenDataset`
accepts records with an `input_ids` integer list.

`configs/publication.toml` uses padding ID 9,157, mask ID 9,158, and model
vocabulary size 9,159. `scripts/prepare_publication_inputs.py` downloads the
base tokenizer and token shard from their recorded Hugging Face locations and
verifies their SHA-256 digests before training.

No private gpt2-nano artifact is required for the smoke path.

## FineWeb-Edu provenance

The publication configuration is designed for token blocks that an operator may derive locally from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). This repository does not fetch, transform, package, or redistribute FineWeb-Edu.

The FineWeb-Edu dataset card identifies the collection license as ODC-By 1.0. Web documents can also carry rights and usage conditions from their original sources. Anyone preparing a local training split must review the current dataset card, preserve required attribution, respect source restrictions, and record the exact dataset revision and filtering procedure. There is no dataset card here because this project creates no derived dataset.

## Fair autoregressive comparison

A comparison with autoregressive gpt2-nano must hold tokenizer, data split, parameter budget, training-token budget, batch size, precision, hardware, prompt set, generated token count, and random seeds constant. Both systems need the same warmup and accelerator synchronization. Report median wall time and peak memory. Report the number of network forward passes because diffusion spends one pass per denoising step while autoregression generally spends one cached step per generated token.

Quality needs held-out metrics appropriate to each objective and blinded sample
review. Diffusion masked-token cross entropy is not directly comparable with
next-token cross entropy. The benchmark accepts `--autoregressive-checkpoint`
and loads the exact staged `gpt2-nano` implementation. Without that argument it
records the baseline as pending rather than inserting a number from a different
machine or checkpoint. The artifact also states that diffusion is
fixed-length and unconditional while the autoregressive path starts from one
prompt token.

## Publication run

Review the external data path and hardware settings first, then run the script.

```sh
./scripts/train_publication.sh
```

To run the full publication workflow on Colab with verified input preparation,
training, evaluation, and benchmarking in one command:

```sh
./scripts/colab_publication.sh
```

`artifacts/manifest.json` distinguishes measured outputs from simulations and pending work. Generated checkpoints remain ignored by Git by default.

## Space

`space/app.py` visualizes every reveal step. It does not bundle a model. Set `CHECKPOINT_PATH` to a compatible checkpoint and install `space/requirements.txt` before launching the app.

## License

Code and original documentation are available under the MIT License. External model, tokenizer, and dataset artifacts retain their own terms.
