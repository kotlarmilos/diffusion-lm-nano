---
title: "Building a diffusion language model from scratch"
thumbnail: /blog/assets/nano-diffusion-lm/thumbnail.png
authors:
- user: kotlarmilos
---

# Building a diffusion language model from scratch

## TL;DR

I have spent most of this series making language models smaller without hiding the
machinery. This project keeps the familiar parts of a transformer language model
and changes the central operation. We still tokenize text, still train a
Transformer with cross entropy, and still sample from token logits. The change is
that generation no longer has to be a strictly left-to-right process. Masked
discrete diffusion corrupts a clean sequence, trains a bidirectional Transformer to
restore it, and then generates by revealing masked positions over several parallel
passes.

The measured result is negative. A clean 10,000-step RTX A6000 run collapsed
onto the end-of-word token, and adaptive calibration selected no early-stop
rule. Diffusion completed one fixed-length timing workload `5.62x` faster than
`gpt2-nano`, but its generated sample is blank. The timing is a systems
measurement, not evidence of language quality or overall model superiority.

## The problem

An autoregressive model learns from a shifted sequence. Every position predicts the
token immediately to its right and a causal attention mask prevents access to later
positions. Repeat that one token at a time and text appears. Diffusion belongs next
in the series because it changes that central operation while keeping enough of the
surrounding system familiar. That difference is easier to understand when the model
is small enough to inspect.

## Intuition

### Corrupting text instead of shifting it

This project uses masked discrete diffusion. I take a clean token sequence and
choose a normalized time between zero and one. A schedule converts that time into a
mask probability. Each selected token becomes a dedicated mask token. The model
receives the damaged sequence and the time, then predicts the original values at
the damaged positions. At time zero, the sequence is clean. At time one, every
active token is masked. Between those endpoints, a linear, cosine, or power
schedule controls how quickly information disappears.

This is an absorbing-state process. Once a token enters the mask state during the
forward process, later corruption does not turn it into a different vocabulary
token. That simple transition is the useful part of the D3PM and MDLM line for a
nano implementation. We can express the marginal corruption directly without
building a dense transition matrix over the vocabulary.

### A Transformer that can look both ways

The denoiser is not a causal decoder. It is a bidirectional Transformer encoder. A
masked position can use visible evidence on its left and right, which is exactly
what denoising requires. Token and position embeddings provide the usual sequence
representation. A sinusoidal embedding represents continuous diffusion time. A small
multilayer perceptron projects that time embedding into the model width, and the
result is added at every sequence position. The Transformer therefore knows whether
it is solving a lightly damaged sequence or a nearly empty one. There is no
triangular attention mask. Padding still gets an attention mask, but visible content
can interact across the full sequence. The output projection shares weights with
the token embedding and produces one clean-token distribution at every position.

### Turning masks back into text

Generation begins with a sequence made entirely of mask tokens. The model predicts
all positions in parallel. The sampler commits some predictions, keeps the rest
masked, lowers the time value, and calls the model again. Repeating this loop
produces a complete sequence. The repository includes four reveal rules.
Fixed-fraction decoding reveals enough randomly selected masked positions to finish
within the remaining steps. Fixed-order decoding reveals masked positions from left
to right. Both still use the model predictions at those positions, but position
selection does not depend on confidence. Confidence-based decoding ranks the
sampled predictions and commits the most confident positions first. Low-confidence
positions remain masked and can be reconsidered with more context on the next pass.
Adaptive decoding uses the same ranking, then commits all remaining current
predictions when unresolved normalized entropy is low or unresolved confidence has
stalled for a configured patience after a minimum pass count. This resembles the
practical unmasking rule used by many masked generative systems.

## The math

The training target matters. Clean tokens are context, not prediction targets for
that sampled corruption. Their target value is the ignore index, so they contribute
no cross entropy. A batch with no selected tokens returns a differentiable zero
rather than a numerical error. The masked cross entropy is the unweighted
specialization of the discrete-diffusion ELBO for an absorbing-state process, which
is why a plain masked-token loss trains the denoiser correctly.

All reveal rules guarantee termination. The number revealed at each iteration is
at least the ceiling of the remaining masks divided by the remaining steps. The
final pass therefore has no masks left. The sampler also prevents the dedicated
mask and padding ids from being emitted as content. Temperature zero gives greedy
token choices. Positive temperature uses multinomial sampling with an explicit
generator. A fixed seed, zero dropout, and the smoke configuration make the
complete sampling path deterministic.

The contrast with autoregression is now concrete. An autoregressive model commits
one new position after another. With a key-value cache, each step processes the new
token while reusing earlier attention state. A length of sixty-four normally means
sixty-four sequential token decisions. The diffusion model updates many positions
in one network pass. A sixty-four-token result might use sixteen or sixty-four
denoising passes depending on the configuration. Each pass processes the whole
current sequence with bidirectional attention. The decisions within a pass are
parallel, but the passes remain sequential.

## In the code

The absorbing-state corruption lives in `src/diffusion_lm_nano/corruption.py`, and
the mask-probability schedules live in `src/diffusion_lm_nano/schedules.py`. The
masked cross entropy lives in `src/diffusion_lm_nano/objective.py`. The
bidirectional denoiser with sinusoidal time conditioning lives in
`src/diffusion_lm_nano/model.py`, the reveal rules in
`src/diffusion_lm_nano/sampler.py`, training in `src/diffusion_lm_nano/train.py`,
held-out evaluation in `src/diffusion_lm_nano/evaluate.py`, and paired timing in
`src/diffusion_lm_nano/benchmark.py`. The custom BPE adapter and the reserved ids
live in `src/diffusion_lm_nano/tokenizer_data.py`. `LEARNING.md` derives the
absorbing corruption, the ELBO specialization, and iterative unmasking with exact
line references.

## Results and honesty boundary

The publication checkpoint came from clean source commit `2bb34f0` and has
SHA-256 `96f78f1d51fab2a365142bf2a861feb109633a189658fd97864087e913c3a9db`.
The run used 10,000 optimizer steps, 10 million tokens from the pinned
`gpt2-nano` FineWeb-Edu shard, and the 9,157-token custom BPE vocabulary. The
checkpoint records `git_dirty: false` at training start.

### The quality result is negative

Evaluation at diffusion time `0.5` covered 256 contiguous held-out sequences
and 19,390 randomly masked tokens. Masked-token accuracy was `0.36029`, and
masked cross entropy was `5.14059`.

That accuracy does not establish useful denoising. Token `9156`, the
`<|EOW|>` token that decodes as a space, occupies `0.36205` of all 65,536
positions in the same held-out sequences. This constant-token control uses all
positions while evaluation uses the randomly corrupted positions, so the two
figures are not an exact paired comparison. Their near equality is still an
important warning. The unconditional sample confirms the failure. All 256
generated positions are token `9156`, producing 256 spaces and no readable
text. I therefore treat this checkpoint as collapsed and make no language
quality claim.

### Adaptive stopping rejected early stopping

The adaptive study used 32 calibration sequences and a later disjoint window of
128 held-out sequences. Calibration compared 12 preregistered configurations
with a maximum of 32 passes. It selected `min_passes = 4`,
`entropy_threshold = 0`, `confidence_delta = 0`, and
`confidence_patience = 0`. The zero thresholds disable both early-stop rules.

On held-out data, confidence and adaptive decoding were numerically identical
at the aggregate level. Both recovered `7,334/20,284` initially masked tokens,
for accuracy `0.36157`. Both recovered zero complete sequences. Every example
used all 32 forward passes and stopped at the maximum. Fixed-order and
fixed-fraction accuracy were `0.25597` and `0.25419`. Adaptive mean wall time
was `0.13782` seconds per example versus `0.12537` for confidence, but this
field includes Python control overhead and does not support a decoder speed
claim. The experiment answers the extension question negatively. The calibrated
rules did not reduce work or improve reconstruction.

### The timing result is workload-specific

The same-host benchmark used an NVIDIA RTX A6000, PyTorch `2.4.1+cu124`,
batch size one, 256 output tokens, five timed repeats, and float32 execution.

| Decoder | Parameters | Network passes | Median seconds |
| --- | ---: | ---: | ---: |
| Diffusion | 44,749,824 | 62 full-sequence passes | 0.25060 |
| `gpt2-nano` with KV cache | 44,062,661 | 256 cached passes | 1.40847 |

The nominal ratio is `5.62x` in favor of diffusion for this exact systems
workload. It is not a model-superiority result. Diffusion performs fixed-length
unconditional generation, while `gpt2-nano` starts from one token and follows a
different objective. The models do not have matched quality, the diffusion
sample is blank, and no blinded sample review was performed. The benchmark only
shows that 62 parallel full-sequence passes completed faster than 256 cached
autoregressive passes under this configuration.

The smoke path uses a byte tokenizer so the repository can run without downloading
data or a tokenizer. Bytes make a poor final vocabulary, but they make the smallest
experiment reproducible and public. `configs/smoke.toml` uses a two-layer model,
twenty-four byte positions, four optimizer steps, and six greedy reveal steps on
CPU. It checks wiring. It does not estimate model quality.

**Development-scale extension result.** `diffusion_lm_nano.decoding_study` now
measures reconstruction after deterministic corruption on a local copy-pattern
JSONL corpus. The setup uses `configs/decoding_signal.toml`,
`artifacts/decoding-signal-checkpoint.pt`, seed `15`, initial timestep `0.75`,
and reveal budgets `1`, `2`, `3`, `4`, `6`, and `12`. Confidence decoding records
`156/476`, `164/476`, `173/476`, `160/476`, `154/476`, and `154/476` correct
initially masked tokens. Fixed-order decoding records `156/476`, `151/476`,
`152/476`, `156/476`, `150/476`, and `151/476`. Accuracy rises from `0.3277` to
`0.3634` through three reveal steps, then falls and plateaus. This supports a
development-scale knee at three reveal steps rather than a claim that more steps
always help. Confidence ordering matches fixed order at one step and beats it at
every larger equal budget in this run. The corpus is synthetic and local, so this
result demonstrates decoding behavior, not publication-scale language quality.

The publication path meets the previous `gpt2-nano` project at the data boundary.
An adapter loads that project's custom BPE vocabulary and merge table. Two IDs after
its 9,157-token vocabulary are reserved for padding and masking. Dataset adapters
read the common shard format with a 1,024-byte metadata header followed by
little-endian unsigned sixteen-bit token IDs, or JSON lines containing token ID
arrays. The parser validates the magic value, version, declared count, exact
file size, endianness, and vocabulary range before training.

The publication checkpoint uses the pinned `gpt2-nano` shard rather than a new
derived dataset. The input-preparation script downloads the tokenizer, corpus,
and autoregressive checkpoint from immutable locations and verifies each
SHA-256. FineWeb-Edu identifies the collection license as ODC-By 1.0, while
original web documents can retain additional rights and conditions.

Every generated result has one of three states. Measured means a command actually
produced it. Simulated means it is synthetic demonstration data and cannot support
an empirical claim. Pending means the run has not happened. A command that lacks
its required checkpoint records the missing comparison as pending rather than
inserting an invented result. The Space contains the interface and trajectory
visualization, but it requires a real checkpoint path. The publication checkpoint
does not produce readable text because it collapsed, and the Space must display
that limitation rather than implying a successful text model.

Expected publication locations, not yet live:

- Source repository at https://github.com/kotlarmilos/diffusion-lm-nano
- Model repository at https://huggingface.co/kotlarmilos/diffusion-lm-nano
- Interactive Space at https://huggingface.co/spaces/kotlarmilos/diffusion-lm-nano

## References

- Jascha Sohl-Dickstein and collaborators, Deep Unsupervised Learning using
  Nonequilibrium Thermodynamics at https://arxiv.org/abs/1503.03585
- Jacob Austin and collaborators, Structured Denoising Diffusion Models in Discrete
  State-Spaces at https://arxiv.org/abs/2107.03006
- Subham Sekhar Sahoo and collaborators, Simple and Effective Masked Diffusion
  Language Models at https://arxiv.org/abs/2406.07524
- FineWeb-Edu dataset at
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
