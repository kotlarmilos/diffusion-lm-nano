# Building masked discrete diffusion language modeling from first principles

This note derives the model in this repository from the probability
factorization up to the sampler. Each derivation stays tied to the exact code
that implements it. The implementation is a masked discrete diffusion language
model, not an image diffusion model and not an autoregressive decoder.

## Motivation: why diffusion changes the generation problem

### Motivation

An autoregressive language model chooses one next token at a time. It turns a
sequence distribution into a product of conditional distributions over a growing
prefix.

```
x = (x_1, x_2, ..., x_L)
p_AR(x) = prod_{i=1}^L p(x_i | x_1, ..., x_{i-1})
```

This factorization gives a simple training target. Shift the sequence by one
position, hide all future tokens, and train position `i` to predict `x_{i+1}`.
It also gives a strict decoding order. Once the model writes token `i`, later
steps condition on that choice. A cached autoregressive decoder can make the
computation efficient, but it still makes `L` sequential token decisions for a
length `L` sample.

Masked diffusion starts from a different question. Instead of asking for the
next token after a prefix, it asks which clean sequence could have produced a
damaged sequence. The denoiser sees all positions at the same time and predicts
a clean-token distribution at every position.

```
damaged sequence z_t
diffusion denoiser p_theta(x_0,i | z_t, t) for every position i
```

This changes the generation loop. The model can update many unresolved
positions in one network pass. It can also leave uncertain positions masked and
reconsider them later with more visible context. In this repository, committed
positions stay committed, so revision means that still-masked positions get
new predictions after other positions become visible.

### Derivation

The autoregressive product rule follows from the ordinary chain rule of
probability.

```
p(x_1, ..., x_L)
  = p(x_1) * p(x_2 | x_1) * ... * p(x_L | x_1, ..., x_{L-1})
  = prod_{i=1}^L p(x_i | x_<i)
```

The masked diffusion objective starts from a latent corruption variable instead.
For a clean sequence `x_0`, sample a time `t` and a corrupted sequence `z_t`
from a known forward process `q_t`.

```
x_0 ~ p_data
t ~ p(t)
z_t ~ q_t(z_t | x_0)
p_theta predicts x_0 from z_t and t
```

The model does not need a causal mask because the prediction target is not the
next token. A masked position should use evidence on both sides. The likelihood
object at one corrupted time has the denoising form.

```
p_theta(x_0 | z_t, t)
  = prod_{i in S_t} p_theta(x_0,i | z_t, t)
```

Here `S_t` is the set of positions that the forward process selected for
masking. Clean visible positions already equal their targets, so the training
signal focuses on the coordinates that lost information.

### Where it lives in the code

`article.md` introduces the contrast directly. Lines 11 to 13 describe the
autoregressive shifted target and the masked diffusion target. Lines 23 to 27
state that the denoiser is a bidirectional Transformer with no triangular
attention mask.

`src/diffusion_lm_nano/model.py` implements that noncausal denoiser. The class
`MaskedDiffusionTransformer` says that it predicts clean tokens from a
partially masked sequence on lines 33 to 34. The forward path adds token,
position, and time embeddings on lines 83 to 86, passes the sequence through
`nn.TransformerEncoder` on line 88, and returns per-position vocabulary logits
on line 89. No causal mask enters this call. Only a padding mask can enter
through `src_key_padding_mask` on lines 87 to 88.

## Derivation: the absorbing mask forward process

### Motivation

A diffusion model needs a noising process that is known and easy to sample.
For text, Gaussian noise does not apply to token IDs because token IDs are
categories, not points in a continuous space. This repository uses an
absorbing-state process. A token either stays unchanged or becomes a dedicated
mask token.

The mask probability grows with normalized diffusion time `t`. At `t = 0`, the
sequence is clean. At `t = 1`, every active token is masked. Between those
endpoints, a schedule controls how quickly information disappears.

### Derivation

Let `V` be the content vocabulary and let `M` be the dedicated mask symbol. Let
`rho(t)` be the marginal probability of masking at time `t`.

```
0 <= t <= 1
0 <= rho(t) <= 1
rho(0) = 0
rho(1) = 1
```

For one active token `x_i in V`, the corruption distribution is categorical
with only two possible outcomes.

```
q_t(z_i = M   | x_i) = rho(t)
q_t(z_i = x_i | x_i) = 1 - rho(t)
q_t(z_i = v   | x_i) = 0 for every v not in {x_i, M}
```

For a full sequence, the code samples the positions independently once `t` is
chosen.

```
q_t(z | x) = prod_{i=1}^L q_t(z_i | x_i)
```

Padding positions can be excluded by an attention mask. If `a_i = 0`, the
position remains untouched.

```
q_t(z_i = x_i | x_i, a_i = 0) = 1
q_t(z_i = M   | x_i, a_i = 0) = 0
```

The term absorbing means that the mask state represents information that has
already disappeared. A Markov construction with monotone `rho` would keep `M`
as `M` at later times. The implementation samples the marginal distribution
directly, so it does not build or multiply a transition matrix.

The repository supplies three schedules.

```
linear: rho(t) = clamp(t, 0, 1)
cosine: rho(t) = clamp(1 - cos(clamp(t, 0, 1) * pi / 2), 0, 1)
power: rho(t) = clamp(t, 0, 1)^power
```

### Where it lives in the code

`src/diffusion_lm_nano/schedules.py` defines the schedule interface
`MaskSchedule.probability` on lines 13 to 15. `LinearSchedule.probability`
returns `t.clamp(0.0, 1.0)` on lines 18 to 21.
`CosineSchedule.probability` implements `1 - cos(t * pi / 2)` on lines 24 to
29. `PowerSchedule.probability` validates a positive power on lines 32 to 38
and returns the clamped power schedule on lines 40 to 41. `make_schedule`
selects the configured schedule on lines 48 to 55.

`src/diffusion_lm_nano/corruption.py` implements the forward corruption in
`corrupt_tokens`. Lines 22 to 27 validate the sequence shape and the batch time
shape. Lines 28 to 33 build or validate the active-token attention mask. Lines
35 to 38 compute `rho(t)` and expand it to the token matrix shape. Lines 39 to
45 draw independent uniform random values and select positions where the draw
falls below the mask probability. Lines 46 to 47 force exact endpoint behavior,
so probability zero selects nothing and probability one selects every active
position. Line 48 writes the mask token into selected positions. Line 49 writes
the original token into the target only at selected positions and writes the
ignore index elsewhere.

The tokenizer boundary matters because the mask symbol must be a real reserved
ID. `src/diffusion_lm_nano/tokenizer_data.py` gives the smoke byte tokenizer
`vocab_size = 258`, `pad_token_id = 256`, and `mask_token_id = 257` on lines
27 to 30. The publication adapter reads the existing `gpt2-nano` vocabulary,
sets `base_size = len(vocab)`, reserves `pad_token_id = base_size` and
`mask_token_id = base_size + 1`, and sets `vocab_size = base_size + 2` on
lines 63 to 69. For the recorded base vocabulary of 9,157 tokens, that gives
`pad_token_id = 9157`, `mask_token_id = 9158`, and `vocab_size = 9159`.
`configs/publication.toml` records those exact IDs on lines 1 to 10.

## Derivation: the masked denoising training objective

### Motivation

The denoiser receives the corrupted sequence and the time. Its job is not to
copy visible tokens. Visible tokens already carry their clean value through the
forward process. The useful target is the original token at each masked
position.

This gives a sparse cross entropy. The network still emits logits for every
position, but only the positions selected by corruption contribute to the loss.
That choice follows the probabilistic structure. If a coordinate is visible,
the posterior over the clean token is deterministic. If a coordinate is masked,
the posterior depends on the rest of the sequence and must be learned.

### Derivation

Start with the negative log likelihood of a clean data sequence and introduce
a known forward process `q`.

```
-log p_theta(x_0)
  <= E_q [
       -log p_theta(x_0 | z_1)
       + sum_{k=2}^K KL(
           q(z_{k-1} | z_k, x_0)
           ||
           p_theta(z_{k-1} | z_k)
         )
     ]
```

This is the negative form of the discrete-diffusion evidence lower bound. The
forward process is fixed, so learning enters through the reverse distributions.
For an absorbing mask process, a reverse step only has uncertainty at
coordinates that are currently masked. A visible coordinate already reveals
`x_0,i`.

```
if z_t,i != M:
  q(x_0,i | z_t,i, x_0) = delta(x_0,i = z_t,i)

if z_t,i == M:
  q(x_0,i | z_t,i, context) must be predicted
```

A common parameterization predicts the clean token first and then uses that
prediction to define reverse transitions. Under this parameterization, the
learned part of the bound becomes a masked clean-token cross entropy, possibly
with a time-dependent weight.

```
L(theta)
  = E_{x_0, t, z_t} [
      w(t) * sum_{i=1}^L 1[z_t,i = M] *
      -log p_theta(x_0,i | z_t, t)
    ] + constants
```

The repository uses the unweighted version with `w(t) = 1` and averages over
the selected masked tokens in the batch.

```
S = { (b, i) such that target_{b,i} != IGNORE_INDEX }

loss(theta)
  = (1 / |S|) * sum_{(b,i) in S}
      -log softmax(logits_{b,i})[target_{b,i}]
```

If `S` is empty, the mathematically useful value is zero with a valid gradient
path. That keeps a rare all-clean sampled batch from producing an undefined
mean.

Training samples time uniformly above a small lower bound.

```
u ~ Uniform(0, 1)
t = train_min_t + u * (1 - train_min_t)
```

The lower bound avoids spending training steps on almost-clean examples where
few or no tokens are selected.

### Where it lives in the code

`src/diffusion_lm_nano/train.py` implements the training loop. Lines 121 to
123 build the schedule. Lines 142 to 157 sample a clean batch. Lines 159 to 162
sample `t` uniformly and shift it into `[train_min_t, 1]`. Lines 163 to 169
call `corrupt_tokens` to produce `corrupted` inputs and sparse `targets`. Line
170 computes `masked_cross_entropy(model(corrupted, t), targets)`. Lines 171 to
173 run the optimizer step.

`src/diffusion_lm_nano/objective.py` implements the exact loss. Line 10 of
`src/diffusion_lm_nano/corruption.py` defines `IGNORE_INDEX = -100`.
`masked_cross_entropy` checks that logits and targets align on lines 11 to 14.
Line 15 selects exactly the targets that are not equal to the ignore index.
Lines 16 to 17 return the differentiable zero if no position is selected. Line
18 calls `F.cross_entropy(logits[selected], targets[selected])`, so the loss is
an ordinary categorical cross entropy over masked positions only. No
time-dependent ELBO weight appears in this function or in the training call.

`src/diffusion_lm_nano/evaluate.py` measures the same objective on held-out or
provided text. Lines 121 to 135 build an attention mask from non-pad tokens,
sample corruption at the requested timestep, and keep padding out of the
selected set. Lines 136 to 145 compute logits, masked accuracy, and a
token-count-weighted average of `masked_cross_entropy`. Lines 146 to 156 write
the measured fields, including masked tokens, masked accuracy, and masked cross
entropy.

## Derivation: the reverse sampler

### Motivation

Generation reverses the direction of the forward process. At the end of
forward diffusion, every active position is the mask token. Sampling therefore
starts from a full mask sequence, predicts all positions in parallel, commits
some predictions, and repeats.

This repository uses a fixed length and unconditional sampler. It does not
accept a prompt. It also prevents the model from emitting the special pad and
mask IDs as content.

### Derivation

Let `K` be the number of sampling steps and let `y_k` be the current token
matrix. The initial state is all masks.

```
y_0,i = M for i = 1, ..., L
```

At step `k`, the sampler uses a decreasing time value.

```
t_k = 1 - k / K
```

It calls the denoiser and turns logits into a token proposal for every
position.

```
logits_k = model(y_k, t_k)
logits_k[..., M] = -inf
logits_k[..., PAD] = -inf

if temperature = 0:
  proposal_i = argmax_v logits_{k,i,v}
  confidence_i = max_v softmax(logits_{k,i})_v

if temperature > 0:
  proposal_i ~ softmax(logits_{k,i} / temperature)
  confidence_i = softmax(logits_{k,i} / temperature)_{proposal_i}
```

The same proposal distribution provides normalized entropy over the non-special
token support `V - 2`.

```
entropy_i
  = -sum_v p_i(v) log p_i(v) / log(V - 2)

mean_entropy_k
  = mean_{i in R_k} entropy_i

mean_confidence_k
  = mean_{i in R_k} confidence_i

confidence_improvement_k
  = mean_confidence_k - mean_confidence_{k-1}
```

Only currently masked positions can be revealed. Let `R_k` be that set. The
configured schedule gives a target remaining mask fraction, and a termination
rule guarantees progress.

```
rho_k = schedule.probability(t_k)
target_remaining_k = ceil(rho_k * L)
schedule_count_k = |R_k| - target_remaining_k
termination_count_k = ceil(|R_k| / (K - k))

reveal_count_k
  = min(
      |R_k|,
      max(schedule_count_k, termination_count_k)
    )
```

The confidence strategy chooses the `reveal_count_k` positions with largest
proposal confidence. The fixed-order strategy chooses the first masked positions
from left to right. The fixed-fraction strategy chooses that many masked
positions uniformly at random. Revealed positions receive the proposed token,
and the rest stay masked.

The adaptive strategy normally uses the confidence ordering and the same quota.
After a configured minimum pass count, it instead commits every unresolved
proposal and stops when either condition holds.

```
mean_entropy_k < entropy_threshold

or

confidence_improvement_k < confidence_delta
for confidence_patience consecutive eligible passes
```

Entropy threshold zero and confidence patience zero disable their respective
conditions. Early finalization is monotone because it only replaces current
masks. If neither condition fires, the termination quota still reveals every
remaining mask on the declared final pass.

```
y_{k+1,i}
  = proposal_i if i is revealed at step k
  = y_{k,i}    otherwise
```

The final pass must reveal every remaining position because
`termination_count_k` equals `|R_k|` when one step remains. The sampler then
checks that no mask tokens remain.

### Where it lives in the code

`src/diffusion_lm_nano/sampler.py` implements the reverse process. `sample` and
`sample_from_tokens` retain the tensor-returning interface.
`sample_with_metadata` and `sample_from_tokens_with_metadata` additionally
return actual forward passes, the stopping reason, and a typed per-pass
trajectory. `_draw_predictions` preserves greedy temperature-zero behavior and
positive-temperature multinomial generator semantics. `normalized_entropy`
computes the audit statistic from the already computed proposal probabilities.
The reveal loop evaluates adaptive conditions before the ordinary quota, commits
all current unresolved predictions on a trigger, and otherwise applies the
configured fixed or confidence-ranked reveal rule.

`src/diffusion_lm_nano/train.py` uses the same sampler after training. Lines
218 to 232 call `sample` with the trained model, the configured number of
sampling steps, strategy, temperature, schedule, and a callback that records
the trajectory. Lines 238 to 249 write a measured sample payload when the
command actually produced the sample.

## Extension study for reveal budget and ordering

### Motivation

More reveal steps can help because the sampler leaves some positions masked and
recomputes logits after other positions become visible. The next denoising pass
therefore sees extra bidirectional context. This benefit can stop growing when
the remaining masked positions lack useful context or when earlier committed
tokens already dominate the context.

Confidence ordering can beat fixed order when the model assigns higher
probability to positions that are already supported by visible context. It then
commits reliable positions first and delays uncertain ones. Fixed order ignores
those scores, so it can commit a low-confidence token before the context that
would have made the prediction easier.

### Derivation

Let `C_k` be the visible committed set after step `k`. More steps allow a
strictly smaller reveal set on each pass.

```
C_{k+1} = C_k union A_k
A_k subset R_k
```

The denoiser at step `k + 1` conditions on `C_{k+1}` rather than `C_k`. If the
newly visible tokens reduce uncertainty for remaining masks, the next proposal
distribution can improve.

```
p_theta(x_i | y_{k+1}, t_{k+1})
  can be sharper than
p_theta(x_i | y_k, t_k)
```

The improvement has a limit in this implementation because a revealed token is
never revised. If `A_k` contains a wrong token, later steps condition on that
wrong token as visible context. More steps also stop helping when every
remaining masked position receives the same proposal with or without additional
visible tokens.

Confidence ordering uses the model as a local reliability estimator.

```
A_k = top confidence positions in R_k
```

Fixed order uses sequence position only.

```
A_k = first positions in R_k
```

The confidence rule can dominate fixed order when high confidence correlates
with correctness. It can also tie or lose when the confidence estimate is
uncalibrated or when the model is too undertrained for ordering to matter.

### Where it lives in the code

`src/diffusion_lm_nano/sampler.py` recomputes logits on every pass. Lines 17 to
53 make unconditional sampling a wrapper around `sample_from_tokens`. Lines 85
to 97 identify masked positions, set `t`, call the denoiser, block special
tokens, and convert logits into proposals and confidence values. Lines 111 to
114 compute how many currently masked positions to reveal, so larger budgets
usually reveal fewer positions per pass. Line 129 commits only selected
positions and leaves the rest masked for the next pass.

Ordering differs in the same loop. Lines 118 to 120 implement confidence
ordering with `topk` over current confidence values. Lines 121 to 122 implement
fixed order with the first masked positions. Lines 123 to 127 preserve the
random fixed-fraction baseline.

`src/diffusion_lm_nano/decoding_study.py` wraps that sampler without changing
the reverse process. Lines 90 to 105 parse ordering strategies. Lines 108 to
123 aggregate token accuracy over the positions selected by initial corruption.
Lines 133 to 212 compute the best point by budget, the Pareto frontier, and
equal-budget ordering comparisons. Lines 286 to 378 corrupt held-out copy
sequences, call `sample_from_tokens` for each requested budget and ordering,
count actual forward passes through the callback, and aggregate final
reconstruction accuracy.

The measured local run uses `artifacts/decoding-copy-corpus.jsonl`,
`configs/decoding_signal.toml`, `artifacts/decoding-signal-checkpoint.pt`, seed
`15`, and initial timestep `0.75`. Confidence decoding reaches `156/476`,
`164/476`, `173/476`, `160/476`, `154/476`, and `154/476` correct initially
masked tokens at budgets `1`, `2`, `3`, `4`, `6`, and `12`. Fixed order reaches
`156/476`, `151/476`, `152/476`, `156/476`, `150/476`, and `151/476`. The
Pareto frontier improves through three reveal steps and then stops improving.
Confidence ordering ties fixed order when one step reveals every selected mask
and beats fixed order at every larger budget in this development-scale run.

### Pending adaptive publication protocol

The adaptive question is whether confidence and entropy can identify examples
whose unresolved predictions are ready to finalize before a declared
maximum-pass budget without selecting thresholds on the final evaluation
window. `src/diffusion_lm_nano/adaptive_study.py` prepares each corruption once,
reuses it across all policies, and assigns the same decoding seed to each paired
policy run.

`configs/adaptive_study_publication.toml` preregisters a feasible grid for one
RTX A6000. The first disjoint window inside the training tail holdout calibrates
the adaptive configuration. Selection maximizes initially masked token
reconstruction accuracy, then exact sequence recovery, then minimizes realized
forward passes. A later non-overlapping window evaluates fixed-order,
fixed-fraction, confidence, and the selected adaptive policy. The artifact
stores paired per-example metrics and trajectories, actual forward calls,
stopping reasons, synchronized wall time, input hashes, precision, and hardware.
The wall-time field includes Python overhead and cannot support a speed claim by
itself. All real-corpus GPU result fields remain pending until that command
produces and hashes the publication artifact.

## Honesty boundary: what is measured and what remains pending

### Motivation

The repository separates an implementation claim from an empirical claim. The
code implements masked discrete diffusion and includes commands that can
produce artifacts. That does not by itself establish model quality or a fair
speed comparison against an autoregressive baseline.

### Derivation

A measured result must come from a command that actually writes bytes. A smoke
result measures pipeline execution under a tiny deterministic configuration.
It checks wiring, shape compatibility, artifact writing, and sampler
termination. It does not estimate final language-model quality.

```
measured artifact
  = produced by the stated command
  + stored as local bytes
  + recorded with provenance and a hash

smoke result
  = measured artifact from a tiny configuration
  + useful for pipeline verification
  + not a final quality or benchmark claim

pending result
  = an intended artifact or comparison that has not been produced
```

A fair comparison with autoregression needs controlled tokenizer, data,
parameter budget, prompt policy, generated length, batch size, precision,
hardware, warmup, and metrics. Diffusion masked-token cross entropy and
autoregressive next-token cross entropy answer different questions, so one
should not place them in one table as if they were the same measurement.

### Where it lives in the code

`article.md` states the boundary in prose. Lines 55 to 62 describe the
benchmark protocol and its limitations. Lines 64 to 71 explain why diffusion
masked-token loss and autoregressive next-token loss need separate
interpretation. Line 73 says that this release candidate does not include
comparative performance numbers. Lines 91 to 100 describe the smoke
configuration and the measured, simulated, and pending status categories.

`configs/smoke.toml` records the smoke path. Lines 1 to 10 define a tiny byte
model with `vocab_size = 258`, `mask_token_id = 257`, and `pad_token_id = 256`.
Lines 12 to 18 set a linear schedule, six sampling steps, confidence decoding,
and greedy temperature zero. Lines 20 to 27 set CPU execution, seed seven, batch
size two, and four optimizer steps.

`configs/publication.toml` records the intended larger path. Lines 1 to 10 set
the 9,159-token vocabulary with `pad_token_id = 9157` and `mask_token_id =
9158`. Lines 12 to 18 set cosine diffusion, 64 sampling steps, confidence
decoding, and temperature `0.8`. Lines 20 to 29 set the training seed, 10,000
steps, batch size 32, an external `data/train.bin`, and the contiguous tail
validation fraction.

`src/diffusion_lm_nano/artifacts.py` enforces status vocabulary for generated
artifacts. Lines 20 to 29 define `record_artifact` and accept only `measured`,
`simulated`, or `pending`. Lines 35 to 42 record path, kind, status, hash,
byte size, and provenance. Lines 43 to 53 update the manifest.

`src/diffusion_lm_nano/benchmark.py` keeps comparison claims explicit. Lines
121 to 127 record the autoregressive result as pending when no compatible
checkpoint is supplied. Lines 183 to 217 write measured diffusion timing fields
and the comparison protocol only after the benchmark command runs.

`artifacts/manifest.json` records the current state. Lines 28 to 39 mark the
smoke benchmark as measured. Lines 40 to 82 mark the smoke checkpoint as
measured. Lines 83 to 94 mark smoke evaluation as measured. Lines 95 to 137
mark the smoke iterative sample as measured. Lines 3 to 26 mark the publication
checkpoint and publication sample as pending. Lines 162 to 166 define
`measured`, `pending`, and `simulated`.

`SIGNOFF.md` matches that boundary. Lines 17 to 27 state that local smoke
training, evaluation, and timing completed, while publication training and the
same-hardware publication comparison remain pending. The same section states
that no comparative performance or quality claim is asserted.
