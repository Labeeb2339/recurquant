# Experiment 004: loss-sensitive row allocation

> **Status: working protocol draft - not frozen.**
>
> This document defines the next falsifiable RecurQuant experiment. It reports
> no positive result and supports no novelty, improvement, or breakthrough
> claim. Formulas, data manifests, thresholds, and implementation commits must
> be frozen before their corresponding evaluation stage begins.

Date opened: 2026-07-22

## Research question

At one exact resident recurrent-state byte budget, can a static row-level
INT4/INT8 policy selected by downstream loss sensitivity preserve the outputs
of Qwen3.5 Gated DeltaNet layers better than H1 read risk, row reconstruction
MSE, the v0.2 static policy, and prespecified random policies?

The primary candidate measures the signed first-order loss change along the
actual INT4-to-INT8 promotion direction on a cache that has already followed a
repeated-INT4-QDQ trajectory. A model-Fisher risk difference is the label-free
secondary candidate. Both are ranking approximations. Neither is treated as a
result until it predicts physical row interventions and then passes untouched
confirmation.

## Why the experiment changed

Experiment 003 rejected the longer-horizon HRR candidate in its same-data
diagnostic. H32 was worse than H1 even though the two policies overlapped
strongly. One untested hypothesis is a structural mismatch: HRR propagated one
error through a frozen full-precision trajectory, while the real cache stores
a newly quantized recurrent state after prefill and after every update. The
existing artifacts do not isolate this cause.

I will not retrofit that failure into evidence for the new method. Experiment
004 starts a new hypothesis and tests it against direct physical
interventions. If loss sensitivity does not predict those interventions, the
experiment stops before a large confirmation run.

## Existing exploratory artifacts

The local `loss-sensitivity-diagnostic-8task.json` selector artifact and
`loss-sensitivity-quality-diagnostic-8task.json` quality artifact were created
before the full Stage A gate below was completed. They are exploratory
engineering diagnostics at rung 0 only: they may expose implementation failures,
but they do not satisfy Stage A, freeze a ranked policy, open Stage B or held-out
evaluation, or support a positive result. Any protocol-gated ranking must be
regenerated only after Stage A passes from a clean committed implementation.

## Runtime and notation contract

For recurrent layer `l`, head `h`, physical state row `r`, and storage boundary
`t`, let `u^4_(t,r)` be the row immediately before storage on the trajectory
generated with every recurrent-state row stored as INT4 after every update.
The superscript records the trajectory, not the precision of this transient
pre-storage value.

Let `Q4` and `Q8` be the exact RecurQuant QDQ maps used by the packed
evaluator, including grouping, rounding, clipping, stored FP16 scale
conversion, padding, and dequantization.
Define

```text
s^4_(t,r)       = Q4(u^4_(t,r))
delta_(t,r)     = Q8(u^4_(t,r)) - Q4(u^4_(t,r))
e4_(t,r)        = Q4(u^4_(t,r)) - u^4_(t,r)
e8_(t,r)        = Q8(u^4_(t,r)) - u^4_(t,r)
```

`delta` is the actual local promotion direction. It must not be replaced by a
weight norm, activation magnitude, gate mean, or an error measured on a
full-precision cache. Calibration uses the same recurrent-state timing and
quantizer implementation as packed evaluation. The cache is quantized after
prefill and after every recurrent update.

The deployed policy is static: one precision bit per physical row, reused at
every storage boundary. It is not allowed to inspect confirmation targets or
choose precision per prompt.

## Primary candidate: signed Taylor promotion gain

For a teacher-forced calibration sequence, let `L_next` be the target negative
log-likelihood for the next token after a storage boundary. Evaluate its
gradient at the real repeated-INT4 trajectory:

```text
g_(t,r) = d L_next / d s^4_(t,r)
```

The first-order predicted loss reduction from promoting row `r` is

```text
T_r = E_task [ (1 / |T_task|) sum_t -<g_(t,r), delta_(t,r)> ]
```

A positive value predicts that the INT8 move lowers loss; a negative value
predicts harm. I will retain the sign. I will not square the directional
derivative, take its absolute value, or clamp negative coordinates merely to
make rankings look stable.

The expectation first averages scored storage boundaries within a task or
document and then gives tasks/documents equal weight. A multi-domain policy
also gives each frozen domain equal weight. This prevents long examples or one
domain from determining the policy by token count alone.

This is deliberately an immediate-loss score: one reverse pass scores every
row, but it does not pretend to differentiate a full future suffix through
hard repeated quantization. A truncated-suffix straight-through estimator is
a separate possible method and may not silently replace this candidate after
results are visible. The primary score still ignores higher-order curvature,
changes in later quantization bins, future effects beyond the next read, and
interactions among many simultaneous row promotions. The full physical oracle
and equal-byte evaluation are therefore mandatory rather than optional
validation.

## Label-free secondary: row-block model-Fisher risk difference

The secondary candidate uses model-sampled pseudo-labels instead of dataset
targets. At each calibration window, sample `M` pseudo-label paths from the
model distribution at the chosen local expansion point. For sample `m`, let
`h_m` be the gradient of its negative log-probability with respect to the
flattened time-by-value block for physical row `r`. Using the corresponding
flattened INT4 and INT8 error blocks, define

```text
F_r = (1 / (2M)) sum_m [ (h_m^T e4_r)^2 - (h_m^T e8_r)^2 ]
```

The two squared terms are evaluated separately before subtraction. The method
must not use `(h^T (e4 - e8))^2`, because that is the local Fisher sensitivity
of the move between endpoints, not the predicted reduction in risk relative
to the unquantized row. It must not use
`sum_i h_i^2 max(e4_i^2 - e8_i^2, 0)`, because coordinate-wise clipping changes
the diagonal-Fisher objective.

Pseudo-labels, RNG seeds, sample count `M`, expansion-point construction, and
Monte Carlo error bars will be frozen. This score is only "label-free" in the
sense that it does not consume benchmark targets; it still depends on the
model distribution and calibration inputs. The empirical Fisher is not a
generic Hessian substitute, so this candidate remains secondary unless local
KL checks and physical interventions validate it.

## Comparators and controls

All row-format selectors receive the same promotion count and store the same
precision mask.

1. `taylor_signed_next`: the primary score `T_r`.
2. `model_fisher`: the secondary score `F_r`.
3. `hrr_h1`: the Experiment 003 immediate-read marginal risk score, recomputed
   on the same calibration manifest.
4. `row_mse`: the task-macro mean of
   `||e4_(t,r)||_2^2 - ||e8_(t,r)||_2^2`, with negative values retained.
5. `v02_layer0_static`: layer 0 in INT8 and the remaining recurrent layers in
   INT4, using its real mask-free layout.
6. `random_rows`: at least 20 row policies generated from prespecified seeds.
   The full distribution is reported; the most favorable seed is not promoted
   to the named baseline after results are visible.
7. `uniform_int4`: a lower-storage control, not an equal-byte competitor.
8. `fp32_state`: the quality reference, not a compressed competitor.

The strongest equal-byte comparator is whichever of H1, row MSE, static v0.2,
and the prespecified random distribution has the lowest frozen-development
macro excess NLL. I will not compare only against the easiest baseline.

## Stage A: analytic and finite-difference sanity

Before ranking policies, the implementation must pass four checks on tiny
models and then on a stratified sample of real Qwen3.5 storage events.

1. **Trajectory parity.** Captured `u^4`, `Q4(u^4)`, scales, codes, and logits
   must match the packed repeated-INT4 path within frozen numerical tolerances.
2. **Directional derivative.** At one selected row and storage boundary,
   inject `alpha * delta` after INT4 QDQ and evaluate the same next-token loss.
   Compare `<g, delta>` with the central difference
   `(L(+epsilon) - L(-epsilon)) / (2 epsilon)` for
   `epsilon in {1/4, 1/8, 1/16, 1/32}`.
3. **Convergence.** For derivatives whose finite-difference magnitude exceeds
   the frozen near-zero floor, require at least 95% sign agreement, median
   relative error at most 10%, and a decreasing-error trend over the final
   three epsilon values. Near-zero cases use an absolute error bound chosen
   from FP32 repeat noise before the test is opened.
4. **Fisher/KL check.** On the same strata, compare the Monte Carlo model-Fisher
   quadratic with the measured local KL at shrinking error scales. Report
   correlation, calibration slope, and Monte Carlo confidence intervals. A
   Fisher score that does not approach the local KL as scale shrinks is an
   implementation or approximation failure, not evidence against Taylor
   sensitivity.

The strata cover every recurrent layer, multiple heads, early/middle/late
tokens, both domains, score signs, and quantization-error deciles. Any failed
check blocks the physical-oracle stage. Thresholds above are provisional until
the development freeze; changing them after opening test outputs creates a new
experiment version.

## Stage B: stratified physical row oracle

The oracle measures what each scalable score is supposed to predict. It uses a
held-out oracle partition that is not used to compute scores.

For each sampled physical row `r`, run two complete packed trajectories:

```text
baseline: all recurrent-state rows stored as INT4
oracle r: row r stored as INT8 at every boundary; all other rows stored as INT4
O_r      = L_baseline - L_oracle_r
```

This is an actual recurrent intervention, not a QDQ replay of a completed FP32
trace. It includes altered future states, later quantization, and the exact
row policy machinery. One-row trials all have the same incremental payload;
their purpose is causal ranking, not the final equal-byte claim.

Rows are sampled before oracle outputs are opened. The sampling frame is
balanced across recurrent layer, head, calibration-score decile, MSE decile,
and sign, with a prespecified random seed. At least 384 unique rows per
model-by-domain development cell are evaluated unless a blinded power analysis
freezes a larger count. All sampled rows are retained, including negative and
catastrophic interventions.

For every candidate and comparator, report:

- Spearman rho and Kendall tau-b against `O_r`, with row-stratified bootstrap
  intervals;
- oracle gain in each predicted score decile;
- precision and regret for the top 10% of sampled rows;
- results by layer, head, token-position stratum, and score sign; and
- the frequency with which an allegedly beneficial promotion has `O_r < 0`.

The primary candidate passes this gate only if its rank association has a
95% interval strictly above zero in every model-by-domain cell, its macro
top-decile oracle gain is at least 20% higher than the strongest of H1 and row
MSE, and no cell has negative top-decile mean gain. These are diagnostic gates,
not confirmation claims. If Taylor fails, I will publish the failure and stop;
I will not select a favorable layer subset after inspecting the oracle.

## Stage C: split-half policy stability

Calibration task/document IDs are divided into fixed halves A and B within
each domain. I will build one balanced two-domain policy from A and another
from B for each model, without using oracle or quality-evaluation targets.

Before the development quality run, require in every model cell:

- all-row Spearman correlation of at least 0.70 between A and B scores;
- Jaccard overlap of at least 0.50 between the promoted row sets; and
- no more than 10 percentage points of promotion share moving into or out of
  any recurrent layer.

The final development policy uses the complete calibration partition only
after these thresholds pass. Failure means the allocator is too dependent on
the calibration sample for a static public policy. Averaging unstable halves
after seeing the failure does not count as a pass.

## Exact physical byte budget

For the currently targeted batch-one Qwen3.5 recurrent-state layout, each
physical row contains 128 values. INT4 stores 64 payload bytes per row, INT8
stores 128, and one promotion therefore costs 64 payload bytes. There are
36,864 selectable rows in the 18-layer, 16-head layout.

| Component | Row-format policy | Static v0.2 |
| --- | ---: | ---: |
| All-INT4 payload | 2,359,296 | 2,359,296 |
| 36,864 FP16 scales | 73,728 | 73,728 |
| Precision mask | 4,608 | 0 |
| INT8 promotion payload | 126,464 | 131,072 |
| Promoted rows | 1,976 | 2,048 |
| **Resident recurrent-state total** | **2,564,096** | **2,564,096** |

Every row selector promotes exactly 1,976 rows. Static v0.2 keeps its genuine
metadata advantage and therefore promotes 72 more rows at the same total.
Uniform INT4 occupies 2,433,024 bytes; FP32 recurrent states occupy 18,874,368
bytes for this layout.

The evaluator will derive these counts from live tensor shapes and storage,
not accept the table as proof. Payload, stored scales, mask, indices, alignment,
padding, prefix tables, allocator-owned persistent tensors, transient bytes,
allocated bytes, and reserved bytes are recorded separately. Any new kernel
metadata is charged to the method that requires it. A model revision with a
different state layout gets a newly derived equal-byte budget before freeze.
Resident-state savings do not imply peak-memory or speed gains; those require
their own measured results.

## Data separation and freeze sequence

The confirmation matrix contains two base checkpoints and two domains:

| Checkpoint | Code | Natural text |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B-Base` | fixed public partition | fixed public partition |
| `Qwen/Qwen3.5-2B-Base` | fixed public partition | fixed public partition |

These two checkpoints test scale, not architectural transfer. The separate
architecture gate for any later breakthrough-level claim is
[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), whose erase and write
gates are channel-wise and decoupled. Its
[official implementation](https://github.com/NVlabs/GatedDeltaNet-2) exposes
the same `128 x 128` recurrent-state geometry but is not a drop-in Transformers
cache, and its source has different license terms. A third-party checkpoint
must be pinned, audited, and covered by a separate integration protocol before
it can count as evidence. Tiny random models or a port that merely runs do not
count as cross-architecture quality validation.

The current
[370M community checkpoint](https://huggingface.co/LLM-OS-Models/gdn2-370m-fineweb-edu-100b)
does not satisfy that gate: its pinned Hub revision contains raw `.pth` weights
but not the exact runnable model wrapper, tokenizer, config, or reference
logits, and the linked training-code repository is unavailable. The official
NVIDIA source uses noncommercial license terms and its released GPT wrapper
does not currently thread the GDN-2 cache. GDN-2 is therefore deferred from
Experiment 004 rather than treated as an easy extra result.

Exact dataset names, immutable revisions, example IDs, serialization,
tokenization, context windows, content hashes, and exclusion rules must be
written into the freeze record before any corresponding outputs are opened.
Previously inspected RecurQuant confirmation examples are ineligible for an
"untouched" claim.

The sequence is:

1. **Engineering sandbox:** unit tests, tiny-model checks, profiler work, and
   explicitly non-evidentiary debugging prompts.
2. **Calibration:** compute scores from fixed calibration partitions. Target
   labels are allowed only for `taylor_signed_next` and are never reused as held-out
   evidence.
3. **Oracle development:** open only the prespecified stratified one-row trials.
4. **Development freeze:** record formulas, normalization, data manifests,
   seeds, pseudo-label count, row sample, byte formulas, metrics, thresholds,
   implementation commit, dependencies, and environment before the full
   development quality matrix runs.
5. **Development run:** execute every comparator. Proceed only if all gates
   pass without changing the candidate.
6. **Confirmation freeze:** lock the selected formula, one policy per model,
   packed format, evaluator, statistical plan, and claim language. Hash the
   complete record.
7. **Untouched confirmation:** run the complete 2-by-2 matrix once. No partial
   previews, policy sweeps, prompt edits, seed replacement, or favorable-subset
   filtering.
8. **Independent replication:** release the frozen artifacts and require a
   reproduction outside the original implementation/evaluation run before any
   breakthrough wording is considered.

Any change to the score, row budget, model, dataset, seed set, threshold, or
primary metric after its freeze opens a new numbered experiment. Negative
results remain in the repository.

## Full-policy development and confirmation gates

Primary metric: task/document-macro excess NLL relative to the FP32-state
reference. The statistical unit is the task or document, not the token. Use
10,000 paired bootstrap resamples with frozen seed 2339.

Secondary metrics are mean token KL, worst-10% token KL, top-1 agreement,
worst-10% task/document excess-NLL CVaR, free-generation task metrics,
long-context retrieval when applicable, exact bytes, decode throughput, and
p50/p95 single-token latency.

At both development and untouched confirmation, `taylor_signed_next` must:

1. occupy exactly 2,564,096 resident recurrent-state bytes for the frozen
   layout and pass packed-versus-explicit-QDQ parity;
2. reduce macro excess NLL by at least 20% relative to the strongest equal-byte
   comparator;
3. have a paired 95% bootstrap interval strictly above zero for that reduction;
4. improve excess NLL in every model-by-domain cell;
5. remain non-inferior on frozen margins for CVaR, top-1 agreement, retrieval,
   and free generation; and
6. disclose throughput, latency, peak allocated memory, and peak reserved
   memory even when they regress.

A development pass permits the confirmation run, not a public performance
claim. A confirmation failure is the result. It is not repaired by redefining
the primary metric or removing the failing cell.

## Claim ladder

Each rung permits only the wording shown below.

| Rung | Required evidence | Permitted claim |
| --- | --- | --- |
| 0. Hypothesis | This working draft | "RecurQuant is testing signed downstream-loss sensitivity for recurrent-state row allocation." |
| 1. Implementation | Trajectory parity and finite-difference checks pass | "The implemented score matches local directional derivatives on the tested repeated-QDQ path." |
| 2. Oracle validity | The frozen stratified physical-row gate passes | "The score predicts held-out single-row promotion effects on the named development sample." |
| 3. Held-out result | Split-half, development, and untouched 2-by-2 confirmation gates pass | "At the stated byte budget, the frozen policy reduced excess NLL by X% over the strongest named comparator on the named checkpoints and domains." |
| 4. Research contribution | Rung 3, closest-method reproduction, released code/artifacts, and measured systems behavior | "RecurQuant provides a validated loss-sensitive mixed-precision recurrent-state allocation method within the evaluated scope." |
| 5. Breakthrough consideration | Independent replication, additional recurrent architecture and hardware, meaningful advantage over closest current methods, and no hidden quality or systems regression | Only then may "breakthrough" be evaluated as a community claim; it is never inferred from this project's own benchmark alone. |

RecurQuant is currently at rung 0 for Experiment 004. Passing one checkpoint,
one domain, a same-data diagnostic, or a correlation plot cannot skip rungs.

## Prior-work and novelty boundary

The broad ingredients are established. [SqueezeLLM](https://arxiv.org/abs/2306.07629)
uses sensitivity information in post-training LLM quantization;
[HAWQ-V2](https://arxiv.org/abs/1911.03852) and
[HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html) develop
Hessian-aware mixed precision and hardware-aware integer inference; and
[GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) studies
gradient-guided low-bit quantization. The limitations of treating empirical
Fisher as a general curvature proxy are analyzed directly in
[On the Relationship Between the Fisher Information and the Hessian](https://arxiv.org/abs/1905.12558).

The recurrent setting is also active prior art. [Gated DeltaNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)
defines the gated delta recurrence used by this model family, and the
[official Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
is the executable architecture reference. [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/)
quantizes recurrent state caches. [TQS-PTQ](https://arxiv.org/abs/2606.13300),
[RateQuant](https://arxiv.org/abs/2605.06675), and
[Block-GTQ](https://arxiv.org/abs/2606.24033) cover adjacent trajectory,
rate-allocation, and blockwise mixed-precision ideas.

[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) is directly relevant to
the generalization boundary: it replaces the tied scalar erase/write gate with
separate channel-wise gates while retaining a fixed recurrent matrix state.
It is a strong external architecture test, not evidence for the current Qwen
implementation. RecurQuant will not claim architectural generality until a
pinned pretrained GDN-2 checkpoint passes an independently frozen protocol.

Therefore, signed Taylor scoring, Fisher scoring, mixed bit widths, and
recurrent-cache quantization are not individually novel. The open question is
narrower: whether measuring the real INT4-to-INT8 promotion direction on a
repeatedly requantized Gated DeltaNet path yields a stable, exact-byte row
allocator that survives physical interventions and untouched evaluation. A
positive answer could support a scoped contribution. It would not establish a
"first" or a breakthrough without the higher claim rungs.

## Required public artifacts

If the protocol advances, I will release:

- the frozen protocol and hashes before confirmation;
- source code for trace capture, both scores, all comparators, mixed packing,
  and physical interventions;
- immutable model and dataset revisions plus exact example manifests;
- per-task metrics, all random seeds, bootstrap inputs, and negative rows;
- exact byte ledgers and memory/latency profiler outputs;
- environment and hardware records; and
- raw artifacts sufficient to recompute every table and graph.

The README may visualize a result only after its rung is earned. A graph from
the calibration set must be labeled calibration; an oracle plot must be
labeled diagnostic; and neither may be presented as held-out performance.
