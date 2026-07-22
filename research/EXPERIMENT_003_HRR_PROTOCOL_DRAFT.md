# Experiment 003 — quantizer-aware horizon row allocation

Status: historical protocol draft; superseded by
[`EXPERIMENT_003_HRR_DIAGNOSTIC.md`](EXPERIMENT_003_HRR_DIAGNOSTIC.md)

The same-data diagnostic rejected H32. This file preserves the plan that was
tested; its H32 primary, forward stages, gates, and claim ladder are not active
and are not evidence of an improvement.

Date opened: 2026-07-22

## Question

At one exact resident recurrent-state byte budget, can a static row-level
INT4/INT8 policy selected from full-precision calibration traces preserve
Qwen3.5 Gated DeltaNet predictions better than the strongest equal-byte
baselines?

The candidate is **horizon read risk (HRR)**. It propagates the measured error
from quantizing one recurrent-state row through the frozen Gated DeltaNet
trajectory and scores its effect on future reads. This is a local sensitivity
method, not a claim that the complete quantized model remains on the same
trajectory.

## Why this is a separate experiment

The earlier hand-built gate, forgetting, update-norm, and residual-magnitude
proxies were weak and prompt-dependent. Their best observed absolute Spearman
correlation was about 0.41 and the sign changed across prompts. Experiment 003
does not reinterpret that negative result. It replaces the proxy with an
empirical, quantizer-aware sensitivity calculation and requires new baselines
and new evidence.

The v0.2 policy remains frozen. Experiment 003 cannot rewrite its calibration,
development, or confirmation artifacts.

## Local score

Holding the full-precision teacher trajectory fixed, the recurrent-state error
obeys

```text
E_(t+1) = A_t E_t
A_t     = exp(g_t) (I - beta_t k_t k_t^T)
```

For a read at future position `tau`, define

```text
r_(tau,t) = q_tau^T A_tau ... A_t
```

The isolated squared contribution of source row `b` is

```text
r_(tau,t,b)^2 ||E_t[b, :]||_2^2
```

HRR averages this contribution over the next `H` available reads and then
task-macro averages the result. The implementation applies the transition to a
backward query vector, so it does not allocate one full error matrix per source
row.

The allocator must rank the **marginal value of promotion**, not raw INT4 risk:

```text
DeltaHRR_b = HRR_b(E_INT4) - HRR_b(E_INT8)
```

Negative values remain negative. They must not be clamped merely to make the
selector look stable.

## Trace contract

- Run the model in evaluation and inference mode with an ordinary FP32
  recurrent state and otherwise pinned BF16 model arithmetic.
- Prefill normally. During single-token teacher-forced decode, capture the
  state immediately before the token update and the `q`, `k`, `g`, and `beta`
  values consumed by that update.
- Normalize `q` and `k` exactly once and apply the kernel's `1/sqrt(key_dim)`
  query scale exactly once.
- Quantize the captured state independently to INT4 and INT8 using the actual
  grouped quantizer. Store only per-row error energies in the calibration
  trace, not the full error matrices.
- Drain traces at the task boundary. No task trace may leak into another task.
- A packed evaluation must quantize and store the recurrent state after prefill
  and after every recurrent update. Quantizing a completed FP32 trace once is
  not an implementation test.

## Physical format and budget

For the batch-one Qwen3.5 layout under test:

- 18 Gated DeltaNet layers;
- 16 heads per layer;
- 128 state rows per head;
- 128 values per row;
- 36,864 independently selectable rows in total;
- symmetric signed INT4 or INT8 payload;
- one FP16 scale per row;
- one packed precision bit per row.

At this layout:

| Component | Bytes |
| --- | ---: |
| All-INT4 payload | 2,359,296 |
| FP16 row scales | 73,728 |
| Precision mask | 4,608 |
| 1,976 INT4-to-INT8 promotions | 126,464 |
| **Total** | **2,564,096** |

The static v0.2 layer policy reaches the same total with 2,048 promoted rows
and no row mask. HRR therefore receives 72 fewer INT8 rows at equal bytes. This
metadata disadvantage is part of the comparison, not something to hide.

`2,564,096 bytes` means the sum of live recurrent-state payload, scale, and
precision-metadata tensors. It does not mean total cache, model, allocator-peak,
or whole-process memory. Any index, alignment, prefix table, or kernel metadata
introduced later must be counted.

## Candidate and equal-byte comparators

1. `v02_layer0_static`: model layer 0 at INT8 and the other recurrent layers at
   INT4. It uses 2,048 promotions and no row mask.
2. `random_rows`: 1,976 row promotions for at least 20 prespecified seeds.
3. `row_mse`: rank the measured reduction in row reconstruction error from
   INT4 to INT8.
4. `activation_weighted_row_mse`: weight the same marginal reconstruction
   error by a prespecified activation statistic. Do not call this AWQ.
5. `directional_fisher`: a calibration-only directional loss score using the
   actual INT4-to-INT8 error difference.
6. `hrr_h1`: immediate-read risk with the same physical row format.
7. `hrr_h32`: the primary finite-horizon candidate in this historical plan;
   the subsequent diagnostic rejected it.
8. `one_row_oracle`: optional expensive single-row perturbations on a small
   calibration subset only. It is a diagnostic ceiling, never a deployable
   baseline.

Every row-format method promotes exactly 1,976 rows and stores the identical
4,608-byte mask. Static v0.2 is allowed its real metadata-free layout but may
not exceed the shared total byte target.

## Models and domains

The intended confirmation matrix is:

| Model | Code | Natural text |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B-Base` | one fixed public split | one fixed public split |
| `Qwen/Qwen3.5-2B-Base` | the same protocol | the same protocol |

The official configurations currently expose the same 24-layer hybrid layout
with 18 recurrent layers and `16 x 128 x 128` recurrent states. Testing both
sizes is scale validation, not validation on another recurrent architecture.

MBPP training examples may be used for calibration and its validation examples
for development. The v0.2 MBPP confirmation split has already been inspected,
so it cannot serve as untouched v0.3 confirmation. Exact code and natural-text
confirmation datasets, revisions, example IDs, serialization, token windows,
and content hashes must be written here before their outputs are run.

## Metrics

Primary:

- task/document-macro excess NLL against the FP32 recurrent-state reference.

Secondary:

- mean and worst-10% token KL;
- top-1 token flip rate;
- worst-10% task/document excess-NLL CVaR;
- packed/QDQ numerical parity;
- payload, scale, mask, padding, transient, allocated, and reserved bytes;
- decode tokens/s and p50/p95 single-token latency;
- policy overlap and rank stability across calibration subsets.

The task or document window is the statistical unit. Use 10,000 paired
bootstrap resamples with seed 2339. Tokens from one sequence are not independent
samples.

## Historical sequence of work

The sequence below records the pre-diagnostic plan. Experiment 003 stopped at
the diagnostic pilot after H32 failed; the later stages were not opened.

### A. Engineering validity

1. Prove the analytic scorer against explicit isolated-row propagation.
2. Prove the row-energy form matches the full-error form.
3. Prove mixed packing matches explicit per-row QDQ.
4. Prove the exact byte decomposition above.
5. Prove lossless packed beam reorder and cache integration on tiny Qwen
   configurations.

### B. Diagnostic pilot

Run a small, explicitly non-confirmatory 0.8B trace to catch normalization,
state-timing, policy-mask, and performance defects. Pilot outputs may change the
implementation but cannot be described as generalization evidence.

### C. Freeze

Before development, record model revisions, dataset revisions and IDs, token
manifests, the complete method matrix, horizon, averaging, seeds, byte
accounting, numerical tolerances, evaluator commit, and environment.

### D. Development gate

The plan would have run every method on the complete frozen development
partitions and proceeded only if the primary candidate:

- uses exactly 2,564,096 resident recurrent-state bytes;
- agrees with explicit QDQ to absolute tolerance `1e-6`;
- reduces macro excess NLL by at least 20% versus every equal-byte comparator;
- has a paired 95% bootstrap interval above zero against the strongest
  comparator;
- does not regress worst-10% excess-NLL CVaR or top-1 flips beyond frozen
  margins; and
- has a positive effect in every model-by-domain development cell.

Any failed gate is a publishable negative result. Changing the selector,
horizon, dataset, or threshold creates a new experiment version.

### E. Untouched confirmation

Run the frozen matrix once on the untouched confirmation partitions. No partial
preview, row sweep, prompt change, or favorable-subset filtering is allowed.

## Claim ladder

No claim rung in this historical plan was earned.

Before development passes:

> RecurQuant is testing an exact-byte row-level INT4/INT8 Gated DeltaNet cache
> and a quantizer-aware finite-horizon allocator.

After all frozen confirmation gates pass:

> On the two named Qwen3.5 checkpoints and two named domains, HRR reduced excess
> NLL by X% over the strongest equal-byte baseline at 2,564,096 resident
> recurrent-state bytes.

Passing this local protocol would support a narrow research contribution, not
the word “breakthrough.” That stronger description additionally requires a
fused packed kernel, measured memory and speed improvements on at least two GPU
classes, longer-context downstream tasks, another Gated DeltaNet-family model,
and preferably independent replication.

## Closest-work boundary

The broad ingredients are not new:

- [WriteSAE](https://arxiv.org/abs/2605.12770) analyzes how Gated DeltaNet state
  interventions propagate to future reads and logits.
- [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) quantizes recurrent
  Mamba state caches to low precision.
- [TQS-PTQ](https://arxiv.org/abs/2606.13300) uses finite-horizon trajectory
  sensitivity for quantization.
- [RateQuant](https://arxiv.org/abs/2605.06675) performs quantizer-aware
  exact-budget mixed-precision cache allocation.
- [MixKVQ](https://arxiv.org/abs/2512.19206) combines quantization difficulty
  with future-query relevance for mixed-precision KV caches.

The research question is whether their still-uncovered intersection—actual
INT4-to-INT8 marginal error, analytic Gated DeltaNet transitions, exact-byte
row allocation, and physical recurrent-cache packing—produces a repeatable
benefit. A literature search cannot establish a “first” claim, so the project
will not make one.
