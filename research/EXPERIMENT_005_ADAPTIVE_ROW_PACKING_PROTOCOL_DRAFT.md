# Experiment 005: quota-calibrated adaptive row packing

> **Status: pre-holdout protocol freeze; implementation complete; holdout not
> opened.**
>
> This experiment is a new hypothesis after Experiment 004's signed-Taylor
> policy failed its same-calibration diagnostic. It makes no improvement,
> novelty, speed, or breakthrough claim. The selector artifacts, implementation
> commit, and source hashes must be frozen before the ranked calibration
> holdout is opened.

Date opened: 2026-07-23

## Research question

At the same exact resident recurrent-state byte budget, does separating the
allocation problem into:

1. an offline, target-directional Fisher estimate of **how many** INT8 rows
   each recurrent layer receives; and
2. a runtime reconstruction-error decision about **which** rows receive those
   slots at each storage boundary

preserve Qwen3.5 Gated DeltaNet outputs better than static row identities,
H1 read-risk allocation, row MSE, the v0.2 layer-0 policy, and random rows?

The method is called **quota-calibrated adaptive row packing** in this protocol.
That is a descriptive name, not a novelty claim.

## Why this is a separate experiment

The first Experiment 004 pilot ranked rows by the signed local Taylor benefit

```text
-gradient dot (Q8(raw) - Q4(raw)).
```

On the eight inspected calibration tasks, that complete 1,976-row policy was
worse than H1 and the properly formed squared endpoint-risk diagnostics. The
sign formula itself remains subject to an independent real-storage-boundary
finite-difference check, but a correct one-row derivative would not rescue a
poor simultaneous static policy. Experiment 005 therefore does not relabel the
failed Taylor policy or silently make another Experiment 004 score primary.

The new hypothesis addresses the observed prompt dependence directly: layer
budgets remain frozen, while row identities may respond to the actual state
being stored.

## Frozen candidate

### Offline layer quotas

Calibration follows the repeated-INT4 quantize/dequantize trajectory. For row
`r` at a scored storage boundary, with target-NLL gradient `g` and aligned
endpoint errors

```text
e4 = Q4(raw) - raw
e8 = Q8(raw) - raw,
```

the row score is

```text
D_r = task_macro_mean[(g dot e4)^2 - (g dot e8)^2].
```

Negative values are retained. Globally ranking `D_r` under the exact physical
budget produces a static plan. Experiment 005 discards that plan's row
identities after using only its per-layer promotion counts `q_l`.

This target-gradient score is a supervised calibration statistic. It is not
the label-free model Fisher and must not be described as one.

### Runtime row choice

For every raw recurrent-state write at layer `l`, compute each physical row's
aligned reconstruction-error reduction:

```text
A_(t,l,r) = mean((Q4(raw_(t,l,r)) - raw_(t,l,r))^2)
            - mean((Q8(raw_(t,l,r)) - raw_(t,l,r))^2).
```

Promote exactly the top `q_l` rows to INT8 for that write. Ties are resolved by
stable flattened `[head, key-row]` order. The selection is batch-one only in
this experiment. It cannot inspect the next token, target label, logits, or a
future query at inference time.

The primary method name in artifacts is:

```text
adaptive_mse_target_directional_fisher_quota
```

## Required ablations and controls

Every adaptive and selector-plan static row-format policy has the same 1,976
promotions, mask format, and exact resident bytes. The v0.2 control reaches the
same resident-byte total with 2,048 promotions because it has no precision
mask. Uniform INT4 is a lower-byte control. These differences are part of the
frozen comparison and may not be hidden by calling every policy equal-format.

1. `adaptive_mse_target_directional_fisher_quota` — frozen primary.
2. `adaptive_mse_hrr_h1_quota` — identical runtime selector with H1-derived
   per-layer quotas; isolates whether the offline quota signal matters.
3. `target_directional_fisher_difference_int4` — static row identities from
   the same Fisher score; isolates whether per-update adaptation matters.
4. `hrr_h1` — strongest pre-existing static diagnostic comparator.
5. `hrr_h32` — retained negative Experiment 003 hypothesis.
6. `row_mse` — static task-macro reconstruction-error allocation.
7. `v02_layer0_static` — layer 0 INT8, remaining recurrent layers INT4, with
   its genuine mask-free metadata advantage.
8. `random_rows_s1101` — prespecified random static policy.
9. `uniform_int4` — lower-byte control, not an equal-byte competitor.

The signed-Taylor and other Experiment 004 diagnostic policies remain in the
artifact for transparency but are not eligible to replace the frozen primary
after holdout results are visible.

## Exact storage contract

For Qwen3.5-0.8B-Base at batch one:

| Component | Adaptive/static row format | Static v0.2 |
| --- | ---: | ---: |
| All-INT4 payload | 2,359,296 B | 2,359,296 B |
| FP16 scales | 73,728 B | 73,728 B |
| Precision mask | 4,608 B | 0 B |
| INT8 promotion payload | 126,464 B | 131,072 B |
| Promoted rows | 1,976 | 2,048 |
| **Resident total** | **2,564,096 B** | **2,564,096 B** |

Adaptive row identities change without changing the number or size of packed
INT4 payloads, INT8 payloads, FP16 scales, or mask bytes. Exact storage is
checked from live packed tensors after every evaluated task. This is not a
claim about model weights, full KV memory, temporary workspace, CUDA allocator
peaks, throughput, or latency.

## Frozen local derivative validator

Before the heldout-calibration window is opened, the independent storage-boundary
diagnostic must run in FP32 and return `passed: true`. It uses one pinned
calibration task and target transition, four rows chosen only from fixed geometry,
physical Q4 and Q8 row endpoints, and the epsilon grid
`{1/4, 1/8, 1/16, 1/32}`. At the Q4 endpoint it checks the local target-NLL
directional derivative

```text
dL/dalpha at alpha=0 = g dot (Q8(raw) - Q4(raw))
```

against central differences. Equivalently, its local predicted benefit is the
negative of that derivative. The validator checks storage timing, endpoint
construction, derivative sign, and local finite-difference consistency. It does
**not** validate the algebra or implementation of the squared quota score:

```text
(g dot e4)^2 - (g dot e8)^2
```

The squared-score implementation is covered by selector unit tests; its arrays
and derived plans are authenticated by artifact hashes. Whether the resulting
layer quotas improve quality is the separate question tested by the frozen
quality diagnostic.

The derivative artifact passes only if every condition below holds:

1. the model and recurrent-state calculation use FP32;
2. the maximum repeated-`alpha=0` loss difference over all rows and epsilons is
   at most `1e-7`;
3. a row is informative when the absolute autograd derivative is strictly
   greater than `1e-8`, and at least three of the four rows are informative;
4. central-difference sign agreement over all informative row/epsilon checks is
   at least `0.95`;
5. the median absolute relative derivative error over those informative checks
   is at most `0.10`;
6. at least `0.75` of informative rows converge, where convergence means the
   absolute error at epsilon `1/32` is no greater than at epsilon `1/8`; and
7. for every near-zero row, the absolute central-versus-autograd derivative
   error is at most `2e-7` at every epsilon.

Failure is recorded and keeps the holdout closed. These thresholds, row strata,
epsilon grid, transition, and FP32 requirement may not be changed after seeing
the diagnostic output and called the same validation gate.

## Freeze and data separation

The first ranked eight rows of the pinned MBPP calibration population are the
selector partition. They were already inspected while developing Experiments
003 and 004 and can provide no held-out claim.

The first Experiment 005 generalization check is fixed before opening it:

```text
phase: calibration
ranked offset: 8
task count: 8
window: [8, 16)
```

Experiment 005 heldout mode is fail-closed: it requires both frozen selector
artifacts, `--calibration-offset 8`, and `--limit 8`. Any other offset or task
count is not this protocol and must be refused rather than emitted under the
Experiment 005 heldout artifact kind.

Before tokenization or model loading, the evaluator must authenticate each
selector artifact against the first eight rows of the pinned ranked MBPP
calibration population. Ordered task IDs, canonical row-content hashes,
dataset/config/revision, source split, selection namespace, and formatter
version must match the pinned selector prefix. It must then prove that window
`[8, 16)` is disjoint from the union of all authenticated selector-artifact task
IDs. Self-consistency between two selector artifacts is not sufficient.

The evaluator records the heldout window's canonical row-content manifest,
formatter version, token counts, model and dataset revisions, implementation
commit, command, packages, and hardware. It records an explicit map of relevant
implementation source paths and hashes both before and after the run, requires
those maps to be identical, and requires the same clean Git commit at both
boundaries. A dirty or changing implementation cannot produce a passing
heldout artifact. This remains a **heldout-calibration diagnostic**, not
development or confirmation evidence.

The holdout may be opened only after:

- the selector scripts and adaptive evaluator are committed;
- new selector artifacts are generated from that committed implementation;
- their canonical hashes and quantizer contracts verify;
- both selector manifests authenticate against the pinned first-eight-row
  prefix;
- the real storage-boundary derivative diagnostic satisfies every numeric local
  gate above and records `passed: true`;
- the evaluator starts and ends on the same clean implementation commit with
  unchanged source hashes; and
- this candidate, method list, offset, task count, primary metric, and gates
  are committed.

## Metrics and gates

Primary metric: task-macro excess next-token NLL relative to an FP32 recurrent
state, scored only on code transitions that occur after a quantized recurrent
state has been stored. The prompt-to-first-code-token prediction is excluded
because the cache policy cannot affect it. Use 10,000 paired bootstrap
resamples with seed 2339.

The equal-byte static comparator set is frozen as `hrr_h1`, `hrr_h32`,
`row_mse`, `random_rows_s1101`, `v02_layer0_static`,
`signed_taylor_next_int4`, `target_directional_fisher_difference_int4`,
`target_diagonal_fisher_difference_int4`, and
`delta_direction_magnitude_int4`. No member may be excluded after results are
visible. The strongest comparator is selected deterministically as the member
with the lowest macro excess NLL on this frozen window. Uniform INT4 is excluded
because it uses fewer resident bytes; `adaptive_mse_hrr_h1_quota` is adaptive
and is tested separately. The primary passes this small heldout-calibration gate
only if all conditions hold:

1. resident recurrent-state bytes equal 2,564,096 exactly on every task;
2. macro excess NLL is lower than every equal-byte static comparator;
3. the paired 95% interval versus the strongest equal-byte static comparator
   is strictly above zero when expressed as comparator minus primary;
4. relative excess-NLL reduction versus that comparator is at least 20%;
5. the paired 95% interval versus `adaptive_mse_hrr_h1_quota` is also strictly
   above zero when expressed as H1-adaptive minus primary, so the supervised
   quota signal contributes beyond adaptation alone;
6. macro top-1 agreement is no more than `0.01` below the strongest equal-byte
   static comparator;
7. macro worst-token KL CVaR95 is no more than `0.10` above that comparator;
8. on every task, the primary's excess NLL is at most `1.0` worse than that
   comparator's excess NLL;
9. no metric or model output is non-finite; and
10. all code, artifact, manifest, and source hashes verify, the repository is
    clean, and the start/end implementation records match.

The evaluator must record each condition, its threshold, observed value, and
pass/fail result, plus an overall conjunction. There is no discretionary
interpretation of “materially worse” or “catastrophic” after the results are
visible. If the strongest comparator has non-positive excess NLL so that the
relative-reduction condition is not meaningful, condition 4 fails closed.

Eight tasks are too few for a research claim. Passing advances the method to a
larger calibration split, a separately frozen development domain, free
generation, long-context retrieval, multiple rounding seeds, and a second
checkpoint. Failing is recorded and stops this candidate without changing its
formula after the fact.

## Systems boundary

The current Python prototype quantizes both candidate endpoints to score rows
and then physically packs the chosen endpoint. It is expected to add overhead.
No speed claim is permitted. A fused Triton/CUDA kernel is a later engineering
gate and must benchmark p50/p95 decode latency, throughput, peak allocated and
reserved memory, and kernel workspace against static packing.

Stochastic rounding is also a separate experiment. It may not be introduced
after viewing this nearest-rounding holdout and called the same candidate.

## Prior work and novelty boundary

The broad ingredients are not new:

- [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) quantizes Mamba
  state caches and introduces state/channel decoupled scaling plus reconstruction.
- [OuroMamba](https://arxiv.org/abs/2503.10959) uses per-time-step dynamic
  outlier-channel selection for mixed-precision Vision Mamba inference.
- [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
  reports recurrent cache-error accumulation and uses stochastic rounding for
  its FP16 Mamba state cache.
- [RateQuant](https://arxiv.org/abs/2605.06675) calibrates mixed-precision KV
  cache allocation with rate-distortion models.
- [TQS-PTQ](https://arxiv.org/abs/2606.13300) studies trajectory-based
  sensitivity for quantized dynamical systems.

Therefore dynamic mixed precision, state-cache quantization, calibration,
reconstruction error, Fisher/Taylor sensitivity, and layer quotas cannot be
claimed individually. The narrow open contribution is whether the particular
combination of supervised recurrent-layer quotas, exact-byte physical
INT4/INT8 matrix-row packing, and causal per-write row selection works on a
Gated DeltaNet language-model cache. Even a positive result on this protocol
would support only a scoped experimental contribution.

## Claim ladder

1. **Implementation:** exact packed bytes and deterministic adaptive masks pass
   unit and real-model integration tests.
2. **Local validity:** the independent local target-directional derivative
   checks pass; this does not validate the squared quota formula or policy
   quality.
3. **Heldout-calibration signal:** the frozen eight-task gate above passes.
4. **Development result:** larger code and natural-text cells, second model,
   free generation, long context, and systems measurements pass a new freeze.
5. **Research contribution:** closest-method reproductions and released raw
   artifacts support a stable advantage.
6. **Breakthrough consideration:** independent replication, another recurrent
   architecture and hardware, and a meaningful advantage without hidden
   systems or quality regression.

Experiment 005 is currently below rung 3. The word "breakthrough" is not
permitted in the README, release title, social post, or result figure.
