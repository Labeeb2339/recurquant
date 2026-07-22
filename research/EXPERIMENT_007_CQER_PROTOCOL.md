# Experiment 007: CQER-32 protocol

> **Status: development protocol frozen before the first quality run.**
>
> CQER-32 means **Causal Query-weighted Error Reduction with a 32-token
> half-life**. Its primary formula, normalization, EMA decay, quotas, task
> partition, metrics, controls, and advancement rule are fixed below. The
> ranked MBPP window `[8, 16)` remains unopened. This protocol makes no
> improvement, novelty, speed, or breakthrough claim.

Date opened: 2026-07-23

## Why Experiment 006 stopped

Experiment 006 combined a static loss-sensitivity rank with an instantaneous
reconstruction-error rank. On the already inspected selector partition, its
frozen equal-rank primary had macro excess NLL `0.514873`, compared with
`0.493302` for plain adaptive MSE and `0.535781` for the strongest static
method. Its paired intervals against both endpoints crossed zero, and its
3.90% point improvement over the strongest static method was below the frozen
20% threshold. Better intermediate-weight ablations were visible only after
the result and cannot replace the primary.

That result suggests a missing variable rather than a missing fusion weight.
Reconstruction MSE estimates how much a state row is damaged by low-bit
storage. It does not estimate how strongly the model will read that row.

## Research question

At the same frozen per-layer INT8 promotion quotas, does multiplying each row's
causal INT4-to-INT8 reconstruction benefit by a causal estimate of normalized
query energy reduce recurrent-state quantization error at model outputs?

The candidate is motivated by the Gated DeltaNet read. Ignoring cross-row
error covariance, a row's contribution to expected squared read error scales
with both:

1. the row's state reconstruction error; and
2. the squared query coordinate used to read that row.

CQER-32 tests this factorization directly. The ignored covariance and the use
of past query energy to predict future energy are explicit approximations.

## Frozen primary

The artifact method name is:

```text
query_ema32_weighted_mse_target_fisher_quota
```

The short research name is `CQER-32`.

### Query normalization

For query tensor `q` after Qwen3.5's convolution, head reshape, and value-head
repetition, detach and convert it to FP32, then normalize every token and head
with the pinned Gated DeltaNet formula:

```text
q_hat = q / sqrt(sum(q^2, key_row) + 1e-6)
energy = q_hat^2
```

The candidate supports only ordinary batch-one prefill, cached multi-token
continuation, and single-token decode. Packed sequences with `cu_seqlens` fail
closed in this version.

### Causal query-energy EMA

The half-life is frozen at 32 query tokens:

```text
rho = 2^(-1 / 32)
```

For each model layer, value head, and key row:

```text
EMA_t = rho * EMA_(t-1) + (1 - rho) * energy_t
```

Before the first token and after cache reset:

```text
EMA_0 = 1 / key_row_count
```

For an `n`-token prefill or continuation chunk, the implementation must be
numerically equivalent to folding tokens in chronological order:

```text
EMA_new = rho^n * EMA_old
          + (1 - rho) * sum_j rho^(n-1-j) * energy_j
```

No future query may contribute. The current call's queries are observed only
after its Gated DeltaNet kernel succeeds. They select the recurrent state that
is stored after that call and can affect only later model outputs; they cannot
change the output already produced by the same kernel call.

### Per-write reconstruction benefit

For incoming recurrent state `S` and each `[head, key_row]`, compute aligned
INT4 and INT8 endpoints with the same group geometry, scale type, nearest
rounding, and seed:

```text
benefit = mean_v((Q4(S) - S)^2) - mean_v((Q8(S) - S)^2)
```

The primary score is:

```text
CQER_score = EMA_new * benefit
```

Within every layer, select the fixed quota with the largest score. Exact score
ties preserve canonical flattened `[head, key_row]` order. Scores, sorting,
and selection run without autograd.

### Frozen layer quotas

CQER-32 does not search a new layer allocation. It inherits the authenticated
target-directional-Fisher per-layer quota vector from the inspected selector
partition. Across the 18 recurrent layers the quotas sum to exactly 1,976
INT8 rows. Every other row is INT4.

The quota vector is fixed; only row identities may change at a state write.
This isolates the within-layer read-importance hypothesis.

## Stage-and-consume causality contract

The observer and cache use one fail-closed record per layer write:

1. the observer identifies the exact cache object passed to the Qwen3.5 layer;
2. the Gated DeltaNet kernel receives the post-convolution query;
3. only after that kernel succeeds, the observer stages the query observation;
4. the immediately following recurrent-state write consumes it once;
5. packing success commits the new EMA and consumes the record; and
6. any exception clears an unconsumed record.

Missing, duplicate, stale, wrong-layer, wrong-update, wrong-shape,
wrong-device, batch-greater-than-one, or non-finite observations raise. A
reference, static, adaptive-MSE, or other unregistered cache passes through
without receiving observations. A model path that bypasses the wrapped kernels
cannot silently fall back to plain MSE.

The observer restores every wrapped model callable on context exit, including
exceptional exits. Multiple registered CQER caches remain independent and are
dispatched by object identity, not by class alone.

## Storage contract

The packed recurrent-state representation remains identical in byte budget to
the other exact-quota mixed methods:

| Component | Bytes |
| --- | ---: |
| INT4/INT8 payloads | 2,485,760 |
| FP16 group scales | 73,728 |
| precision masks | 4,608 |
| **packed recurrent-state resident bytes** | **2,564,096** |

CQER-32 also stores one persistent FP32 `[heads, key_rows]` EMA per recurrent
layer. For the pinned 18-layer, 16-head, 128-row geometry:

```text
18 * 16 * 128 * 4 = 147,456 bytes
```

Therefore the selector-aware total is:

```text
2,564,096 + 147,456 = 2,711,552 bytes
```

Artifacts and documentation must report all three fields separately:

```text
packed_recurrent_state_bytes
selector_auxiliary_bytes
resident_bytes_including_selector
```

The temporary query, endpoint, score, and sort workspaces are not resident
cache bytes and must not be hidden if peak-memory measurements are later made.
CQER-32 is not equal-total-memory to a method that has no selector state.

## Frozen development diagnostic

The first run may use only the already inspected MBPP selector-task prefix:

```text
phase: calibration
ranked offset: 0
task count: 8
task IDs: 945, 794, 657, 702, 651, 720, 903, 918
```

This data already influenced the target-Fisher quotas and prior hypotheses. A
result on it can reject an implementation or candidate but cannot establish
generalization.

The primary metric remains task-macro excess next-token NLL relative to FP32
recurrent state on aligned code transitions after a quantized state has been
stored. The prompt-to-first-code-token prediction is excluded. Also report
mean KL, worst-token KL CVaR95, top-1 agreement, task/token counts, all per-task
rows, and 10,000 paired bootstrap contrasts with seed 2339.

Required controls are:

```text
adaptive_mse_target_directional_fisher_quota
target_directional_fisher_difference_int4
adaptive_mse_hrr_h1_quota
hrr_h1
v02_layer0_static
uniform_int4
```

The authenticated Experiment 006 rank-fusion methods may be reported for
context but are not advancement comparators.

### Development advancement rule

CQER-32 advances to construction of its independent numeric/packing gate only
if all conditions hold on the inspected eight-task diagnostic:

1. every task and layer has the exact fixed promotion count;
2. packed recurrent-state bytes equal 2,564,096 and selector bytes equal
   147,456 after every completed task;
3. macro excess NLL is lower than both plain adaptive target-Fisher-quota MSE
   and static target-directional Fisher;
4. relative excess-NLL reduction versus plain adaptive MSE is at least 5%;
5. relative excess-NLL reduction versus the strongest static method is at
   least 20%;
6. top-1 agreement is no more than `0.01` below the better of those two
   comparators;
7. CVaR95 KL is no more than `0.10` above the lower of those two comparators;
8. every value and model output is finite;
9. every staged observation is consumed exactly once; and
10. code, model, dataset, selector, source, and artifact hashes verify from one
    clean stable commit.

This is a development filter, not a heldout quality pass. A paired confidence
interval is reported but is not used to tune `rho`, the formula, or quotas.
Failure stops this frozen candidate. Changing the half-life, adding a static
term, or changing the quota is a new experiment.

## Candidate-aligned prerequisite before holdout

If the development rule passes, a separately committed verifier must replay
the selector independently in CPU FP64 from authenticated source tensors. It
must cover prefill, cached chunk, and recurrent decode and verify:

- exact normalized query energies;
- exact chronological EMA recurrence and the chunk/step equivalence bound;
- aligned Q4 and Q8 row errors;
- CQER scores;
- stable ranks and masks;
- promotion counts and per-layer quotas;
- packed payload, scale, and mask bytes; and
- selector auxiliary bytes and reset/offload/prefetch behavior.

Every checked primitive must meet a predeclared forward-error bound, and every
rank, mask, quota, logical-array hash, and physical-byte hash must match
exactly. There is no aggregate pass percentage. One mismatch fails closed.

The verifier, raw sidecar schema, sampled coordinates, arithmetic bounds, and
pass conjunction must be frozen in a clean commit before the `[8, 16)` window
is tokenized or loaded.

## Holdout and claim boundary

The ranked MBPP calibration window `[8, 16)` remains reserved. CQER-32 cannot
open it unless the development rule and independent candidate-aligned
prerequisite both pass from committed, authenticated artifacts.

Even a pass on that eight-task window would establish only a scoped signal for
the pinned model, cache geometry, and teacher-forced code metric. A defensible
research contribution would still require a larger preregistered development
and test split, natural text, retrieval, long-context and free-generation
tasks, multiple checkpoints and recurrent architectures, closest-method
comparisons, systems kernels and benchmarks, and independent reproduction.

Query-aware mixed precision, EMA predictors, reconstruction-error selection,
and recurrent-state quantization all have prior art. The narrow candidate under
test is their exact causal combination for physical Gated DeltaNet recurrent
state rows under fixed per-layer byte quotas. Novelty is unresolved and cannot
be inferred from a quality result.

## Decision rule

- **Development rule fails:** authenticate the result, keep `[8, 16)` closed,
  and stop CQER-32.
- **Numeric prerequisite fails:** authenticate the failure, keep `[8, 16)`
  closed, and stop CQER-32.
- **Both pass:** freeze the one-time heldout evaluator before any protected
  data access.
- **Heldout result fails:** publish the complete negative result; do not tune
  on the protected tasks.
- **Heldout result passes:** advance to broader preregistered evaluation. Do
  not use breakthrough or state-of-the-art language.
