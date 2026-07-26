# Experiment 008: CORA-C2 protocol

> **Status: frozen before quality evaluation; development gate later failed.**
>
> CORA-C2 means **Causal Observability Row Allocation with Confirmation-2**.
> The state-transition formula, diagonal approximation, normalization,
> confirmation rule, quotas, storage accounting, development window,
> comparators, metrics, and advancement rule are fixed below. Ranked MBPP
> window `[8, 16)` remains protected and unopened. The authenticated outcome
> is recorded in [`EXPERIMENT_008_RESULT.md`](EXPERIMENT_008_RESULT.md); the
> protocol below remains the pre-result specification.

Protocol frozen: 2026-07-23

## Question

Can a causal approximation to recurrent-state observability allocate a fixed
INT8 row budget more faithfully than static Fisher sensitivity, per-write
reconstruction MSE, and CQER-32, while a parameter-free two-hit admission rule
prevents unstable precision-mask switching?

The frozen artifact method name is:

```text
causal_observability_confirm2_mse_target_fisher_quota
```

Experiment 007 showed a useful but insufficient CQER-32 point estimate. Its
macro excess NLL improved by 6.18% over adaptive MSE and 13.62% over static
target-Fisher, but the static reduction missed the frozen 20% gate, both paired
intervals crossed zero, and top-1 agreement missed the allowed margin. A
read-only postmortem also found that final-transition mask churn correlated
negatively with CQER NLL gain, although only the last transition was recorded.
Those observations motivate this experiment; they are not confirmation data.

## Verified transition orientation

For one Qwen3.5 Gated DeltaNet head, the recurrent state has shape
`[key_row, value]`. With normalized key `k`, scalar write gate `beta`, and
scalar decay `a = exp(g)`, the state update is:

```text
S_t = a_t (I - beta_t k_t k_t^T) S_(t-1) + beta_t k_t v_t^T
```

Define the state-error transition:

```text
T_t = a_t (I - beta_t k_t k_t^T)
```

This follows the pinned Transformers 5.14.1 torch recurrence and the Gated
DeltaNet update. It does not assume that gates alone predict end-to-end loss.

## Causal diagonal observability filter

The exact finite-horizon observability Gramian depends on future transitions
and queries, so it is unavailable to a causal inference-time selector.
CORA uses a normalized, locally stationary Lyapunov filter instead:

```text
P_bar_t = q*_t q*_t^T + T_t^T P_bar_(t-1) T_t
```

This is explicitly a predictor, not the true future Gramian. CORA stores only
the diagonal `p = diag(P_bar)`. Query and key vectors are L2-normalized with
`eps = 1e-6`, exactly matching the pinned kernel contract. If key-row width is
`d`, the read query is `q* = q_hat / sqrt(d)`.

For one head and token, let:

```text
u = k_hat^2
c = sum_j(p_j * u_j)
```

The frozen `O(d)` diagonal recurrence is:

```text
r_j = q*_j^2 + a^2 * (
    p_j * (1 - beta * u_j)^2
    + beta^2 * u_j * max(c - p_j * u_j, 0)
)
```

The `max` only suppresses negative roundoff in a mathematically non-negative
sum. After every token, trace-normalize across every head and key row in the
layer:

```text
p = r / sum(r)
```

Frozen numerical and causal details:

- initialize and reset `p` to `1 / (heads * key_rows)`;
- process multi-token chunks strictly in chronological order;
- compute normalized query, key, decay, beta, recurrence, and normalization in
  FP64 workspace, then commit persistent `p` in FP32;
- use the post-convolution, repeated-head query and key passed to the Gated
  DeltaNet kernel;
- stage the candidate only after the kernel succeeds and before its state write;
- commit `p` only after packing and every postcondition succeed;
- reject missing, duplicate, stale, wrong-shape, wrong-device, batch-greater-
  than-one, non-finite, or non-positive-trace observations; and
- reject packed-sequence `cu_seqlens` in this experiment.

## Row error and raw allocation

For incoming state row `S_i`, use the exact aligned physical quantizers from
the packed cache:

```text
benefit_i = ||S_i - Q4(S_i)||_2^2 - ||S_i - Q8(S_i)||_2^2
score_i = p_i * benefit_i
```

`Q4` and `Q8` use one group per `[head, key_row]`, FP16 scales, nearest
rounding, seed 2339, epsilon `1e-12`, and the existing physical packer. The raw
mask `R_t` is the stable top quota by `score`; exact ties preserve flattened
`[head, key_row]` order.

The per-layer quotas remain the authenticated target-directional-Fisher vector:

```text
0:355, 1:380, 2:269, 4:179, 5:185, 6:105,
8:80, 9:43, 10:84, 12:30, 13:62, 14:54,
16:45, 17:27, 18:7, 20:9, 21:7, 22:55
```

They sum to exactly 1,976 promoted INT8 rows. CORA does not search a new layer
allocation.

## Confirmation-2 switching

Let `R_t` be the raw CORA top-quota mask and `M_t` the committed precision
mask. The first write uses:

```text
M_1 = R_1
```

For every later state write:

```text
eligible_t = M_(t-1) OR (R_(t-1) AND R_t)
M_t = stable_top_quota(score_t restricted to eligible_t)
```

Incumbents remain eligible. A new row must appear in the raw top quota on two
consecutive writes before it can replace one. Because `M_(t-1)` already
contains the full quota, the eligible set always contains at least that many
rows and the committed mask always has the exact quota. Stable score order and
then canonical flattened order resolve ties.

Persistent `p`, the bit-packed previous raw mask, the packed state, counters,
and evidence form one transaction. Any failure rolls all of them back and
clears the staged observation.

## Storage contract

| Component | Bytes |
| --- | ---: |
| Packed INT4/INT8 recurrent state, scales, and committed masks | 2,564,096 |
| FP32 observability diagonal | 147,456 |
| Bit-packed previous raw masks | 4,608 |
| **Selector auxiliary tensors** | **152,064** |
| **Resident bytes including selector** | **2,716,160** |

Temporary query, key, gate, recurrence, endpoint, score, and sort workspaces
are not resident cache bytes. This experiment makes no equal-total-memory,
peak-memory, latency, throughput, or fused-kernel claim.

## Frozen development partition

The new development diagnostic uses the frozen MBPP calibration ranking but a
window not used for E005-E007 quality evaluation:

```text
dataset: google-research-datasets/mbpp, full
revision: 4bb6404fdc6cacfda99d4ac4205087b89d32030c
phase/source: calibration/train
selection namespace: rq-v0.2
ranked offset: 16
task count: 16
window: [16, 32)
```

The exact ordered task IDs, content manifest, and tokenizer manifest may be
resolved after this protocol commit, but they must be committed in a
pre-quality identity amendment before model loading. No quality metric from
`[16, 32)` may be observed before that amendment.

That identity amendment is now frozen in
[`EXPERIMENT_008_DEVELOPMENT_IDENTITY.md`](EXPERIMENT_008_DEVELOPMENT_IDENTITY.md):

```text
ordered task IDs:
666, 795, 944, 653, 857, 884, 878, 822,
687, 820, 920, 771, 869, 851, 728, 704

content-manifest SHA-256:
21dcc6e1955918a9f6baae3d02e7ba2781600405f91fe42bbe18eac8ca6dde5e

token-manifest SHA-256:
5a8e7b56528e3ccecc95ff83b2e59749d81dab27d0233fefafc510622a973f87
```

Ranked window `[8, 16)` remains protected. It cannot be loaded, tokenized, or
evaluated unless every development and independent numerical prerequisite in
this protocol passes from committed artifacts.

The pre-quality implementation meaning of dataset-loader access versus retained
evaluation rows is recorded in
[`EXPERIMENT_008_DATA_ACCESS_CLARIFICATION.md`](EXPERIMENT_008_DATA_ACCESS_CLARIFICATION.md).
No protected row is retained, canonicalized, formatted, tokenized, or evaluated.

## Frozen methods and ablations

All methods use the same model revision, quantizers, per-layer quotas, packed-
state byte contract, tasks, and teacher-forced transition alignment:

1. static target-directional Fisher;
2. adaptive target-Fisher per-write MSE;
3. CQER-32;
4. CORA without Confirmation-2;
5. CQER-32 with Confirmation-2; and
6. **CORA-C2**, the frozen primary.

Ablations must be reported but cannot replace the primary after results are
visible.

## Metrics and instrumentation

Primary metric: task-macro excess next-token NLL relative to FP32 recurrent
state on code transitions after a packed state has been stored. Exclude the
prompt-to-first-code-token transition exactly as in Experiments 005-007.

Also record:

- mean KL, token CVaR95 KL, maximum KL, and top-1 agreement;
- reference and candidate NLL, task/token counts, and all per-task rows;
- full-code secondary metrics;
- 10,000 paired task bootstrap contrasts with seed 2339;
- all-logit and all-metric finiteness;
- exact packed, selector, and combined resident bytes;
- per-layer raw and committed mask hashes and quotas;
- cumulative raw and committed XOR churn after the first write;
- normalized churn `total_xor / (2 * quota * transition_count)`;
- mask overlap, dwell/admission counts, cutoff score and normalized gap;
- transition observations staged/consumed and tokens processed; and
- repository commit, clean state, source hashes, artifact hashes, dataset
  manifest, tokenizer manifest, model revision, and command.

## Frozen development gate

CORA-C2 advances only if every condition holds:

1. exact quotas, packed bytes, selector bytes, stage/consume counts, stable
   commit/source hashes, authenticated manifests, finite logits, and finite
   metrics;
2. primary macro excess NLL is lower than static Fisher, adaptive MSE, and
   CQER-32;
3. NLL reduction is at least 20% versus static Fisher;
4. NLL reduction is at least 5% versus adaptive MSE;
5. NLL reduction is at least 5% versus CQER-32;
6. the paired 95% improvement lower bound is above zero versus both static
   Fisher and CQER-32;
7. primary top-1 agreement is at least the better of static Fisher and adaptive
   MSE minus `0.01`;
8. primary top-1 agreement is not lower than CQER-32;
9. primary CVaR95 is no more than `0.10` above the better static/adaptive value;
10. raw CORA improves NLL by at least 3% versus CQER-32, isolating the
    transition-aware signal;
11. C2 reduces normalized committed churn by at least 50% versus raw CORA;
12. C2 does not lower top-1 agreement versus raw CORA; and
13. C2 worsens raw-CORA NLL by no more than 1%.

If any condition fails, authenticate the result, keep `[8, 16)` closed, and
stop Experiment 008. Thresholds and ablations cannot be changed and relabelled
after observing `[16, 32)`.

## Independent numerical prerequisite

Only after the development gate passes, build a verifier that does not import
the production recurrence, ranking, quantization, or packing helpers. It must:

- replay the diagonal recurrence in CPU FP64;
- compare every step with explicit `T^T diag(p) T` on adversarial and captured
  fixtures;
- verify chronological chunk/step equivalence;
- replay aligned Q4/Q8 endpoints, scores, stable ranks, raw masks, C2 masks,
  quota counts, and physical bytes;
- cover zero/constant rows, half-step rounding boundaries, extreme finite
  values, ties, reset, transfer, and exception rollback; and
- require exact mask/byte hashes and derived numerical error bounds rather than
  an aggregate pass percentage.

Any ambiguity or mismatch stops the experiment before holdout.

## Prior-art and claim boundary

The mathematical ingredients are established, and novelty is unresolved:

- [Gated DeltaNet](https://arxiv.org/abs/2412.06464) defines the host recurrent
  update;
- [WriteSAE](https://arxiv.org/abs/2605.12770) derives the same state transition
  for later-logit cache interventions, while CORA applies a diagonal online
  filter to physical row-precision allocation;
- [MixKVQ](https://arxiv.org/abs/2512.19206) combines query magnitude and
  quantization difficulty for KV-cache channels and is the nearest query-aware
  mixed-precision mechanism;
- [TQS-PTQ](https://arxiv.org/abs/2606.13300) uses trajectory sensitivity for
  offline time-series weight allocation; and
- Q-Mamba, Quamba2, OuroMamba, and RateQuant already cover adjacent state-cache,
  SSM, dynamic-channel, and rate-distortion quantization ideas.

CORA-C2 differs by combining the actual Gated DeltaNet error transition, an
online diagonal observability approximation, physical Q4/Q8 recurrent-state
row error, and exact-quota two-hit admission. That distinction is a research
hypothesis, not proof of novelty.

Experiment 008 cannot support “breakthrough,” “first,” “novel,” or
“state-of-the-art” language without passing the frozen development and
numerical gates, one-time holdout, broader models/tasks, closest-method
comparisons, fused-system measurements, and independent reproduction.
