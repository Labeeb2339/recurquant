# Experiment 009: right-RHT CQER protocol

> **Status: screening and development protocol frozen before quality
> evaluation.**
>
> This protocol defines a cheap falsification screen on one already-open task
> and, only if that screen passes, a separately authenticated 32-task
> development run. Ranked MBPP window `[8, 16)` remains protected and must not
> be selected, retained, canonicalized, formatted, tokenized, passed to the
> model, or evaluated by either stage. Public-stream traversal is defined in
> [`EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md`](EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md).

Protocol frozen: 2026-07-26

## Question

Does a deterministic orthonormal transform along each Gated DeltaNet state
row's value axis reduce absmax INT4 outlier error enough to improve CQER-32 at
the same packed recurrent-state and selector byte counts?

The candidate method name is:

```text
right_rht_query_ema32_weighted_mse_target_fisher_quota
```

The short research name is **RHT-CQER-32**. This is an experimental codec
variant, not a novelty, speed, or breakthrough claim.

## Codec

For recurrent row `s` with value width `d = 128`, layer `l`, and value head
`h`, define a deterministic sign diagonal `D[l,h]` and the unnormalized
Walsh-Hadamard matrix `H` satisfying `H H^T = d I`:

```text
encode(s) = s D[l,h] H / sqrt(d)
decode(z) = z H D[l,h] / sqrt(d)
```

The FP32 encode/decode pair is orthonormal up to floating-point error. It
operates only within one existing `[head, key-row]` group, so it does not mix
CQER row identities or query coordinates.

The sign schedule is frozen as follows:

- seed: `2339`;
- domain separator: `recurquant.right-rht.signs.v1\0`;
- SHA-256 input fields: seed, model-layer index, head index, width, and block
  counter, each encoded as unsigned little-endian 64-bit values;
- digest bits are consumed byte-major and least-significant-bit first;
- bit one maps to `+1`, bit zero maps to `-1`;
- the concatenated INT8 sign schedule for model layers
  `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22`, 16 heads, and width 128
  has SHA-256
  `2d5137b5ebeb325f100b34190618783b9e47bd2ce9b27b6cdf3cdc94459dabc3`.

Signs and FP32 transform workspaces are regenerated transiently in the current
reference implementation. No sign tensor is resident in the cache. Runtime,
temporary workspace, allocator peak, and whole-model memory must be reported
separately before any systems claim.

## Quantization and selection

RHT-CQER-32 uses the existing physical Q4/Q8 group formats, FP16 scale per
row, nearest rounding, and one packed precision bit per row. Both CQER
endpoints are evaluated through the same codec:

```text
e4 = s - decode(Q4(encode(s)))
e8 = s - decode(Q8(encode(s)))
benefit = mean_value(e4^2 - e8^2)
score = EMA32(normalized_query^2) * benefit
```

The causal query-energy EMA, stable ranking, stage/consume handshake, and
target-directional-Fisher per-layer quotas are unchanged from Experiment 007.
Confirmation-2 is forbidden.

The quotas remain:

```text
0:355, 1:380, 2:269, 4:179, 5:185, 6:105,
8:80, 9:43, 10:84, 12:30, 13:62, 14:54,
16:45, 17:27, 18:7, 20:9, 21:7, 22:55
```

They sum to 1,976 Q8 rows. For both CQER-32 and RHT-CQER-32:

| Component | Bytes |
| --- | ---: |
| Q4/Q8 payloads | 2,485,760 |
| FP16 scales | 73,728 |
| precision masks | 4,608 |
| **packed recurrent state** | **2,564,096** |
| FP32 query-energy EMA | 147,456 |
| **resident bytes including selector** | **2,711,552** |

## Stage A: already-open falsification screen

Stage A uses only task `666`, already opened and authenticated in Experiment
008:

```text
ranked calibration index: 16
row SHA-256:
b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9
prompt tokens: 69
code tokens: 39
aligned scored tokens: 38
```

The evaluator must compare reference FP32 state, frozen CQER-32, and
RHT-CQER-32 in one teacher-forced run. It must record aligned excess NLL, mean
KL, CVaR95 KL, maximum KL, top-1 agreement, per-write state error, exact
storage, quotas, observation handshakes, finiteness, source hashes, repository
identity, and artifact hashes.

Stage A passes only if every condition holds:

1. the repository is clean and unchanged throughout the run;
2. the row identity and token counts above match before model loading;
3. all packed states occupy exactly `2,564,096` bytes and selector-aware
   totals equal `2,711,552` bytes;
4. every layer has its exact frozen quota, all observations are consumed once,
   and all logits and metrics are finite;
5. independent unit evidence gives right-RHT inverse relative L2 below
   `3e-7` and exact physical-pack versus transformed-QDQ reconstruction;
6. closed-loop aggregate per-write state SSE is at least 50% lower than
   CQER-32;
7. RHT-CQER-32 aligned excess NLL is at least 10% lower than CQER-32;
8. RHT-CQER-32 mean KL is lower; and
9. RHT-CQER-32 top-1 agreement is not lower.

Task `666` was selected before RHT quality was measured because it is the first
task in the already-authenticated Experiment 008 window and one of CORA's
recorded tail failures. Passing one exposed task authorizes only Stage B.
Failure stops this candidate without trying another sign seed, normalization,
transform axis, or threshold on the same task.

## Stage B: new 32-task development run

Only after Stage A passes from a committed authenticated artifact may the
ranked calibration window `[32, 64)` be resolved. The exact ordered task IDs,
row hashes, content manifest, prompt/code token counts, and tokenizer manifest
must be committed in an identity amendment before model weights are loaded or
any quality metric is observed.

Identity resolution must first rank an ID-only stream, then canonicalize and
tokenize only the 32 selected target rows. Non-target source records may be
inspected only for `task_id` and must be discarded immediately. Dataset
transport may deserialize complete records; this transport fact is not treated
as experiment-level content access.

The Stage B methods are frozen to:

1. static target-directional Fisher Q4/Q8;
2. adaptive per-write MSE under target-Fisher layer quotas;
3. CQER-32; and
4. RHT-CQER-32.

Primary metric: task-macro aligned excess next-token NLL relative to FP32
recurrent state. Also report mean KL, task-macro CVaR95 KL, maximum KL, top-1
agreement, full-code secondary metrics, task/token counts, and 10,000 paired
task bootstraps with seed 2339.

Stage B advances only if all integrity conditions from Stage A hold and:

1. RHT-CQER-32 lowers macro excess NLL by at least 20% versus CQER-32;
2. the paired 95% lower bound for CQER-minus-RHT excess NLL is above zero;
3. RHT-CQER-32 wins on at least 20 of 32 task-level excess-NLL comparisons;
4. macro mean KL is lower than CQER-32;
5. macro CVaR95 KL is no higher than CQER-32;
6. macro top-1 agreement is no more than `0.005` below CQER-32;
7. no task's RHT-minus-CQER excess-NLL disadvantage exceeds `0.25`; and
8. aggregate per-write state SSE is at least 50% lower than CQER-32.

If any condition fails, authenticate the negative result and stop
RHT-CQER-32. Do not change the transform seed, thresholds, task window, or
method name after results are visible.

## Claim and prior-art boundary

Randomized Hadamard and learned rotation methods are established in
[QuIP#](https://arxiv.org/abs/2402.04396),
[QuaRot](https://arxiv.org/abs/2404.00456),
[SpinQuant](https://arxiv.org/abs/2405.16406), and
[TurboQuant](https://arxiv.org/abs/2504.19874). MambaQuant applies
rotation-based outlier suppression to an adjacent state-space architecture.
CQER also has close query-relevance and quantization-difficulty precedent in
MixKVQ.

Therefore a positive result would support only this narrow engineering and
empirical contribution: a deterministic physical right-RHT codec composed
with causal row allocation for a pinned Gated DeltaNet recurrent cache under
an exact byte contract. It would not establish that rotations, Hadamard
quantization, query-aware precision, or recurrent-state quantization are new.

Before release-level or paper-level claims, the candidate still requires an
independent numeric verifier, another model size, natural-text and long-context
tasks, closest-method comparisons, optimized packed kernels, end-to-end
latency and peak-memory measurements, and independent reproduction. Ranked
window `[8, 16)` stays closed until those prerequisites are separately frozen
and passed.
