# Experiment 010: StateLease-H5 protocol

> **Status: protocol frozen before new development-identity resolution or
> quality evaluation.**
>
> This document fixes the candidate, storage accounting, comparators,
> evaluation sequence, advancement gates, and claim boundary. It contains no
> Experiment 010 quality result. A separate identity artifact must be
> committed before any Stage-B model forward pass.

Protocol frozen: 2026-07-30

## Research question

Can a physically packed RHT-CQER recurrent-state checkpoint plus a bounded
five-token Gated DeltaNet update buffer reduce reference-aligned recurrent
trajectory error and next-token loss at no more than six resident bits per
state element, when compared with fixed checkpoint periods and equal-resident-
byte no-replay codecs?

The frozen implementation identity is:

```text
statelease_cut4_cut5_right_rht_query_ema32_weighted_mse_fisher_quota
```

The short name is **StateLease-H5**.

This is a development hypothesis, not a novelty, speed, deployment,
state-of-the-art, or breakthrough claim.

## Why this experiment follows Experiment 009

Experiment 009 established a positive but scoped development result for
RHT-CQER-32 on one pinned model and one 32-task MBPP window. Relative to
CQER-32, RHT-CQER-32 reduced task-macro aligned excess NLL by `52.73%` and
local codec reconstruction SSE by `57.85%` at the same packed-state and
selector byte counts. See
[`EXPERIMENT_009_STAGE_B_RESULT.md`](EXPERIMENT_009_STAGE_B_RESULT.md).

That result does not show that a state already damaged by earlier
quantization is repaired. Experiment 009's local SSE compares each codec
output with that method's own pre-pack source state. It does not compare the
method's recurrent trajectory with the matched FP32 recurrent-state
trajectory.

Experiments 007 and 008 also constrain the next hypothesis:

- CQER-32 did not pass its frozen development gate by itself.
- CORA and CORA-C2 did not beat CQER-32.
- confirmation-gated mask persistence greatly reduced switching but worsened
  NLL.

StateLease therefore does not introduce another hand-crafted gate, learned
threshold, or promotion debounce. It tests a direct representation choice:
when a five-token buffer is full, which of two legal checkpoint boundaries
produces less current-state handoff distortion?

## Prior-art boundary

Periodic quantized state checkpointing with cached-input replay is established
prior art. Nemotron 3 Ultra reports a fixed checkpoint period of eight in
emulated Mamba-cache quantization experiments. ReplaySSM publicly describes
and implements Gated DeltaNet replay using a checkpoint and buffered
`(u, k, g)` records, including Qwen3.5.

StateLease-H5 cannot claim:

- the first checkpoint/replay recurrent cache;
- the first Gated DeltaNet input or update buffer;
- the first reduction in recurrent-state write frequency;
- the first randomized-Hadamard or rotation quantizer;
- the first online or adaptive cache-precision controller;
- a certified or bounded end-to-end error guarantee; or
- repair of historical recurrent-state error.

The narrow research question is whether a threshold-free, local c4-versus-c5
handoff choice helps a **physically packed RHT mixed-precision Gated DeltaNet
checkpoint under a strict resident-memory cap**. A positive experiment would
still require a renewed literature search before any novelty statement.

## Frozen model and state geometry

The Stage-A screen and Stage-B development run use:

```text
model: Qwen/Qwen3.5-0.8B-Base
revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
batch size: 1
recurrent layers: 18
value heads per recurrent layer: 16
key rows per head: 128
value width: 128
state dtype before packing: FP32
```

The persistent recurrent state therefore contains:

```text
18 * 16 * 128 * 128 = 4,718,592 elements
4,718,592 * 4 = 18,874,368 FP32 bytes
```

The identity artifact must independently re-read the pinned model config and
fail closed if any geometry differs. Chunked decode, beam search, speculative
decode, packed `cu_seqlens`, and batch sizes above one are outside the Stage-B
quality claim.

## Frozen Gated DeltaNet replay record

For the repository's `[key_row, value_width]` state orientation, one successful
Gated DeltaNet token transition is represented as:

```text
S_t = exp(g_t) * S_(t-1) + k_t outer u_t
```

Here `k_t` is the normalized key actually consumed by the successful state
kernel and `u_t` is the post-correction value-axis update produced for that
transition. The replay buffer stores:

- normalized `k_t` as BF16, shape `[16, 128]`;
- post-correction `u_t` as BF16, shape `[16, 128]`; and
- log-decay `g_t` as FP32, shape `[16]`.

It does not store `q_t`, raw `v_t`, or `beta_t`. Caching raw `v_t` would require
recomputing each correction sequentially because `u_t` depends on the incoming
state. Caching the successful post-correction `u_t` fixes that ambiguity.

The buffer is a five-slot contiguous per-layer array, not a ring. After a c4
handoff, the one retained tail record is copied to slot zero. After a c5
handoff, the count becomes zero. One INT32 count per recurrent layer is
sufficient; no hidden ring-head pointer is permitted.

## Frozen checkpoint codec

Every checkpoint uses the complete Experiment 009 RHT-CQER-32 codec without
retuning:

- deterministic right-side randomized Hadamard transform;
- sign seed `2339` and the Experiment 009 sign derivation;
- symmetric signed Q4/Q8 row formats and FP16 scales;
- nearest rounding;
- the frozen target-Fisher layer quotas;
- 1,976 Q8 rows in total;
- 32-token causal normalized-query-energy EMA;
- exact physical INT4-to-INT8 reconstruction benefit;
- the same row-index tie ordering; and
- the same packed payload, scale, and one-bit precision-mask contract.

RHT signs are generated from the frozen schedule and are not persistent state.
The query EMA is updated by the current token before a full-buffer boundary is
selected. Both c4 and c5 candidates use that same decision-time EMA. This is
causal: the choice is made after token five and changes only the
representation used by later tokens. It does not recompute an earlier output.

The persistent CQER EMA itself is unchanged from Experiment 009: same FP32
storage, 32-token decay, causal update ordering, initialization, and row
semantics. For handoff risk, each layer uses the per-head sum-normalized view
of the same current candidate EMA for both risk calls. There is no
candidate-specific reweighting.

## Frozen prefill and decode boundary

Stage B measures autoregressive-style, one-token-at-a-time decode after a
shared prefill:

1. The matched FP32 path computes the complete prefill state.
2. Every compressed candidate packs that same point in its own frozen format.
3. StateLease starts with the packed prefill checkpoint and an empty buffer.
4. The controller operates only on subsequent one-token teacher-forced decode.

No candidate may gain a different prefill, prompt truncation, token alignment,
or first scored token. A future streaming-prefill implementation is a separate
experiment.

## Frozen H5 c4-versus-c5 controller

There is no early handoff, risk ratio, learned gate, or tunable threshold. A
decision occurs only when a layer's buffer reaches exactly five chronological
records.

For recurrent layer `l`, let `S5_l` be the raw successful state-kernel output
after its fifth token, before a new checkpoint is committed. Let `w_l,h,r` be
that layer's decision-time causal query-energy EMA, normalized to sum to one
over key rows `r` for each head `h`.

At a full buffer:

1. Quantize the fifth successful transition record to its frozen BF16/FP32
   buffer dtypes, place it in slot four, and set that layer's valid count to
   five before either candidate is scored.
2. Take `S4_l`, the exact resident state supplied as the fifth successful
   kernel call's initial state. It is the current checkpoint replayed through
   records 1 through 4.
3. Pack `S4_l` with the frozen RHT-CQER codec, unpack it, and replay the
   BF16/FP32 form of record 5. This gives `S5_hat_c4_l`.
4. Pack and unpack `S5_l` directly. This gives `S5_hat_c5_l`.
5. Compute both causal query-weighted local risks in FP32:

```text
D_c,l = mean over heads h of
        sum over key rows r of
        w_l,h,r * mean over value width v of
        (S5_hat_c_l,h,r,v - S5_l,h,r,v)^2
```

6. For that layer, select c4 only when `D_c4,l < D_c5,l`. An exact FP32
   scalar tie selects c5.
7. For c4, commit the packed `S4_l`, discard records 1 through 4, move record 5
   to slot zero, and set the layer count to one.
8. For c5, commit the packed `S5_l`, clear all five records, and set the layer
   count to zero.

Each recurrent layer makes this choice independently. Its checkpoint advances
are therefore always four or five tokens, which is why the exact storage
contract includes one INT32 valid count per layer. The choice is causal and
threshold-free. A layer-macro summary of handoff risk may be reported as a
diagnostic, but it is never controller input and does not synchronize layer
boundaries.

`D_c,l` is local self-consistency distortion relative to StateLease's own
transient layer state. It is not an FP32 trajectory metric, an error
certificate, or a proof about logits. Thresholded checkpoint helpers,
unweighted handoff risk, synchronized global handoffs, and other lease lengths
are report-only engineering primitives and cannot replace this frozen
primary.

## Exact resident-byte contract

The packed checkpoint retains Experiment 009's exact storage:

| Checkpoint component | Bytes |
| --- | ---: |
| Q4/Q8 payloads | 2,485,760 |
| FP16 scales | 73,728 |
| one-bit precision masks | 4,608 |
| **packed checkpoint** | **2,564,096** |

The complete StateLease allocation is:

| Persistent component | Derivation | Bytes |
| --- | --- | ---: |
| packed RHT-CQER checkpoint | fixed above | 2,564,096 |
| FP32 query-energy EMA | `36,864 * 4` | 147,456 |
| BF16 normalized-key buffer | `18 * 5 * 16 * 128 * 2` | 368,640 |
| BF16 update buffer | `18 * 5 * 16 * 128 * 2` | 368,640 |
| FP32 log-decay buffer | `18 * 5 * 16 * 4` | 5,760 |
| INT32 per-layer counts | `18 * 4` | 72 |
| **total allocated resident bytes** |  | **3,454,664** |

This is:

```text
3,454,664 * 8 / 4,718,592 = 5.857110 bits per state element
18,874,368 / 3,454,664 = 5.463445 times smaller than the FP32 state
```

Every allocated slot counts even when empty. Tensor storage, not the number of
live records, determines the primary resident-byte result. Actual live bytes,
CUDA allocator reservation, temporary workspaces, host mirrors, and
end-to-end peak memory must also be reported separately.

The correctness-first Python path may transiently materialize FP32 states.
Those workspaces are not persistent-cache bytes, but they must be visible in
peak-memory reporting and prohibit a deployment-memory claim.

Any additional persistent tensor, alignment padding, ring pointer, saved raw
activation, FP32 state mirror, or duplicated device buffer changes the byte
count and fails the primary storage identity. Python object metadata that
does not own tensor storage is listed separately.

## Mandatory comparators

All quality comparators use the same model, prompts, token alignment, FP32
reference, RHT signs, query-EMA update ordering, rounding rule, and evaluator.

### Equal-allocation fixed replay

Each fixed replay comparator reserves the same five-slot buffer and therefore
the same `3,454,664` resident bytes, even when its live occupancy is lower:

1. `fixed_cut4_in5`: at a full H5 buffer, always checkpoint `S4` using the
   decision-time EMA and retain record 5.
2. `fixed_cc5`: checkpoint the current state every five records and clear.
3. `fixed_cc4`: checkpoint every four records and clear.
4. `fixed_cc2`: checkpoint every two records and clear.
5. `fixed_cc1`: checkpoint after every decode token.

`fixed_cut4_in5` is mandatory because it has the same delayed decision timing
as StateLease. Comparing only with eager CC4 would confound boundary choice
with the timing of quantization.

The strongest fixed replay comparator is the one with the lowest frozen
Stage-B workload-macro excess NLL. It is selected mechanically after all
fixed results exist; it is not hand-picked per task.

### Equal-total-byte no-replay codecs

These comparators spend the complete `3,454,664`-byte StateLease allocation on
the current state instead of replay:

1. **Expanded RHT Q4/Q8.** All-row Q4 payloads, FP16 scales, one-bit precision
   codes, and the FP32 query EMA cost `2,585,088` bytes. Exactly 13,587 rows
   are promoted to Q8, adding `869,568` bytes, with 8 explicit reserved padding
   bytes. Promotions use global descending causal query-weighted physical
   Q4-to-Q8 reconstruction benefit; exact ties use earlier flattened rows.
2. **RHT Q4/Q6/Q8.** All-row Q4 payloads, FP16 scales, two-bit precision codes,
   and the FP32 query EMA cost `2,589,696` bytes. The exact allocator receives
   27,030 32-byte marginal steps, adding `864,960` bytes, with 8 explicit
   reserved padding bytes. The complete-state dynamic program from
   [`MULTIBIT_REFERENCE_DESIGN.md`](MULTIBIT_REFERENCE_DESIGN.md) determines
   Q4/Q6/Q8 choices from causal weighted physical distortions.
3. **RHT residual-Q4.** All rows receive a first RHT-Q4 code. A one-bit lease
   mask, FP16 scales, and the FP32 query EMA retain the same `2,585,088`-byte
   base. Exactly 13,175 rows receive an additional Q4 code plus FP16 residual
   scale (`66` bytes each), adding `869,550` bytes, with 26 explicit reserved
   padding bytes. The residual is computed in transformed coordinates against
   the first Q4 reconstruction. Selection uses exact causal
   reconstruction-benefit ranking.

The Q4/Q6/Q8 and residual codecs are mandatory Stage-B comparators, not
substitutes for StateLease. Their implementations must pass Stage 0 before a
development identity is committed.

For later gates:

- the **strongest fixed replay comparator** is the one of the five
  equal-allocation fixed schedules with the lowest Stage-B workload-macro
  excess NLL; and
- the **strongest equal-byte comparator** is the one with the lowest
  Stage-B workload-macro excess NLL across all five fixed schedules and all
  three equal-total-byte no-replay codecs.

Both identities are selected mechanically after all mandatory results exist.
Neither may change per task.

### Anchors and off-budget prior-art references

- The original Experiment 009 RHT-CQER path at `2,711,552` selector-aware
  bytes remains a historical lower-memory anchor.
- The matched FP32 recurrent-state path is the quality reference.
- A Nemotron-style fixed CC8 checkpoint/replay reference is required but
  explicitly off budget. With the same checkpoint, EMA, dtypes, and one INT32
  count per layer, H8 allocates `3,900,488` bytes, or `6.612969` bits per state
  element. It cannot satisfy an equal-byte advancement gate.
- A full-FP32 ReplaySSM-style checkpoint is an upper-fidelity systems
  reference, not a compressed comparator.

No result may present the off-budget references as if they had equal storage.

## Stage 0: algebra, implementation, and independent verification

Stage 0 uses only synthetic tensors and already captured traces. It must finish
before any new development identity is resolved.

An independent dense CPU verifier must not import production replay, packing,
row-ranking, boundary-selection, or metric helpers. It must verify:

1. chronological replay of FP32 `(u, k, g)` records against the pinned Qwen3.5
   Gated DeltaNet transition;
2. the exact normalized key and post-correction `u` captured from a successful
   kernel call;
3. BF16 buffer round-trip and replay ordering;
4. c4 and c5 candidate construction against a direct dense implementation;
5. per-layer query-EMA-weighted handoff-risk calculation, identical normalized
   weights for both candidates, and exact c5 tie handling;
6. c4 compaction to slot zero and c5 clearing;
7. cache reset, exception rollback, interruption resume, dtype, device,
   contiguity, shape, and finite-value invariants;
8. exact component byte counts and absence of a persistent FP32 state mirror;
9. the physical Q4/Q8, Q4/Q6/Q8, and residual-Q4 comparators; and
10. the protected-window guard.

For unquantized replay, the dense and production final states must satisfy:

```text
relative L2 <= 3e-6
maximum absolute error <= 1e-5
```

The identity artifact may tighten but not loosen these limits. A CC1
compatibility test must reproduce the current RHT-CQER trajectory, aligned
metrics, row plans, and hashes within the already frozen evaluator tolerances.

Any failed Stage-0 condition stops Experiment 010 before new data access.

## Stage A: cheap falsification on already-open data

Stage A may use only this previously opened MBPP task:

```text
666
```

It may not resolve or inspect a new MBPP identity. The ordered task and token
manifests must be copied from authenticated existing evidence, not regenerated
from a wider stream.

Stage A compares StateLease-H5 with FP32, original RHT-CQER, all five fixed
replay schedules, and all three equal-total-byte no-replay codecs. It is a
falsification screen only.

All of the following are required to continue:

1. every Stage-0 and artifact-integrity check passes;
2. exact allocated bytes equal `3,454,664`;
3. every StateLease interval is four or five and every tie selects c5;
4. aligned excess NLL is at least 10% lower than equal-allocation `fixed_cc1`;
5. aligned excess NLL is no more than 5% worse than the strongest fixed replay
   comparator;
6. reference-aligned trajectory NMSE AUC is lower than `fixed_cc1`;
7. task-macro top-1 agreement trails the best fixed comparator by no more than
   `0.01`; and
8. all primary values are finite.

If a comparator's excess NLL is zero or negative, a relative-improvement gate
against it fails closed. Stage A cannot support a public improvement claim.

## Stage-B development identity: rules before identities

No Stage-B row, prompt, token, or generated RULER instance has been selected by
this protocol. The next artifact must be:

```text
research/EXPERIMENT_010_DEVELOPMENT_IDENTITY.md
```

with a compact machine-readable companion under `evidence/`.

The identity step may load dataset text and the pinned tokenizer, but it may
not load model weights, compute logits, reconstruct recurrent states, or
produce any candidate quality metric. The artifact and evaluator source hashes
must be committed before the first Stage-B model forward pass.

The development panel is frozen to three workload families:

1. **PG19 validation natural text** from the official DeepMind source: 32
   books, one 4,608-token contiguous segment per book, with 4,096 prefill
   tokens and 512 scored teacher-forced continuation tokens.
2. **NVIDIA RULER**: one verified official configuration from each of
   single-key retrieval, multi-key retrieval, variable tracking, and
   common-word aggregation; context lengths 2,048, 4,096, and 8,192; seeds
   `2339`, `2340`, `2341`, and `2342`, for 48 generated instances. Answer
   tokens are scored teacher-forced; deterministic generated exact match is
   secondary.
3. **EvalPlus HumanEval+**: 32 SHA-ranked tasks. The prompt is prefill and the
   canonical solution is the scored teacher-forced continuation, capped at
   512 scored tokens. Sandboxed generated pass@1 is secondary.

The identity artifact must first pin exact immutable source revisions and
licenses, verify the official field and task names, and then instantiate these
rules. If an official source cannot satisfy a frozen rule, identity creation
stops; it may not silently substitute a different dataset, split, task family,
or row count.

### Deterministic identity construction

The common ranking domain is:

```text
recurquant.experiment010.development.v1\0
```

Ranks use ascending SHA-256 digest of:

```text
domain || UTF8(source_name) || 0x00 || UTF8(canonical_candidate_id)
```

with canonical candidate ID as the exact final tie-breaker.

For PG19:

1. tokenize each validation book with the pinned model tokenizer;
2. retain books containing at least 4,608 tokens;
3. let `m = floor(token_count / 4,608)`;
4. select one non-overlapping within-book window with a separate SHA-256
   `...pg19.offset.v1\0` digest modulo `m`; and
5. SHA-rank eligible books and retain the first 32.

This permits at most one segment per book.

For HumanEval+, rank canonical task IDs and retain the first 32. For RULER,
the frozen family, length, and seed Cartesian product defines all 48
identities; there is no quality-based filtering.

The identity artifact must record:

- source repository or dataset revision and source file hashes;
- license identifiers;
- tokenizer name, revision, files, and hashes;
- ordered canonical IDs and ranking digests;
- exact generator arguments and seeds;
- canonical content hashes;
- prompt and target token IDs or their collision-resistant hashes;
- prefill and scored-token counts;
- selected, formatted, tokenized, and evaluated identity sets;
- model config and revision;
- Python, PyTorch, Transformers, CUDA, GPU, and operating-system identity;
- repository commit and hashes for every imported source file; and
- an explicit statement that no quality output existed before identity commit.

No task may be removed because its candidate loss is large, generation fails,
or a baseline behaves unexpectedly.

## Protected MBPP window

Ranked MBPP window `[8, 16)` remains sealed throughout Experiment 010. It must
not be:

- selected or retained;
- canonicalized or content-hashed;
- formatted;
- tokenized;
- passed to a model;
- evaluated; or
- included in an artifact.

For a public streaming source, RecurQuant may inspect only `task_id` on
non-target rows and must immediately discard the mapping, as defined in
[`EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md`](EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md).
Experiment 010 does not authorize opening `[8, 16)` after a pass. Any future
one-time supplementary legacy audit requires a separately frozen amendment
after the method, implementation, identity, and independent verifier are
fixed. Eight protected tasks would not be sufficient as the main confirmation
set.

## Stage-B metrics

### Primary quality metric

For task `i` and method `m`:

```text
excess_nll_i,m = aligned_nll_i,m - aligned_nll_i,fp32
```

Each workload first takes an unweighted mean over its tasks. The Stage-B
primary is the unweighted mean of the three workload-macro excess NLL values,
so RULER's larger task count cannot dominate the result.

All methods must score identical target token IDs. No unmatched prefix,
candidate-specific truncation, or token omission is allowed.

### Reference-aligned trajectory metric

At each scored recurrent write, compare the candidate state with the matched
FP32 recurrent-state trajectory:

```text
trajectory_nmse_l,t =
    ||S_candidate_l,t - S_fp32_l,t||_F^2
    / (||S_fp32_l,t||_F^2 + 1e-12)
```

Accumulate in FP64. Average within each task over layers and scored writes,
then take task-macro and workload-macro means. This is the primary mechanism
metric.

It must not be replaced by Experiment 009's local packing SSE. Local codec
SSE, c4/c5 handoff risk, and frozen-input replay distortion are secondary
diagnostics.

### Required secondary metrics

Report:

- task-macro and workload-macro mean KL to matched FP32 logits;
- task-macro CVaR95 token KL;
- top-1 agreement;
- paired task wins, ties, and maximum NLL disadvantage;
- local pack and handoff NMSE;
- c4 and c5 counts, tie count, interval histogram, writes per token, and
  buffer occupancy;
- logical packed, selector, buffer, controller, live, allocated, temporary,
  CUDA-allocated, CUDA-reserved, and end-to-end peak bytes;
- RULER deterministic exact match;
- sandboxed HumanEval+ pass@1; and
- any non-finite, interrupted, retried, or excluded record.

Generated-task scores are secondary because the frozen Base model may have
weak instruction-following behavior. They cannot replace aligned NLL.

### Uncertainty

Use 10,000 paired hierarchical bootstrap replicates with seed `2339`:

1. resample tasks with replacement independently within each workload;
2. compute each workload macro;
3. average the three workload macros; and
4. recompute the strongest eligible comparator inside each replicate.

Report the two-sided 95% equal-tailed interval for comparator-minus-StateLease
improvement. This resampling rule accounts for comparator selection rather
than freezing whichever baseline happened to win by point estimate.

## Frozen Stage-B advancement gate

Every condition below is conjunctive:

1. identity, independent verification, finiteness, token alignment, hashes,
   protected-window, and resume-integrity checks pass;
2. StateLease allocates exactly `3,454,664` logical persistent bytes and no
   hidden persistent FP32 state;
3. its workload-macro excess NLL is lower than every equal-allocation fixed
   replay comparator and every equal-total-byte no-replay comparator;
4. its workload-macro excess NLL is at least 15% lower than the strongest
   fixed replay comparator; if that comparator's excess NLL is zero or
   negative, this gate fails closed;
5. the hierarchical paired-bootstrap 95% lower bound for the strongest
   equal-byte comparator minus StateLease excess NLL is above zero;
6. StateLease wins strict task-level NLL comparisons against the strongest
   fixed comparator on at least two thirds of tasks in each workload:
   `22/32` PG19, `32/48` RULER, and `22/32` HumanEval+; ties are not wins;
7. reference-aligned trajectory-NMSE AUC is at least 20% lower than the
   strongest fixed replay comparator in every workload;
8. workload-macro mean KL is lower, CVaR95 KL is no higher, and top-1
   agreement trails by no more than `0.005` versus the strongest equal-byte
   comparator;
9. no task has StateLease minus strongest-equal-byte-comparator excess-NLL
   disadvantage above `0.20` nats per token;
10. every per-layer checkpoint advance is four or five tokens, exact ties
    choose c5, and checkpoint writes do not exceed one per four decode tokens
    per recurrent layer after warm-up;
11. when the FP32 RULER exact-match score is at least 10%, StateLease is no
    more than one percentage point below the strongest fixed comparator and
    no more than two points below FP32; below that FP32 floor, exact match is
    reported but is not an advancement gate; and
12. the independent verifier reproduces all aggregate decisions directly from
    the authenticated artifact.

A failure of any condition authenticates a negative Experiment 010 result and
stops the candidate. A favorable workload, task subset, controller helper,
threshold, lease length, per-layer schedule, alternative risk, precision,
buffer dtype, RHT seed, or comparator cannot be relabelled as the primary.
Such a change requires Experiment 011 and fresh data.

## Run and interruption discipline

Stage A and Stage B each permit one authenticated quality run. The evaluator
must append per-task records transactionally and withhold aggregate output
until the stage is complete.

An infrastructure interruption may resume only when:

- method, source, model, runtime, and identity hashes are unchanged;
- already completed task records pass semantic and cryptographic validation;
- no failed numerical gate has been observed;
- the resume omits exactly the authenticated completed identities; and
- the artifact records every attempt and resume boundary.

A code, method, threshold, dataset, or identity change is not a resume.

## Stage C: future confirmation and cross-model boundary

Stage B is development evidence only. A confirmation claim requires:

1. freezing the final method and verifier after Stage B;
2. a new identity artifact from disjoint PG19 test material, new RULER seeds,
   and unused HumanEval+ identities;
3. one untouched run on the pinned 0.8B model;
4. one untouched run on a separately pinned Qwen3.5 Base checkpoint with a
   different model size; and
5. no model-specific retuning.

Before the cross-model run, its exact recurrent geometry and byte formula must
be derived from the pinned config. No model revision is implied by this
protocol. If geometry or integration differs, it must be documented before
quality access.

For cross-model confirmation, StateLease must have positive point improvement
over the strongest fixed comparator on each model, at least 10% excess-NLL
reduction per model, a positive pooled and per-model paired 95% lower bound,
the same generated-score non-inferiority rule, and no maximum-disadvantage
breach. Failure on the second model limits any finding to the 0.8B checkpoint.

## Stage D: kernel and systems claim boundary

The reference implementation is allowed to be slow and to materialize
temporary FP32 workspaces. It supports only quality and logical
persistent-storage research.

A deployment or speed claim requires a separate fused CUDA or Triton path
that:

- consumes the physically packed RHT checkpoint directly;
- consumes the BF16 `(u, k)` and FP32 `g` buffer directly;
- does not retain or round-trip through a persistent FP32 state mirror;
- preserves the frozen c4/c5 decision and exact byte contract;
- reports temporary and peak HBM rather than hiding workspaces;
- passes the independent numerical verifier; and
- compares against optimized FP32, fixed-replay, and no-replay kernels.

Benchmark at batch size one and at useful serving batch sizes, with fixed
prompt/decode lengths, CUDA events, at least 50 warm-up iterations, 200 timed
iterations, and five fresh process starts. Report medians, interquartile
ranges, and paired confidence intervals.

The packed path may be described as deployment-viable only if end-to-end
throughput is no worse than 5% below the optimized FP32-cache path while
measured peak HBM is lower. A speedup claim additionally requires the 95%
lower confidence bound for end-to-end throughput ratio to exceed `1.00`.
Logical payload arithmetic alone is not a latency or whole-model memory
result.

## Claim boundary after a possible pass

After Stage B alone, the strongest permitted wording is:

> On a pinned Qwen3.5-0.8B-Base checkpoint and three frozen development
> workloads, StateLease-H5 reduced reference-aligned trajectory drift and
> aligned excess next-token NLL at 3,454,664 logical persistent bytes relative
> to the exact named baselines.

That sentence may be used only with the measured values and authenticated
artifact filled in.

Even after a Stage-B pass, do not claim:

- a breakthrough, state of the art, or first method;
- repair or removal of accumulated historical error;
- certified, bounded, lossless, or exact model behavior;
- generality across recurrent architectures or model scales;
- generated-code or long-context superiority beyond the scored panel;
- faster inference;
- lower whole-model or end-to-end peak memory; or
- independent reproduction.

Checkpoint/replay, update buffering, fixed checkpoint periods, adaptive cache
precision, rotations, mixed-bit allocation, and residual correction all have
prior art. Experiment 010 is designed to find out whether this one audited
composition and boundary rule works, not to write the conclusion in advance.
