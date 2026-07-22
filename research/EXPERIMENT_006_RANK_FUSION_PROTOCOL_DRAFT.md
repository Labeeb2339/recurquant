# Experiment 006: quota-constrained rank fusion protocol draft

> **Status: pre-holdout protocol freeze. The ranked `[8, 16)` window is
> unopened.**
>
> The immutable primary is equal rank fusion with `lambda = 0.5`. Values
> `lambda = 0.25` and `lambda = 0.75` are predeclared ablations only; neither
> may replace the primary after results are visible. The `lambda = 0` and
> `lambda = 1` endpoints are controls. This protocol makes no improvement,
> novelty, systems, or breakthrough claim.

Date opened: 2026-07-23

## Relation to Experiment 005

Experiment 005 failed its frozen storage-boundary sign gate: 13 of 16 checks
agreed, or `0.8125`, below the required `0.95`. That failure is permanent and
is recorded in `EXPERIMENT_005_RESULT.md`. It stopped Experiment 005 before
the ranked `[8, 16)` heldout-calibration window was opened.

The failed gate validated a signed local derivative,

```text
g dot (Q8(raw) - Q4(raw)),
```

not the squared endpoint score that generated the layer quotas and not the
reconstruction-error score used at runtime. Experiment 006 therefore uses a
new candidate-aligned numerical and packing gate. The old finite-difference
result remains available as a labeled non-gating diagnostic; it is not erased,
rerun under looser thresholds, or inherited as evidence for this candidate.

## Research question

At the same exact resident recurrent-state byte budget and the same frozen
per-layer INT8 quotas, does a deterministic equal-rank combination of:

1. offline target-directional sensitivity; and
2. causal per-write INT4-to-INT8 reconstruction-error reduction

preserve Qwen3.5 Gated DeltaNet outputs better than either component alone,
the strongest equal-byte static policy, and the H1-quota adaptive control on a
previously unopened eight-task calibration window?

This is a small heldout-calibration diagnostic. It is not a confirmation set.

## Frozen candidate

### Physical rows and layer quotas

Each Gated DeltaNet recurrent layer contains 2,048 physical matrix rows:
16 heads by 128 key rows. A row's canonical coordinate is
`(layer_index, head_index, key_row_index)`, with flattened in-layer index

```text
head_index * 128 + key_row_index.
```

The offline selector computes the frozen
`target_directional_fisher_difference_int4` score for every row. For a scored
storage boundary, with target-NLL gradient `g` and aligned quantization errors

```text
e4 = Q4(raw) - raw
e8 = Q8(raw) - raw,
```

the primitive score is

```text
(g dot e4)^2 - (g dot e8)^2.
```

Task-macro aggregation over the authenticated first eight selector tasks gives
the static row score `S_(l,r)`. Negative values are retained. A global stable
descending rank under the exact format budget selects 1,976 rows. The number
selected in layer `l` becomes its immutable quota `q_l`, with

```text
sum_l q_l = 1,976.
```

The exact quota vector and the global selected mask must be recorded and
hashed in the regenerated selector artifact. Experiment 006 never retunes
`q_l` on the heldout window.

### Per-write reconstruction score

At each real recurrent-state write, the dynamic score for row `r` in layer
`l` is

```text
M_(t,l,r) = mean((Q4(raw_(t,l,r)) - raw_(t,l,r))^2)
            - mean((Q8(raw_(t,l,r)) - raw_(t,l,r))^2).
```

The Q4 and Q8 errors are aligned endpoints of the same source row and the same
quantizer contract. Higher values mean that promoting the row removes more
instantaneous reconstruction error.

### Deterministic per-layer ranks

For each layer independently:

- `r_S(l,r)` is the zero-based descending rank of `S_(l,r)`;
- `r_M(t,l,r)` is the zero-based descending rank of `M_(t,l,r)` at that write;
- rank 0 is best; and
- exact score ties are resolved by ascending canonical flattened in-layer
  coordinate.

Ranks are permutations of the integers 0 through 2,047. No unstable GPU sort,
random tie break, epsilon tie, or rounded display value may determine a rank.
The ranking implementation must produce the same integer arrays as the
canonical CPU FP64 implementation.

### Frozen fusion family

`lambda` is the weight on the **static sensitivity rank**. It does not mix
directional Fisher with diagonal Fisher. Multiplying by four gives exact
integer costs and avoids floating-point tie ambiguity:

| Role | `lambda` | Integer cost to minimize |
| --- | ---: | --- |
| Endpoint control: plain adaptive MSE | `0` | `4 * r_M` |
| Predeclared ablation | `0.25` | `3 * r_M + r_S` |
| **Frozen primary: equal rank fusion** | **`0.5`** | **`2 * r_M + 2 * r_S`** |
| Predeclared ablation | `0.75` | `r_M + 3 * r_S` |
| Endpoint control: static target-Fisher plan | `1` | `4 * r_S` |

For every write and layer, select exactly the `q_l` rows with smallest fusion
cost. Final cost ties are resolved by ascending canonical flattened coordinate.
The physical packer stores those rows as INT8 and all other rows as INT4.

The primary artifact method name is frozen as:

```text
rank_fusion_l050_target_fisher_adaptive_mse
```

The `lambda = 0.25` and `lambda = 0.75` results must be reported regardless of
sign, but they are ablations. A better ablation cannot be promoted to primary,
used for the pass decision, or described as the prespecified result. Any later
choice of a different weight is a new experiment on new data.

### Endpoint invariants

Before any heldout access:

1. `lambda = 0` must reproduce the existing
   `adaptive_mse_target_directional_fisher_quota` mask exactly at every
   audited write;
2. `lambda = 1` must reproduce the frozen static
   `target_directional_fisher_difference_int4` mask exactly at every write;
3. each layer must select exactly `q_l` rows for every lambda value;
4. the total promotion count must be exactly 1,976; and
5. repeated runs from identical inputs must produce byte-identical ranks,
   costs, masks, payloads, and hashes.

One mismatch fails closed. The endpoints are controls, not alternative primary
methods.

### Causality boundary

The experiment is batch-one. Static ranks are frozen from the selector
partition. At runtime, the dynamic rank may use only the current raw state and
its current aligned Q4/Q8 endpoints before that state is stored. It may not use
the next token, future token, target label, future query, future state, or
future logits.

## Exact storage contract

For Qwen3.5-0.8B-Base at batch one:

| Component | Rank-fusion/static row format | Static v0.2 |
| --- | ---: | ---: |
| All-INT4 payload | 2,359,296 B | 2,359,296 B |
| FP16 scales | 73,728 B | 73,728 B |
| Precision mask | 4,608 B | 0 B |
| INT8 promotion payload | 126,464 B | 131,072 B |
| Promoted rows | 1,976 | 2,048 |
| **Resident total** | **2,564,096 B** | **2,564,096 B** |

The evaluator checks these values from live packed tensors after every task
for every rank-fusion method. Resident bytes exclude model weights, the full
attention cache, temporary endpoint tensors, rank workspaces, kernel
workspace, framework bookkeeping, and CUDA allocator reserved memory.

## Candidate-aligned numerical gate

The gate validates the score algebra actually used by this candidate and the
physical masks derived from it. All reference calculations run on CPU FP64
from the exact source FP32 tensors saved before score reduction. The model-side
implementation may remain FP32, but it must agree with the independent
reference within the deterministic bounds below and must produce identical
ranks and masks.

This is an implementation-validity audit, not a statistical sample. Its
geometry is fixed without reading selector scores:

- selector-task positions `0` and `7` in authenticated order;
- the first and last affected code transitions in each selected task;
- recurrent model layers `0`, `9`, `18`, and `22`;
- value heads `0`, `7`, and `15`; and
- key rows `0`, `31`, `63`, `95`, and `127`.

The static dot-product audit therefore contains exactly 240 primitive rows.
For the same 16 task-transition-layer writes, the dynamic audit retains the
whole `[16, 128, 128]` source state and validates all 2,048 row scores, ranks,
and the final mask. This scope is small enough to release as raw sidecar data
while covering early through late layers, heads, rows, short-horizon and
long-horizon boundaries. Expanding or replacing this set after inspecting its
result is a new gate, not an Experiment 006 retry.

Let FP32 unit roundoff be

```text
u = 2^-24 = 5.960464477539063e-8.
```

For a reduction of length 128, freeze

```text
gamma_128 = 128*u / (1 - 128*u)
          = 7.629452739355006e-6

gamma_3 = 3*u / (1 - 3*u)
        = 1.788139663006007e-7.
```

### Static sensitivity score

For every one of the 240 frozen primitive rows, independently compute for `b`
in `{4, 8}`:

```text
d_b = sum_i float64(g_i) * float64(e_b_i)
A_b = sum_i abs(float64(g_i) * float64(e_b_i))
B_b = gamma_128 * A_b
```

and

```text
S_ref = d_4^2 - d_8^2

B_S = 2*abs(d_4)*B_4 + B_4^2
    + 2*abs(d_8)*B_8 + B_8^2
    + gamma_3 * ((abs(d_4) + B_4)^2 + (abs(d_8) + B_8)^2).
```

For all 240 finite primitive rows, the implementation score must satisfy

```text
abs(S_impl - S_ref) <= B_S.
```

There is no aggregate pass percentage. Separately, the selector must retain
authenticated per-task FP64 accumulator arrays for the complete 36,864-row
production score field. Independent CPU code recomputes their task-macro mean,
stable global plan, per-layer `r_S`, `q_l`, and endpoint mask from those stored
arrays. The production artifact must match every complete-array hash, rank,
quota, coordinate, and mask exactly. The sampled primitive audit validates the
underlying dot-and-square implementation; the full-array replay validates its
aggregation and allocation without requiring tens of gigabytes of raw
gradient/error tensors.

### Dynamic reconstruction score

For all 2,048 rows at each of the 16 frozen writes, independently compute

```text
m_b = (1/128) * sum_i float64(e_b_i)^2
M_ref = m_4 - m_8

gamma_256 = 256*u / (1 - 256*u)
          = 1.5259021896696422e-5

B_M = gamma_256 * (m_4 + m_8)
    + 8*epsilon_FP64*max(1, m_4 + m_8).
```

For every finite audited row, require

```text
abs(M_impl - M_ref) <= B_M.
```

The small FP64 term is only a deterministic representation guard. For each
audited write, canonical CPU FP64 values determine all 2,048 `r_M` positions;
the production ranks, integer fusion costs, masks, and packed endpoints must
match exactly.

These bounds assume finite, normal-range arithmetic without overflow or
underflow. A non-finite source, score, metric, payload, or model output, or an
overflow/underflow condition affecting a checked reduction, fails closed and
is recorded rather than omitted. If the implementation is changed to use the
canonical FP64 scores directly, the artifact must still store the exact FP64
array hashes and prove rank, quota, mask, and packing parity.

### Exact tensor-byte authentication

Every released audit source and derived array is content-addressed using its
exact logical bytes, not JSON-rendered decimal strings. The artifact records
dtype, shape, axis meaning, little-endian byte order, contiguous C-order, and
SHA-256 for:

- the 240 sampled source FP32 `g`, `e4`, and `e8` rows;
- the 16 full audited FP32 source states and their physical Q4/Q8 endpoints;
- sampled canonical FP64 primitive scores and complete authenticated per-task
  FP64 selector accumulator arrays;
- canonical FP64 per-write MSE scores for all audited rows;
- per-layer static and dynamic integer ranks;
- the integer quota vector;
- every lambda's integer fusion-cost arrays;
- selected masks and promotion coordinates; and
- packed INT4 payloads, INT8 promotion payloads, FP16 scales, and mask bytes.

Large raw tensors may live in a deterministic `.npz` sidecar, but the JSON gate
artifact must authenticate the sidecar file and every contained logical array.
The validator recomputes task-macro aggregation, stable ties, exact layer
quotas, all 1,976 promotions, and the 2,564,096-byte resident total. Every
check must pass. The gate artifact reports each condition and an overall
conjunction and exits nonzero on failure.

## Artifact and provenance prerequisites

Experiment 006 uses distinct schema and artifact kinds. No Experiment 004 or
005 artifact may be renamed or relabeled as Experiment 006 evidence. Before
holdout access, the following must exist from one clean committed protocol and
implementation:

1. a regenerated target-directional selector artifact;
2. a regenerated H1 selector artifact for the quota-control method;
3. a passing candidate-aligned numerical, rank, endpoint-parity, byte, and
   packing gate artifact; and
4. a preflight record authenticating the unopened holdout and all inputs.

Each artifact must record:

- its schema version, artifact kind, file SHA-256, and canonical evidence
  SHA-256;
- the committed SHA-256 of this protocol and every relevant source file;
- the same clean Git commit at start and end, with identical source maps;
- model `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`;
- dataset `google-research-datasets/mbpp`, config `full`, source split `train`,
  calibration phase, at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`;
- selection namespace `rq-v0.2` and formatter version
  `recurquant.mbpp-prompt-code.v1`;
- the exact tokenizer identity, revision, settings, rendered text hashes,
  ordered task IDs, canonical row-content hashes, and aligned token counts;
- the quantizer contract, rounding mode, scale representation, storage timing,
  tensor shapes, row-coordinate convention, and recurrent-layer list;
- `q_l`, all method names, the complete lambda table, integer cost formulas,
  ranks, masks, exact tensor-byte hashes, and endpoint-parity results;
- bootstrap seed 2339, random-row seed 1101, all other RNG seeds and states,
  deterministic settings, and commands;
- Python, PyTorch, Transformers, CUDA, driver, package-lock, OS, CPU, GPU, and
  hardware details; and
- every gate threshold, observation, pass value, exception, warning, and exit
  status.

The clean commit must contain both the frozen protocol and implementation. All
selector and gate artifacts must be generated afterward from that exact
commit. A dirty tree, a source hash change, an uncommitted protocol, or an
artifact from a predecessor commit fails preflight.

### Existing evidence is exploratory only

The following authenticated artifacts predate this fusion protocol:

| Artifact | File SHA-256 | Canonical evidence SHA-256 | Classification |
| --- | --- | --- | --- |
| `experiment005-storage-boundary-599862e.json` | `61a2936bd20679bad441921d26b556f5986eec61449c4a9743c9b0b5e0bea86d` | `b168330b4c39963b7c149230d4b1ad9fa57b20b02d6d95ee583ee9941f68b19f` | frozen failed E005 gate, `passed: false` |
| `experiment006-hrr-selector-8task-599862e.json` | `7a67e159f9dbab5e92cc9a837831359d3aa180ccd4725807f1836a7d5aeba55a` | `5acbe38d575c741153cf0142bced9227a9654693f946e4d768b9b62847bb672b` | pre-freeze selector input |
| `experiment006-loss-selector-8task-599862e.json` | `b4972b40b32d67557520c0af3d3c4467f69283819765f8c3e1c9f4b92f560bf6` | `9243ec49933442c4d48f6e51321c33010d681fc5a3fffa4cca70599d52cf26d4` | pre-freeze selector input |
| `experiment006-hrr-selector-8task-556e527.json` | `fa02e1d468ecc13c78b7cf8e63f237e372c556d9fed0c1f4b47c9dd901a808dd` | `07e646ccb9b1df5ff9873a94f7bacb07d7a4e2b70136e3a68f40d1619814d899` | pre-freeze selector input |
| `experiment006-loss-selector-8task-556e527.json` | `33bdc5939429281ba5377eeb02d59fac72a0f8da657c713bb7854d235e2fb057` | `ae92af38475720eb1ce19527f1c2de3d0d1fc045a1160a83f8da30ecde282214` | pre-freeze selector input |
| `experiment006-adaptive-same-calibration-8task-556e527.json` | `3495698932b43d93f387bb61492f91fc38840097020980977451be2042a02164` | `f07a6b852a4c427c0b9946ad44c21d023299d173eb24695462e99e3785061d61` | E005 same-calibration postmortem; no fusion |

They may be cited for history and used to test artifact readers. They cannot
satisfy Experiment 006's regenerated selector, candidate-aligned gate,
preflight, or heldout-result requirements.

## Frozen methods and controls

Every row-format method below uses exactly 1,976 promotions and the exact
2,564,096-byte resident recurrent-state total unless explicitly noted:

1. `rank_fusion_l050_target_fisher_adaptive_mse` - frozen primary.
2. `rank_fusion_l025_target_fisher_adaptive_mse` - report-only ablation.
3. `rank_fusion_l075_target_fisher_adaptive_mse` - report-only ablation.
4. `adaptive_mse_target_directional_fisher_quota` - `lambda = 0` endpoint and
   individual adaptive component.
5. `target_directional_fisher_difference_int4` - `lambda = 1` endpoint and
   equal-byte static comparator.
6. `adaptive_mse_hrr_h1_quota` - individual adaptive quota control.
7. `hrr_h1` - static comparator.
8. `hrr_h32` - retained negative static hypothesis.
9. `row_mse` - static task-macro reconstruction-error allocation.
10. `v02_layer0_static` - equal resident bytes with 2,048 promotions and no
    precision mask.
11. `random_rows_s1101` - prespecified random static policy.
12. `signed_taylor_next_int4` - retained static diagnostic.
13. `target_diagonal_fisher_difference_int4` - retained static diagnostic.
14. `delta_direction_magnitude_int4` - retained static diagnostic.
15. `uniform_int4` - lower-byte control, not an equal-byte competitor.

The strongest individual adaptive component is selected deterministically as
the lower-macro-excess-NLL member of methods 4 and 6 on the frozen holdout.
The `lambda = 1` endpoint is static and is not eligible for that label. The
`lambda = 0.25` and `lambda = 0.75` fused ablations are not individual
components and cannot replace or qualify the primary.

## Data separation and one-time holdout audit

The authenticated first eight ranked MBPP calibration rows are the selector
partition. They have already been inspected and can support only development
and same-calibration diagnostics. Repository evidence at this protocol freeze
contains selector-prefix and offset-0 postmortem artifacts, but no artifact
that opens the ranked `[8, 16)` window.

The one-time Experiment 006 window is frozen as:

```text
phase: calibration
ranked offset: 8
task count: 8
window: [8, 16)
```

Before tokenization or model loading, the heldout evaluator must:

1. verify both regenerated selector artifacts, the passing Experiment 006
   numerical/packing gate, this protocol hash, and the same clean commit;
2. reconstruct and authenticate the pinned ranked population independently;
3. require exact arguments `--calibration-offset 8` and `--limit 8`;
4. authenticate the first-eight selector prefix by dataset/config/revision,
   source split, phase, namespace, formatter, ordered IDs, and row hashes;
5. prove `[8, 16)` is disjoint from every selector and development task ID;
6. verify that no prior artifact, log, cache manifest, command record, or
   source change shows that `[8, 16)` was opened; and
7. record a passing preflight before loading any heldout text.

If there is evidence that `[8, 16)` was previously opened, the protocol aborts.
It must not silently choose another offset under the Experiment 006 name.

After preflight, one invocation evaluates every frozen method on the same
ordered tasks and aligned transitions. It writes a complete artifact even when
a metric or gate fails, records the failure, and exits with status 2. A crash
may be resumed only from authenticated, immutable per-task records without
rerunning, omitting, or replacing a revealed task. No lambda, quota, component,
method, threshold, seed, metric, or comparator may change after heldout access.

The evaluator records the heldout row-content manifest, rendered-text hashes,
token counts, source maps, commands, packages, hardware, clean commit, and
start/end hashes. Start and end records must be identical.

## Metrics and frozen quality gate

The primary metric is task-macro excess next-token NLL relative to an FP32
recurrent state, scored only on code transitions after a quantized recurrent
state has been stored. The prompt-to-first-code-token prediction is excluded.
Use 10,000 paired bootstrap resamples with seed 2339.

The equal-byte static comparator set is unchanged from Experiment 005:

```text
hrr_h1
hrr_h32
row_mse
random_rows_s1101
v02_layer0_static
signed_taylor_next_int4
target_directional_fisher_difference_int4
target_diagonal_fisher_difference_int4
delta_direction_magnitude_int4
```

The strongest equal-byte static comparator is the member with the lowest
macro excess NLL on the frozen window. Uniform INT4 is excluded because it
uses fewer resident bytes. Adaptive methods and fused ablations are excluded
from the static set.

The `lambda = 0.5` primary passes this small heldout-calibration gate only if
all conditions hold:

1. resident recurrent-state bytes equal 2,564,096 exactly on every task;
2. primary macro excess NLL is lower than every equal-byte static comparator;
3. the paired 95% interval versus the strongest equal-byte static comparator
   is strictly above zero when expressed as comparator minus primary;
4. relative excess-NLL reduction versus that static comparator is at least
   20%; if its excess NLL is non-positive, this condition fails closed;
5. the paired 95% interval versus `adaptive_mse_hrr_h1_quota` is strictly above
   zero when expressed as H1-adaptive minus primary;
6. the paired 95% interval versus the strongest individual adaptive component
   defined above is strictly above zero when expressed as component minus
   primary;
7. macro top-1 agreement is no more than `0.01` below the strongest equal-byte
   static comparator;
8. macro worst-token KL CVaR95 is no more than `0.10` above that comparator;
9. on every task, primary excess NLL is at most `1.0` worse than that
   comparator's excess NLL;
10. no metric, source tensor, score, payload, or model output is non-finite;
11. the candidate-aligned numerical, rank, endpoint-parity, promotion-count,
    and exact-byte gate passes every row and write; and
12. every code, protocol, artifact, manifest, tensor-byte, and source hash
    verifies, with the same clean repository commit and source map at start and
    end.

Conditions 1 through 5 and 7 through 12 preserve Experiment 005's quality and
evidence thresholds. Condition 6 is the additional rank-fusion requirement:
the fused primary must beat its strongest individual adaptive component with
a paired lower confidence bound above zero. The evaluator records every
threshold, observed value, interval, pass value, and the overall conjunction.
There is no discretionary override.

The two intermediate-lambda ablations are always reported but do not enter the
primary pass conjunction. They can diagnose the shape of the tradeoff; they
cannot rescue a failed `lambda = 0.5` result.

## Systems boundary

This prototype computes two quantized endpoints, CPU-reference audits, stable
ranks, integer fusion costs, and masks in Python. It is expected to add
overhead. Resident packed-state bytes alone do not establish lower end-to-end
memory, faster decoding, or practical deployment value.

No latency, throughput, peak-memory, energy, kernel-efficiency, or speed claim
is permitted from this experiment. A later fused Triton or CUDA implementation
must preserve exact mask and endpoint parity and separately benchmark p50/p95
decode latency, throughput, peak allocated and reserved memory, and workspace
against static packing on named hardware.

## Prior-art and novelty boundary

Rank aggregation, Fisher/Taylor sensitivity, reconstruction-error selection,
mixed-bit quantization, layer quotas, dynamic precision, and recurrent-state
cache quantization are established ideas. No one ingredient is claimed as new.

The narrow testable contribution is the particular exact-byte combination of
frozen target-directional layer quotas, per-layer static sensitivity ranks,
causal per-write aligned Q4-to-Q8 MSE ranks, and deterministic integer rank
fusion for a Gated DeltaNet language-model recurrent state.

A pass on eight MBPP calibration tasks would show only a scoped signal worth
further study. It would not establish architectural generality, useful speed,
state-of-the-art quality, or a breakthrough. Promotion beyond this stage
requires, at minimum:

- a larger untouched code development set and a separately frozen test set;
- natural-text, retrieval, free-generation, and long-context evaluations;
- multiple checkpoints, model sizes, recurrent architectures, and rounding
  seeds;
- closest-method reproductions and full systems benchmarks; and
- independent replication from released code and raw artifacts.

## Decision rule

- **Gate failure before holdout:** record the failed prerequisite; keep the
  holdout closed.
- **Heldout primary failure:** publish the complete artifact and stop this
  candidate. Do not tune lambda on `[8, 16)`.
- **Ablation wins but primary fails:** report the ablation result as exploratory
  and open a new protocol on new data; do not relabel it primary.
- **Primary passes:** advance only to a larger preregistered development stage.
  Do not use breakthrough language.

The current state is below the heldout-calibration rung of the claim ladder.
