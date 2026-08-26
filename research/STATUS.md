# Research status

## v0.3 experimental track

Last updated: 2026-08-25

Experiment 005 stopped before holdout after its frozen real-storage-boundary
sign gate achieved `13/16 = 0.8125`, below the required `0.95`. Its permanent
failure, authenticated artifact, and same-calibration postmortem are recorded
in [`EXPERIMENT_005_RESULT.md`](EXPERIMENT_005_RESULT.md). The ranked MBPP
calibration window `[8, 16)` was not opened.

Experiment 006 tested deterministic ordinal-rank fusion of offline
target-directional sensitivity and causal per-write reconstruction benefit at
the same exact byte budget. On the already inspected eight-task selector
partition, its frozen `lambda = 0.5` primary had macro excess NLL `0.514873`,
worse by point estimate than plain adaptive MSE at `0.493302`; the paired 95%
interval crossed zero. It improved only 3.90% over the strongest static method,
also with an interval crossing zero. The better `0.25` and `0.75` ablations
cannot replace the frozen primary. The candidate was therefore stopped before
its numerical prerequisite or holdout. The ranked `[8, 16)` window remains
unopened. See [`EXPERIMENT_006_RESULT.md`](EXPERIMENT_006_RESULT.md).

Experiment 007 tested CQER-32: a causal 32-token EMA of normalized query energy
times exact per-write INT4-to-INT8 row reconstruction benefit, with the same
frozen target-Fisher layer quotas. On the already inspected eight-task
partition it lowered macro excess NLL to `0.462792`, a 6.18% descriptive
reduction from plain adaptive MSE and 13.62% from static target-Fisher. Both
paired 95% intervals crossed zero. The frozen gate failed because the static
reduction was below 20% and top-1 agreement trailed the better comparator by
`0.02690`, above the `0.01` margin. All exact-byte, causal-handshake, finiteness,
and integrity checks passed. Experiment 007 therefore stopped before its FP64
prerequisite or holdout; ranked `[8, 16)` remains unopened. See
[`EXPERIMENT_007_RESULT.md`](EXPERIMENT_007_RESULT.md).

Experiment 008 tested CORA-C2 on its separately frozen 16-task `[16, 32)`
development window. CORA-C2 improved macro excess NLL by 26.17% over static
target-Fisher and 21.62% over adaptive MSE, but it was 10.54% worse than
CQER-32. Raw CORA was also 3.35% worse than CQER-32. Confirmation-2 reduced
normalized committed-mask churn by 79.99%, yet worsened raw-CORA NLL by 6.96%,
above the frozen 1% limit. Five advancement checks failed. The authenticated
result is recorded in [`EXPERIMENT_008_RESULT.md`](EXPERIMENT_008_RESULT.md).
The independent verifier was not reached and ranked `[8, 16)` remains
protected.

Experiment 009 tested RHT-CQER-32, which composes CQER-32 with a deterministic
orthonormal right-side Hadamard codec while preserving the exact Q4/Q8 packed
state and selector byte counts. Its one-task Stage-A falsification screen on
already-open task 666 passed all nine frozen checks: closed-loop state SSE
fell `59.97%`, aligned excess NLL fell `58.59%`, and aligned mean KL fell
`31.19%` relative to CQER-32 while top-1 agreement and bytes were unchanged.
The authenticated screen is in
[`EXPERIMENT_009_STAGE_A_RESULT.md`](EXPERIMENT_009_STAGE_A_RESULT.md).

The separately frozen 32-task Stage-B development run then passed all eight
advancement checks. Relative to CQER-32, task-macro aligned excess NLL fell
from `0.323944` to `0.153129` (`52.73%`), with 27 of 32 strict task wins and a
paired 95% bootstrap interval of `[0.116082, 0.229438]`. Aggregate local
recurrent-state reconstruction SSE fell from `36,409.363073` to
`15,345.844948` (`57.85%`). Mean KL and CVaR95 KL were lower, top-1 agreement
was higher, and the exact `2,564,096` packed-state bytes plus `147,456`
selector bytes were unchanged.

The frozen protocol, committed identity, authenticated result, post-write
verification record, and compact release manifest are:

- [`EXPERIMENT_009_RHT_CQER_PROTOCOL.md`](EXPERIMENT_009_RHT_CQER_PROTOCOL.md)
- [`EXPERIMENT_009_STAGE_B_IDENTITY.md`](EXPERIMENT_009_STAGE_B_IDENTITY.md)
- [`EXPERIMENT_009_STAGE_B_RESULT.md`](EXPERIMENT_009_STAGE_B_RESULT.md)
- [`EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md`](EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md)
- [`../evidence/experiment009-rht-cqer-stage-b-result-manifest.json`](../evidence/experiment009-rht-cqer-stage-b-result-manifest.json)

The public-stream application-level access boundary remains fixed in
[`EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md`](EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md).
Ranked MBPP window `[8, 16)` remained protected.

Experiment 012 tested the next falsification step for the StateLease controller.
StateLease-H5 adds replay-driven per-layer c4/c5 arbitration with an exact
`3,454,664`-byte allocation (`5.857110` bits per recurrent-state element). Its
one-task Stage-A screen on task `666` passed all eight frozen checks. Excess NLL
was `0.023349`, 17.90% below the strongest fixed-replay schedule at `0.028442`.
It was worse than the two strongest equal-total-byte no-replay codecs, at
`-0.000014` and `0.002461`, so the result does not establish a practical or
general advantage.

The full record is:

- [Experiment 012 StateLease-H5 identity](EXPERIMENT_012_STAGE_A_IDENTITY.md)
- [Experiment 012 StateLease-H5 protocol](EXPERIMENT_012_STATELEASE_PROTOCOL.md)
- [Experiment 012 StateLease-H5 Stage-A result](EXPERIMENT_012_STAGE_A_RESULT.md)
- [Experiment 012 machine-readable result](../evidence/experiment012-statelease-stage-a-666.json)

The StateLease-H5 design originated in Experiment 010. It keeps the frozen
RHT-CQER checkpoint, allocates a five-token BF16 `(u, k)` plus FP32 `g` replay
buffer per recurrent layer, and each layer independently makes one
threshold-free choice between c4 and c5 only when its buffer is full. Both
risks use the same normalized view of that layer's unchanged causal CQER EMA;
no global synchronized choice is used. An exact tie selects c5. The complete
logical persistent
allocation is `3,454,664` bytes, or `5.857110` bits per recurrent-state
element.

The protocol requires schedule-matched fixed replay, equal-total-byte
no-replay, mixed-bit, residual, FP32, and off-budget CC8 references. In
particular, the expanded equal-total-byte RHT Q4/Q8 comparator promotes
exactly 13,587 rows and contains 8 explicit reserved padding bytes. Stage A is
limited to already-open task 666. No StateLease Stage-B development result
exists. Ranked MBPP window `[8, 16)` remains sealed and is not authorized for
this experiment. Stage-A status for this controller is documented in
[`EXPERIMENT_012_STAGE_A_RESULT.md`](EXPERIMENT_012_STAGE_A_RESULT.md). See
[`EXPERIMENT_010_STATELEASE_PROTOCOL.md`](EXPERIMENT_010_STATELEASE_PROTOCOL.md).

Checkpoint/replay and Gated DeltaNet `(u, k, g)` buffering are established
prior art through Nemotron 3 Ultra and ReplaySSM. Even a future protocol pass
would not by itself support a first, novelty, speed, deployment,
state-of-the-art, or breakthrough claim.

Experiment 013 remains scientifically unresolved and has no accepted static
RHT-Q468 result.
The tenth replacement chain at H0
`a5188a0b3e7bc3ab9ab2a27a639cac26d93030bd` and H1
`874e586bda98602cc712a543a071a0047df38659` completed authenticated model
staging, its one Fisher H=1 smoke, full calibration, and post-calibration
Stage-A authorization. Its first sealed Stage-A identity capture then failed
closed before live source or model construction because recursive binding
verification attempted to import Torch inside the metadata-only isolation
boundary. That entire chain is retired; its successful calibration is custody
history, not an accepted static-Q468 result.

The Torch-free metadata repair produced eleventh replacement H0
`5626879cd8fafe422d85e1c3abb7fe46262ce57d`, tagged
`experiment013-h0-5626879`. GitHub Actions run `32791176375` passed all five
jobs. Its only calibration identity capture completed without model weights,
and candidate and promoted-format publication succeeded. The mandatory
pre-H1 comparison nevertheless rejected the result because the RULER formatter
SHA-256 changed outside the allowed pointer set. The formatter fingerprint had
accidentally reused the live capture-procedure version, so the security-only
v6-to-v7 change altered a purported scientific commitment. No H1 was created,
no model was staged, and the chain is retired and non-authorizing.

During root-cause analysis, an automated read-only process also decoded the
four protected seed-2,343 Stage-A RULER receipt bodies and constructed their
formatter targets. It printed no content and ran no model or metric, but the
access occurred outside the sealed Stage-A capture. Those four receipts and
every complete bundle containing them are conservatively retired. Seed 2,344
was selected before replacement generation as the sole new Stage-A seed.

An initial v18/v8/v7 repair draft removed the retired bundle digest, but a
pre-commit adversarial review found cross-phase manifest substitution and
swap/restore gaps. No commit or artifact was published under that rejected
draft. The current working tree instead implements runner v19,
capture/resolver v9, sealed Stage-A runner v8, the procedure-independent frozen
formatter epoch, seed 2,344, and explicit manifest custody through calibration
provenance v3, authorization v3, binding v5 and Stage-A provenance v2. Opaque
Stage-A bundle custody now also rederives the exact 16 calibration and four
Stage-A receipt hashes from authenticated frozen identities, reads each body
once without semantic decoding, and rejects substitutions before network work
or the one-run reservation.

The chain now has two gates. A non-authorizing G0 source commit keeps both the
runner and resolver manifest anchors unset, so official identity capture and
Stage-A authorization fail closed. After clean tests, package gates and
exact-commit CI, its generator may run exactly once from a clean detached
worktree at the published G0 tag and create one fresh 20-receipt bundle in new
roots. Failure or interruption retires those roots; resume and retry are not
allowed. A direct descendant final H0 must make only the two identical anchor
substitutions, freezing that bundle's exact whole-manifest SHA-256 before any
official capture. Capture consumes an immutable phase-scoped snapshot, so a
live A-to-B-to-A swap cannot affect consumed bytes; persistent changes are
separately detected. Final H0 is not authorizing until its own focused, full,
package and exact-commit CI gates pass and it is tagged.

The first G0 candidate,
`2cd83c944bdd0cb570a8abdee2d01520ae92cb41`, is rejected. Its local gates and
four CI packaging/wheel jobs passed, but Linux CI run `32918458848` exposed a
cross-platform raw-byte mismatch in the protocol-bound RULER requirements file:
Windows materialized 838 CRLF bytes while Linux materialized 798 LF bytes. The
repair keeps all 37 package pins unchanged while storing the prior canonical
838 CRLF runtime bytes verbatim in Git and marking only that file unfiltered.
Committed, indexed and materialized bytes must therefore remain identical on
every platform. The rejected commit has no tag, authorized no generation, and
the replacement candidate must repeat every G0 gate before it can be tagged.

A fresh calibration identity must retain all 160 calibration records
byte-for-byte and differ from the last valid retired identity only at the
source/procedure/promotion bindings and the two deterministic RULER
schedule/bundle commitments. No protected Stage-A receipt may be inspected
outside its eventual sealed capture. No Stage-A evaluation, model staging,
quality result, deployment conclusion, novelty claim, or breakthrough claim is
currently authorized. See
[`EXPERIMENT_013_STATIC_RHT_Q468_PROTOCOL.md`](EXPERIMENT_013_STATIC_RHT_Q468_PROTOCOL.md).

In parallel, the repository now contains a reference physical Q4/Q6/Q8
packer and an exact dynamic-programming allocator. Its corrected two-bit
metadata contract provides 3,808 marginal precision steps—not 3,952 rows—at
the same `2,564,096` state bytes. It is not yet cache-integrated or
quality-evaluated. See
[`MULTIBIT_REFERENCE_DESIGN.md`](MULTIBIT_REFERENCE_DESIGN.md).

The nearest known mechanism-level comparisons include MixKVQ's query-magnitude
and quantization-difficulty scoring and established randomized-Hadamard or
rotation quantizers. CQER-32 cannot be described as the first query-aware
mixed-precision method, and RHT-CQER-32 cannot be described as the first
rotation-based quantizer.

Stage B supports a scoped positive development result for one frozen method,
pinned model, and 32-task MBPP window. It is not held-out confirmation for
RHT-CQER-32 and does not support a generalized v0.3, novelty, speed,
state-of-the-art, deployment, or breakthrough claim.

## v0.2 confirmed release

The frozen v0.2 public-data study completed on 2026-07-22. Every preregistered
quality gate passed on all 500 untouched MBPP test tasks and 30,244 scored
tokens. The exact result, integrity anchors, infrastructure-resume record, and
claim boundary are in
[`CONFIRMATION_002.md`](CONFIRMATION_002.md). The frozen design remains in
[`PUBLIC_EVAL_PROTOCOL_V02.md`](PUBLIC_EVAL_PROTOCOL_V02.md), and the earlier
development decision is in [`DEVELOPMENT_002.md`](DEVELOPMENT_002.md).

## v0.1 pilot archive

The remainder of this file preserves the diagnostic v0.1 snapshot. Do not read
its historical "next action" as the current project state.

Snapshot date: 2026-07-22

## Confirmed implementation

- Fresh CUDA environment: PyTorch 2.11.0+cu128, Transformers 5.14.1.
- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB VRAM.
- Pinned model: `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Captured 18 recurrent states at `[1, 16, 128, 128]`, FP32, totalling 18 MiB.
- QDQ occurs once after prefill and after every teacher-forced decode token.
- At this snapshot, all 26 unit/integration tests and lint were green.

## Calibration and development evidence

Both traces use 32 prefill and 32 teacher-forced decode tokens. Results are
diagnostic because the text is synthetic.

Static retrieval baselines:

| State format | Mean token KL | Worst-5% token KL | Top-1 agreement |
|---|---:|---:|---:|
| INT8 nearest | 0.01551 | 0.04394 | 0.90625 |
| INT8 stochastic | 0.03058 | 0.11130 | 0.90625 |
| INT6 nearest | 0.09016 | 0.38548 | 0.84375 |
| INT4 nearest | 2.18483 | 5.94331 | 0.43750 |

Stochastic INT8 was worse than deterministic nearest rounding on this short
trace. That is diagnostic negative evidence, not a general conclusion about
stochastic rounding.

| Profile | Uniform INT4 tail KL | Layer-0 INT8, rest INT4 | Reduction | Top-1 agreement |
|---|---:|---:|---:|---:|
| Retrieval/calibration | 5.9433 | 1.2018 | 79.78% | 0.4375 -> 0.65625 |
| Code/development | 5.2389 | 1.9824 | 62.16% | 0.3750 -> 0.65625 |

The plan averages 4.2222 payload bits per recurrent-state element. It is only
modeled storage; the simulator keeps dequantized FP32 tensors.

Canonical evidence hashes:

- Static nearest baselines: `95ba801eca4af37fac1a5796715dfcb86735b166aea4811a500fa9212c1bdf9d`
- INT8 stochastic baseline: `d6d199247a972f8f5a38456cdf46d23fa90eda58f5249318c1a3417b6240790f`
- Retrieval: `b2b6ce158f98157e37237133c093d15a4fa97a7b93f02e4ef39a90752a052eb0`
- Code: `d59e653de4ee9701540dce30045c9aaf40260c3074cafc62322e5c35a72d9fb2`

## Signal result

- Mean beta, forgetting, update norm, and committed residual were weak or
  inconsistent predictors of layer sensitivity.
- Query-weighted INT4 read-relative error ranked layer 0 highest on both traces.
- Its Spearman correlation with measured tail-KL improvement was 0.4592 on
  retrieval and 0.5046 on code, versus weaker raw state-error correlation.
- The complete negative-to-pivot record is preserved in
  [Experiment 001](EXPERIMENT_001_SIGNAL_PIVOT.md).

## Untouched confirmation

Candidate v0.1 was committed before evaluation, then run once on the untouched
multilingual profile without a layer sweep.

| Measure | Uniform INT4 | Layer 0 INT8, rest INT4 | Change |
|---|---:|---:|---:|
| Worst-5% token KL | 6.13459 | 1.36093 | -77.82% |
| Mean token KL | 2.85262 | 0.43382 | -84.79% |
| Top-1 agreement | 0.25000 | 0.59375 | +0.34375 |

The deterministic repeat matched canonical evidence hash
`1961ecc395d138cf505a20e55c3465260dcfded8b1ac3a9c95c40af775d2d722`.
Gate C passed. See [Confirmation 001](CONFIRMATION_001.md).

## Historical next action

Move from synthetic diagnostics to a preregistered public-data evaluation with
longer horizons, multiple seeds, sequence NLL, equal-byte baselines, and no
confirmation-set policy tuning.
