# Research status

## v0.3 experimental track

Last updated: 2026-07-23

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

Any successor must use a new protocol and development split. The nearest known
mechanism-level comparison is MixKVQ's query-magnitude and quantization-
difficulty channel scoring for KV caches; CQER-32 cannot be described as the
first query-aware mixed-precision method.

No v0.3 improvement, novelty, speed, or breakthrough claim is supported.

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
