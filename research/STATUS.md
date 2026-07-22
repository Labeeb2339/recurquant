# Research status

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

Last updated: 2026-07-22

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
