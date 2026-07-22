# Confirmation 001: frozen read-risk candidate

Date: 2026-07-22

Status: Gate C passed

## Frozen before evaluation

Candidate v0.1 was committed as `a510dc9` before the confirmation profile was
run. The fixed policy was:

- layer 0 at INT8;
- the other 17 Gated DeltaNet states at INT4;
- group size 128 with FP16 scale overhead;
- nearest rounding; and
- 32 prefill plus 32 teacher-forced decode tokens.

The multilingual profile had not been used for a layer sweep. Its artifact
records `sensitivity_sweep: null`.

## Result

| Gate C measure | Uniform INT4 | Frozen mixed plan | Outcome |
|---|---:|---:|---|
| Worst-5% token KL | 6.13459 | 1.36093 | 77.82% lower; pass |
| Mean token KL | 2.85262 | 0.43382 | 84.79% lower; pass |
| Top-1 agreement | 0.25000 | 0.59375 | Higher; pass |
| Finite metrics | Yes | Yes | Pass |

Estimated packed recurrent-state storage, including FP16 scales, was 2,564,096
bytes for the mixed plan versus 18,874,368 bytes for FP32 state, a modeled
7.36x ratio. This is an estimate only: QDQ emulation retains FP32 tensors and
does not realize that memory reduction.

The deterministic repeat produced the same canonical evidence hash:

```text
1961ecc395d138cf505a20e55c3465260dcfded8b1ac3a9c95c40af775d2d722
```

The full confirmation artifact is
[`evidence/qwen35-read-risk-v01-multilingual-confirmation.json`](../evidence/qwen35-read-risk-v01-multilingual-confirmation.json).

## Interpretation

The original mean-gate, forgetting, update-norm, and residual-magnitude
selectors remain rejected. The query-weighted read-risk replacement selected
the same layer that helped on calibration and development, then passed the
untouched diagnostic profile. That fixes the immediate selector failure for
this three-trace pilot.

It does not establish a generally useful or novel quantization method. The
traces are short and synthetic, only one model revision was tested, the
selection budget contains one high-precision layer, and there is no packed
kernel. Public tasks, longer horizons, multiple seeds and checkpoints,
equal-byte baselines, and a wider prior-art audit are required before stronger
claims.

## Registered follow-ups

1. Measure layer interventions using sequence NLL and longer-horizon state
   drift on public data.
2. Test whether a recurrent-state Jacobian or empirical-Fisher approximation
   predicts those interventions more cheaply than exhaustive sweeps.
3. Compare group sizes and static outlier preservation at equal total bytes.
4. Consider causal token-wise allocation only after the static policy is tested
   across datasets; consider a learned controller only with separate train and
   held-out splits.
