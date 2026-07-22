# MBPP v0.2 development decision

Status: all frozen Stage 1 gates passed; the v0.2 policy remains unchanged and
untouched confirmation is authorized

Decision recorded: 2026-07-22

## Frozen run

- Phase: MBPP `validation`, task IDs 511-600 (all 90 tasks).
- Scored target: 5,524 reference-code tokens under teacher forcing.
- Model: `Qwen/Qwen3.5-0.8B-Base` at
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Source commit: `20a5ea95a8ed692600ee1645d2913f3a4b8a6795`.
- Repository state recorded by the evaluator: tracked worktree clean.
- Device: NVIDIA GeForce RTX 5070 Laptop GPU; BF16 model weights; driver
  592.15.
- Command:

  ```powershell
  .\.venv\Scripts\python.exe scripts\evaluate_mbpp.py `
    --phase development `
    --calibration-artifact evidence\mbpp-v02-calibration.json `
    --prepared-manifest evidence\mbpp-v02-development-manifest.json `
    --local-files-only `
    --output artifacts\mbpp-v02-development.json
  ```

The run completed without resuming. Candidate-generated tokens were never fed
back, no generated code was executed, and the first code token was scored from
the prefill logits.

## Primary result

The frozen primary policy stores model layer 0 at INT8 and the other 17 Gated
DeltaNet recurrent states at INT4. Its exact resident state is 2,564,096 bytes,
or 7.361x smaller than the 18,874,368-byte FP32 recurrent-state reference.

| Method | Resident bytes | Macro excess NLL | Mean token KL | Worst-5% token KL | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| Uniform INT4 nearest | 2,433,024 | 2.96447 | 3.12032 | 9.12815 | 0.32694 |
| Layer 0 INT8, rest INT4 nearest | 2,564,096 | 0.76649 | 0.90949 | 4.91772 | 0.68338 |
| Uniform INT8 nearest | 4,792,320 | 0.01559 | 0.02615 | 0.26554 | 0.95873 |

Relative to uniform INT4, the primary policy reduced task-macro excess NLL by
74.14%. The paired task-bootstrap mean improvement was 2.19798 excess-NLL
points with a 95% percentile interval of `[2.08078, 2.32146]`.

At the identical 2,564,096-byte budget, the three prespecified random controls
promoted layers 18, 4, and 13. The primary policy's gain over their within-task
mean was 2.07838 excess-NLL points with a paired 95% interval of
`[1.96598, 2.19451]`.

Calibration MSE independently selected layer 0, so the MSE baseline and frozen
primary policy are identical in this version. This agreement is reported; it is
not a separate successful comparison.

The stochastic-rounding controls were substantially worse on this run: their
macro excess NLL values were 5.80807, 5.81640, and 5.82662 for seeds 2339,
2340, and 2341. This negative result is retained rather than omitted.

## Validity checks

- All seven preregistered continuation gates passed.
- All values were finite.
- Every candidate used the same 90 tasks and 5,524-token manifest.
- Every resident packed-byte total exactly matched its registered layout.
- Packed-cache and QDQ logits matched with maximum absolute difference `0.0`
  across all candidates in the preflight.
- The prepared-manifest, calibration-evidence, dataset-manifest, token-manifest,
  repository-commit, artifact, and canonical-evidence hashes were independently
  rechecked after the run.
- Confirmation data had not been evaluated when this decision was recorded.

Artifact integrity:

- Final artifact SHA256:
  `5980fd58aa0933ad97deb896d4901fcd37350c4a57d8a80022ab218aaf77e727`
- Canonical evidence SHA256:
  `301c52e194bbd23059a0040a8e94aeac97dc33de1100f13edbf17dc877755488`
- Development dataset manifest SHA256:
  `8fed3da0aae864f4e30c70ad70b0269f759d3592dccb9ab87f24444fa24d65dc`

The per-token manifest digest remains embedded in the canonical artifact and
was included in the independent provenance check.

## Decision and claim boundary

Proceed once to the already frozen 500-task MBPP confirmation split. Do not
retune the promoted layer, bit allocation, group size, rounding rule, prompts,
baselines, seeds, or gates after seeing this development result.

This is positive development evidence, not confirmation and not proof of a
breakthrough. It measures teacher-forced reference-code fidelity on one pinned
model. It does not establish generated-code correctness, generality across
models or datasets, total-model memory reduction, latency improvement, or
state-of-the-art performance. This quality run did not collect a controlled
latency result, so no speed claim is permitted.
