# MBPP confirmation report (v0.2)

Date completed: 2026-07-22

Protocol: [`PUBLIC_EVAL_PROTOCOL_V02.md`](PUBLIC_EVAL_PROTOCOL_V02.md)

Result artifact: [`../evidence/mbpp-v02-confirmation.json`](../evidence/mbpp-v02-confirmation.json)

## Verdict

The frozen v0.2 quality hypothesis passed on the untouched MBPP test split.
All preregistered validity, fidelity, equal-byte, and exact-storage gates passed,
including the confirmation-only requirement that the primary-versus-uniform
paired interval remain above zero.

This is a scoped held-out result for one pinned model, dataset construction,
and teacher-forced metric. It is not a breakthrough, state-of-the-art,
first-method, generated-code-correctness, speed, whole-model-memory, or
cross-model claim.

## Frozen scope

- Model: `Qwen/Qwen3.5-0.8B-Base` at
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Evaluated source commit:
  `6bd5bed2b61e192526ba8fdbec8232801cbea843`, with a clean tracked worktree.
- Data: MBPP test task IDs 11 through 510, exactly 500 tasks.
- Scored tokens: 30,244 teacher-forced reference-code tokens.
- Frozen primary: recurrent model layer 0 at INT8, the other 17 Gated
  DeltaNet layers at INT4, group size 128, FP16 scales, nearest rounding.
- Baselines: uniform INT4, uniform INT8, three equal-byte random layer
  placements, the MSE-selected placement, and three stochastic-rounding seeds.

Candidate-generated tokens were not fed back, executed, or graded.

## Primary result

| Measure | Uniform INT4 | Frozen mixed layout | Direction |
|---|---:|---:|---|
| Task-macro excess NLL, nats/token | 2.949743 | 0.803713 | 72.75% lower |
| Token-weighted mean KL | 3.149969 | 0.914580 | lower |
| Token-weighted worst-5% KL | 9.002207 | 4.839139 | lower |
| Token-weighted top-1 agreement | 0.321155 | 0.665190 | higher |

The paired primary-versus-uniform task-macro improvement was 2.146030
nats/token. Its 10,000-sample paired bootstrap 95% interval was
`[2.092249, 2.199866]`, entirely above zero.

Against the mean of the three same-byte random high-precision layer
placements, the paired improvement was 2.033167 nats/token with a 95% interval
of `[1.980220, 2.086136]`, also entirely above zero.

## Physical storage

| Layout | Resident recurrent-state bytes | FP32-state equivalent ratio |
|---|---:|---:|
| FP32 reference | 18,874,368 | 1.000x |
| Uniform INT4 plus FP16 scales | 2,433,024 | 7.758x smaller |
| Layer 0 INT8, rest INT4, plus FP16 scales | 2,564,096 | 7.361x smaller |
| Uniform INT8 plus FP16 scales | 4,792,320 | 3.938x smaller |

Every candidate matched its prespecified resident-byte count. Packed-cache and
QDQ preflight metrics matched exactly for every candidate; the measured maximum
absolute difference was 0.0 against a frozen tolerance of `1e-6`.

These values cover persistent recurrent-state payloads and scales only. The
Python path still materializes one 1,048,576-byte recurrent state while its
layer executes. The result does not measure model weights, ordinary KV caches,
allocator overhead, peak CUDA memory, latency, or throughput.

## Interpretation and negative controls

- The confirmation reduction, 72.75%, is close to the 74.14% development
  reduction without changing the frozen layer placement.
- The MSE selector also chose layer 0, so it is the exact same candidate and
  not an independent win for the read-risk selector.
- All three equal-byte random placements were substantially worse than the
  frozen layer-0 layout.
- Stochastic rounding was substantially worse across all three frozen seeds on
  this protocol. That is a negative result for these settings, not a general
  statement that stochastic rounding is ineffective.
- Uniform INT8 remained much closer to the FP32-state reference, but used
  4,792,320 resident bytes rather than the mixed layout's 2,564,096 bytes.

The evidence therefore supports a narrow result: on this pinned protocol, a
prespecified one-layer INT8 allocation preserved teacher-forced fidelity much
better than uniform INT4 and same-byte random placements while physically
storing the persistent recurrent state in 2,564,096 bytes.

## Integrity anchors

- Final artifact file SHA256:
  `70394c419298fc872cdd08e8aec12d17d5a56aa20f7d3c9f09fe8fdbf26c6ba9`
- Final canonical evidence SHA256:
  `2a652df92f99fa81f785244d966829e909d31f200e5a1520b76e6b46fb45d3e0`
- Final checkpoint file SHA256:
  `df0040cc9cebdbc442992e75d19f9090456f9b249da062c095840e731b6c4609`
- Final checkpoint canonical-state SHA256:
  `1293d93cb620d2193e9251f49c05d0bdaeebde16d3515c1f0e021c96b5d4fe1c`
- Frozen run-signature SHA256:
  `5d15268224357bb078315ef2c2b6e710a7eb8a2734df2527f9637b382951c78a`
- Prepared-manifest file SHA256:
  `c6a7d0db6ef7577a66ac19fbbc0be166279488f6a6be432b364bd9eb6833f7b0`
- Prepared-manifest canonical evidence SHA256:
  `21a6d18c6a0887b1499d156a3d610d4bfafdd59d3557713485b62038e263b96a`

`recurquant verify-confirmation` independently reconstructed the per-task and
token summaries from the final checkpoint arrays, re-ran every gate, verified
the source and manifest anchors, and returned `result: pass`,
`checkpoint_verified: true`, and `outcome_verified: true` with no warnings or
errors.

## Infrastructure record

The evaluator wrote an atomic checkpoint after every task. The long run was
resumed only after infrastructure interruptions, with the same command,
frozen source commit, prepared manifest, calibration artifact, and candidate
plan. One attempted resume stalled before evaluation because the streamed
dataset loader lacked network access; it was stopped before the checkpoint
changed. A later process exited during an atomic checkpoint replacement after
370 accepted tasks, leaving the previous checkpoint intact and a separate
temporary file. The final run resumed from the accepted 370-task checkpoint
and recomputed the interrupted next task once.

No partial candidate metric was inspected, no policy or gate was changed, and
no unfavorable result was rerun. Outcomes were opened only after all 500 tasks
completed and the final artifact existed.
