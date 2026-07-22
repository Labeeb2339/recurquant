# Experiment 005 result: gate failure and same-calibration postmortem

> **Status: permanently failed before holdout.**
>
> The frozen storage-boundary validator achieved 13/16 sign checks
> (`0.8125`) against a required minimum of `0.95`. Experiment 005 therefore
> did not open its ranked `[8, 16)` heldout-calibration window. Later results
> on the already inspected selector partition are postmortem diagnostics only;
> they do not reopen the gate or support an improvement, generalization,
> novelty, or breakthrough claim.

Date recorded: 2026-07-23

## Frozen gate evidence

The permanent gate decision is backed by:

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment005-storage-boundary-599862e.json` |
| Artifact kind | `recurquant_storage_boundary_taylor_diagnostic` |
| Clean implementation commit | `599862eef3b635f14b05c578f02393a0abd072a6` |
| File SHA-256 | `61a2936bd20679bad441921d26b556f5986eec61449c4a9743c9b0b5e0bea86d` |
| Canonical evidence SHA-256 | `b168330b4c39963b7c149230d4b1ad9fa57b20b02d6d95ee583ee9941f68b19f` |
| Created at | `2026-07-22T17:07:41.231231+00:00` |
| Model arithmetic | FP32 |
| Hardware | NVIDIA RTX 5070 Laptop GPU |
| PyTorch | `2.11.0+cu128` |
| Transformers | `5.14.1` |
| Evidence verification | valid, with no verifier errors |

The artifact records `passed: false`. Its only failed conjunction member is
central-difference sign agreement:

| Check | Frozen threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Repeated baseline maximum difference | at most `1e-7` | `0.0` | pass |
| Informative rows | at least 3 of 4 | 4 of 4 | pass |
| Central-difference sign agreement | at least `0.95` | `13/16 = 0.8125` | **fail** |
| Median absolute relative derivative error | at most `0.10` | `0.03400520381314055` | pass |
| Convergent informative rows | at least `0.75` | `0.75` | pass |
| Near-zero-row absolute-error checks | at most `2e-7` | no near-zero rows | pass |

The overall gate is a conjunction, so one failed condition is a failed gate.
The threshold, row strata, epsilon grid, arithmetic mode, and pass rule remain
as frozen. We will not weaken them, delete the failed row, or rerun a modified
validator and label it the Experiment 005 result.

## Failure localization

All three sign misses came from the fixed `early_low` row. The other three
rows produced 12/12 correct sign checks. For `early_low`, the autograd
directional derivative was

```text
+1.534285755042659e-07
```

at a base target NLL of `0.03833549842238426`. One FP32 scalar ULP at that loss
is `3.725290298461914e-09`. Across epsilon values `1/4`, `1/8`, `1/16`, and
`1/32`, the expected central-difference numerators were approximately 20.59,
10.30, 5.15, and 2.57 ULPs. The observed numerator counts were 0, 31, 0, and 0
ULPs, yielding sign flags `false`, `true`, `false`, and `false`. The observed
endpoint benefit for this row was zero.

This pattern is **consistent with finite-precision loss resolution** at that
row and transition. It does not prove that resolution was the sole cause, and
it does not turn the failed gate into a pass. More importantly, this validator
tested the signed local derivative

```text
g dot (Q8(raw) - Q4(raw)),
```

not the actual squared endpoint score used to form the target-directional
Fisher quotas:

```text
(g dot e4)^2 - (g dot e8)^2.
```

Future candidates must validate their own score algebra and packing directly.
They must not inherit this failed signed-derivative check as evidence for a
different formula.

## Holdout remained unopened

Experiment 005 froze the following first generalization window:

```text
phase: calibration
ranked offset: 8
task count: 8
window: [8, 16)
```

The gate failed before tokenization or model loading for that window. No
Experiment 005 heldout artifact was produced, and the ranked `[8, 16)` rows
remain unchanged and unopened. Any later protocol that proposes to use this
window must authenticate that fact before access; it may not silently substitute
a different window after seeing data.

## Same-calibration postmortem

After the failure, we ran one descriptive quality diagnostic on the already
inspected selector partition. Its evidence is:

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment005-adaptive-same-calibration-8task-556e527.json` |
| Artifact kind | `recurquant_adaptive_row_packing_same_calibration_quality_diagnostic` |
| Clean implementation commit | `556e527b75f73d557e2bddf4a9973ce945cf1ab9` |
| File SHA-256 | `3495698932b43d93f387bb61492f91fc38840097020980977451be2042a02164` |
| Canonical evidence SHA-256 | `f07a6b852a4c427c0b9946ad44c21d023299d173eb24695462e99e3785061d61` |
| Created at | `2026-07-22T17:40:53.827148+00:00` |
| Selection | selector-task prefix, offset 0, limit 8, stop 8 |
| Aligned scored tokens | 642 |
| Primary method | `adaptive_mse_target_directional_fisher_quota` |
| Holdout applicability | `false` |
| Holdout pass value | `null` |

The filename begins with `experiment006` because it was generated during later
development, but its primary method is the plain Experiment 005 adaptive-MSE
candidate. It contains no rank fusion and cannot be treated as Experiment 006
evidence. The eight task IDs were `945`, `794`, `657`, `702`, `651`, `720`,
`903`, and `918`; all belong to the selector partition already used for method
development.

The task-macro results were:

| Method | Excess next-token NLL |
| --- | ---: |
| `adaptive_mse_target_directional_fisher_quota` | `0.4933023080229759` |
| `target_directional_fisher_difference_int4` | `0.5357813686132431` |
| `target_diagonal_fisher_difference_int4` | `0.5480052754282951` |
| `delta_direction_magnitude_int4` | `0.5449846908450127` |
| `hrr_h1` | `0.6301188990473747` |
| `hrr_h32` | `0.7533748596906662` |
| `adaptive_mse_hrr_h1_quota` | `0.7794309854507446` |
| `row_mse` | `0.7946401834487915` |
| `v02_layer0_static` | `0.8614744991064072` |
| `uniform_int4` | `2.6861148476600647` |
| `random_rows_s1101` | `2.729534089565277` |

For the primary, mean KL was `0.4791950900107622`, worst-token KL CVaR95 was
`2.902877390384674`, and top-1 agreement was `0.7851893156766891`.

Using 10,000 paired bootstrap resamples with seed 2339, comparator-minus-primary
contrasts included:

| Comparator | Mean improvement | Paired 95% interval |
| --- | ---: | ---: |
| `adaptive_mse_hrr_h1_quota` | `0.2861286774277687` | `[0.1524614840745926, 0.44845127817243335]` |
| `target_directional_fisher_difference_int4` | `0.04247906059026718` | `[-0.026418810337781883, 0.12283947486430402]` |
| `target_diagonal_fisher_difference_int4` | `0.054702967405319214` | `[-0.025941481441259337, 0.14100585989654058]` |
| `delta_direction_magnitude_int4` | `0.05168238282203674` | `[-0.05204220674932003, 0.14878664575517173]` |
| `hrr_h1` | `0.1368165910243988` | `[0.0874815434217453, 0.1885298192501068]` |

The strongest equal-byte static comparator by excess NLL was
`target_directional_fisher_difference_int4`. The descriptive relative
reduction against it was

```text
(0.5357813686132431 - 0.4933023080229759)
/ 0.5357813686132431
= 0.0792843183409219
```

or approximately 7.93%. That is below the frozen 20% requirement, and the
paired interval against this comparator crosses zero. The postmortem therefore
would not satisfy the frozen quality gate even if it had been held out. Its
positive interval against the H1-quota adaptive ablation is useful for forming
a new hypothesis, but it is not evidence of generalization.

## Related selector artifacts

Two selector artifacts were also generated at commit
`556e527b75f73d557e2bddf4a9973ce945cf1ab9`:

| Artifact | File SHA-256 | Canonical evidence SHA-256 |
| --- | --- | --- |
| `artifacts/experiment006-hrr-selector-8task-556e527.json` | `fa02e1d468ecc13c78b7cf8e63f237e372c556d9fed0c1f4b47c9dd901a808dd` | `07e646ccb9b1df5ff9873a94f7bacb07d7a4e2b70136e3a68f40d1619814d899` |
| `artifacts/experiment006-loss-selector-8task-556e527.json` | `33bdc5939429281ba5377eeb02d59fac72a0f8da657c713bb7854d235e2fb057` | `ae92af38475720eb1ce19527f1c2de3d0d1fc045a1160a83f8da30ecde282214` |

These are authenticated exploratory inputs, not heldout results. They predate
the Experiment 006 fusion freeze and cannot satisfy a later protocol's
implementation, score-validation, or provenance prerequisites without
regeneration from that protocol's clean committed implementation.

## Result and claim boundary

The durable findings are limited to:

1. the frozen Experiment 005 numerical gate failed at `0.8125 < 0.95`;
2. the failure was localized to three checks on one low-signal row and is
   consistent with, but not proven to be caused by, FP32 loss resolution;
3. the ranked `[8, 16)` holdout was not opened;
4. plain adaptive MSE was promising on the same eight tasks used to construct
   its quotas, but did not clear the frozen effect-size or strongest-static
   uncertainty requirements; and
5. any successor is a new candidate with a new artifact kind and an independent
   validation gate.

Experiment 005 makes no breakthrough claim. It remains below the
heldout-calibration rung of the claim ladder.
