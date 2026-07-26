# Experiment 006 result: equal-rank fusion rejected before holdout

> **Status: stopped on the inspected selector partition; holdout unopened.**
>
> The frozen `lambda = 0.5` primary did not beat either of its two strongest
> component endpoints with a positive paired confidence bound. It was worse
> than plain adaptive MSE by point estimate, improved on the strongest static
> method by only 3.90%, and its intervals against both methods crossed zero.
> The predeclared `0.25` and `0.75` ablations were better, but the protocol
> forbids promoting an ablation after seeing results. Experiment 006 therefore
> stops without opening ranked MBPP window `[8, 16)` and without an
> improvement, generalization, novelty, systems, or breakthrough claim.

Date recorded: 2026-07-23

## Authenticated diagnostic

The quality run used the same eight MBPP calibration tasks that produced the
selector inputs. It is a candidate-rejection diagnostic, not heldout evidence.

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment006-rank-fusion-same-calibration-8task-c2ad68b.json` |
| Artifact kind | `recurquant_rank_fusion_same_calibration_quality_diagnostic` |
| Clean implementation commit | `c2ad68b22433b1b077df5aafaacc667e305b2294` |
| File SHA-256 | `9824f4db5a3b8eb7de392537768e658ba7d921cc52c1ba7714d8e8433c85dbbf` |
| Canonical evidence SHA-256 | `94699c298767d5d1f1e9f2ca61f766fe541871b90bb2f054c8b2aef9bd292827` |
| Created at | `2026-07-22T18:30:13.878345+00:00` |
| Model | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Device and dtype | CUDA, bfloat16 |
| Task IDs | `945`, `794`, `657`, `702`, `651`, `720`, `903`, `918` |
| Aligned scored tokens | 642 |
| Bootstrap | 10,000 paired resamples, seed 2339 |
| Repository clean and stable | true |
| Evidence verification | valid, with no verifier errors |

Every mixed candidate physically stored 1,976 INT8-promoted rows in exactly
2,564,096 resident recurrent-state bytes: 2,485,760 payload bytes, 73,728
scale bytes, and 4,608 mask bytes. This verifies equal storage for the quality
diagnostic; it is not a latency or whole-model-memory result.

## Frozen-primary result

The primary was fixed before this run as equal ordinal-rank fusion between:

1. offline target-directional-Fisher row sensitivity; and
2. causal per-write aligned INT4-to-INT8 reconstruction-MSE benefit.

Within each layer, the calibrated exact-byte plan fixed the promotion quota.
Only the promoted row identities could change.

| Method | Role | Macro excess NLL | Mean KL | CVaR95 KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| `rank_fusion_l050_target_fisher_adaptive_mse` | frozen primary | `0.5148733034729958` | `0.47354681603610516` | `3.0126770064234734` | `0.7846591621637344` |
| `adaptive_mse_target_directional_fisher_quota` | dynamic endpoint | `0.4933023080229759` | `0.4791950900107622` | `2.902877390384674` | `0.7851893156766891` |
| `target_directional_fisher_difference_int4` | strongest static endpoint | `0.5357813686132431` | `0.5205607209354639` | `3.361491322517395` | `0.7908198460936546` |
| `adaptive_mse_hrr_h1_quota` | H1-quota adaptive control | `0.7794309854507446` | `0.7939629852771759` | `3.3163906931877136` | `0.6900708600878716` |

Comparator-minus-primary paired contrasts were:

| Comparator | Mean improvement | Paired 95% interval | Interpretation |
| --- | ---: | ---: | --- |
| Plain adaptive target-Fisher quota | `-0.021570995450019836` | `[-0.09485602006316185, 0.04348200932145117]` | primary point estimate is worse; interval crosses zero |
| Static target-directional Fisher | `0.020908065140247345` | `[-0.05002020299434662, 0.12526661902666092]` | primary point estimate is better; interval crosses zero |
| H1-quota adaptive control | `0.26455768197774887` | `[0.09466566145420074, 0.4699110761284828]` | primary is better than this weaker control |

Against the strongest static method, the primary's descriptive excess-NLL
reduction was only 3.90%, below the frozen 20% threshold. Against plain
adaptive MSE, excess NLL increased by 4.37%. Conditions requiring a positive
paired lower bound against the strongest static and strongest individual
adaptive methods therefore would fail even before the unavailable
candidate-aligned numerical gate is considered.

## Predeclared ablations

The protocol required both intermediate weights to be reported but prohibited
either from replacing the primary after results were visible.

| Method | Macro excess NLL | Relative to strongest static | Relative to plain adaptive |
| --- | ---: | ---: | ---: |
| `rank_fusion_l025_target_fisher_adaptive_mse` | `0.45971328765153885` | 14.20% lower | 6.81% lower |
| `rank_fusion_l075_target_fisher_adaptive_mse` | `0.45596349984407425` | 14.90% lower | 7.57% lower |

Both ablations had lower excess NLL than the frozen primary on these same
eight tasks. In ablation-minus-primary terms, the paired evidence was strongest
for `lambda = 0.25`: its interval was
`[-0.1031366042792797, -0.01751011610031128]`, entirely below zero. The
`lambda = 0.75` interval was `[-0.11918644607067108,
0.010958641767501831]` and crossed zero.

These results show that equal ordinal rank mixing was not the best point on the
inspected data. They do not authorize selecting `0.25`, `0.75`, or another
weight and calling it prespecified. Any weight-based successor would need a new
protocol and new development data.

## Decision and next hypothesis

Experiment 006 is rejected as a route to the existing holdout:

- the frozen primary did not dominate either strong endpoint;
- the required candidate-aligned numerical/packing gate was not implemented;
- the ranked `[8, 16)` window was never tokenized or loaded; and
- the ablation pattern cannot rescue the primary.

The result also localizes the remaining weakness. Instantaneous reconstruction
MSE measures which rows are hard to store, while the static score measures
average calibrated loss sensitivity. Neither directly measures which state
rows the next recurrent read will use. A successor may test a causal predictor
of row read energy while retaining the same frozen per-layer quotas and exact
resident bytes. That is a new experiment, not a repair or relabeling of
Experiment 006.

Experiment 006 makes no breakthrough claim.
