# Experiment 007 result: CQER-32 rejected before holdout

> **Status: development gate failed; holdout unopened.**
>
> CQER-32 lowered macro excess NLL relative to both frozen components on the
> already inspected eight-task partition, but it failed two preregistered
> requirements. Its reduction against the static target-Fisher comparator was
> `13.62%`, below the required `20%`, and its top-1 agreement disadvantage was
> `0.02690`, above the allowed `0.01`. The complete gate therefore failed.
> Ranked MBPP window `[8, 16)` remains unopened, and Experiment 007 supports no
> generalization, novelty, systems, state-of-the-art, or breakthrough claim.

Date recorded: 2026-07-23

## Authenticated diagnostic

This run used the same eight MBPP calibration tasks that informed the frozen
target-Fisher layer quotas. It can reject the candidate, but it cannot establish
generalization.

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment007-cqer32-same-calibration-8task-3a062771.json` |
| Artifact kind | `recurquant_cqer32_same_calibration_quality_diagnostic` |
| Clean implementation commit | `3a06277171684e6e6bf4aec1153c16c968fea0fe` |
| File SHA-256 | `6f1e6d22eafde9be7c231cc81dc04e7a127c8b14f47ec2865de9326a61040f32` |
| Canonical evidence SHA-256 | `2226dcc055fff3ebff4dadb9a9908166d361dfd2f61b32757d2eb9c650d1381a` |
| Created at | `2026-07-22T19:37:50.097105+00:00` |
| Model | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Device and dtype | CUDA, bfloat16 |
| Task IDs | `945`, `794`, `657`, `702`, `651`, `720`, `903`, `918` |
| Aligned scored tokens | 642 |
| Bootstrap | 10,000 paired resamples, seed 2339 |
| Repository and source files | clean and stable at start and end |
| Evidence verification | valid, with no verifier errors |

Every CQER write consumed exactly one causal query observation. Across all
tasks and layers, the state-update and observation counts matched. All metrics
and logits were finite, every layer realized its exact frozen quota, and the
implementation stored exactly 1,976 promoted rows.

## Frozen-primary result

CQER-32 scores each recurrent-state row with:

```text
32-token EMA(normalized query energy) * (INT4 row MSE - INT8 row MSE)
```

The per-layer target-Fisher quotas were frozen before the run. Only the row
identities selected inside each layer could change at a state write.

| Method | Role | Macro excess NLL | Mean KL | CVaR95 KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| `query_ema32_weighted_mse_target_fisher_quota` | frozen primary | `0.4627917185` | `0.4525271095` | `2.7080783099` | `0.7639170736` |
| `adaptive_mse_target_directional_fisher_quota` | error-only dynamic component | `0.4933023080` | `0.4791950900` | `2.9028773904` | `0.7851893157` |
| `target_directional_fisher_difference_int4` | frozen static component | `0.5357813686` | `0.5205607209` | `3.3614913225` | `0.7908198461` |

Comparator-minus-CQER paired NLL contrasts were:

| Comparator | Mean improvement | Paired 95% interval | Interpretation |
| --- | ---: | ---: | --- |
| Plain adaptive target-Fisher quota | `0.0305105895` | `[-0.0312529538, 0.0830206126]` | 6.18% descriptive reduction; interval crosses zero |
| Static target-directional Fisher | `0.0729896501` | `[-0.0110369474, 0.1781795248]` | 13.62% descriptive reduction; interval crosses zero |

CQER had the lowest mean KL and CVaR95 KL of the three methods, but its top-1
agreement was lower than both. The preregistered gate was conjunctive, so the
two failures cannot be offset by the passing metrics.

## Gate decision

| Frozen check | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Lower NLL than both components | yes | `0.46279 < 0.49330 < 0.53578` | pass |
| NLL reduction vs plain adaptive | at least 5% | 6.18% | pass |
| NLL reduction vs static Fisher | at least 20% | 13.62% | **fail** |
| Top-1 disadvantage vs better comparator | at most 0.01 | 0.02690 | **fail** |
| CVaR95 disadvantage | at most 0.10 | `-0.19480` | pass |
| Exact packed and selector bytes | exact | exact | pass |
| Exact quotas and stage/consume handshake | exact | exact | pass |
| Finite values and authenticated run | required | satisfied | pass |

The packed recurrent-state representation used `2,564,096` bytes. CQER's
persistent FP32 query-energy EMA used another `147,456` bytes, for
`2,711,552` resident bytes including selector state. This is not an equal-total-
memory result against selectors with no persistent auxiliary state, and it is
not a latency or peak-memory measurement.

## Decision

Experiment 007 stops here:

- the frozen development gate failed;
- the candidate-aligned FP64 prerequisite is not reached;
- ranked MBPP window `[8, 16)` remains unopened; and
- the 32-token half-life, formula, quotas, or thresholds will not be tuned and
  relabelled as Experiment 007.

The positive point estimates are useful engineering evidence, but they are not
statistically decisive on eight reused tasks and do not meet the frozen bar. A
successor must be a new preregistered experiment with a new development split
and explicit comparison to the nearest query-aware quantization prior art.

Experiment 007 makes no breakthrough claim.
