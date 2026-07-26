# Experiment 008 result: CORA-C2 rejected before holdout

> **Status: development gate failed; protected holdout unopened.**
>
> CORA-C2 beat static target-Fisher by `26.17%` and adaptive MSE by
> `21.62%` on the frozen 16-task development window. It did not beat CQER-32:
> macro excess NLL was `0.391586` for CORA-C2 and `0.354250` for CQER-32.
> Confirmation-2 reduced normalized committed-mask churn by `79.99%`, but it
> worsened raw-CORA NLL by `6.96%`, above the frozen `1%` limit. Five
> preregistered checks failed. Ranked MBPP window `[8, 16)` remains unopened.

Date recorded: 2026-07-23

## Authenticated diagnostic

The development identity, implementation, six methods, storage contract, and
13-condition advancement rule were frozen before this run. The run used the
separate ranked MBPP calibration window `[16, 32)`.

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment008-cora-c2-development-16task-69eec866.json` |
| Artifact kind | `recurquant_cora_c2_development_quality_diagnostic` |
| Clean implementation commit | `69eec866b90f2ae5386da23a8f34fba4c428d9e1` |
| File SHA-256 | `03da6dc27641e7816f6f03bd059876322974530911d28e1d9719dd992af8f9db` |
| Canonical evidence SHA-256 | `24877f73068f765ef0bae6b2bee54ff95d5204977b3c08dfe95516d651c2fb70` |
| Created at | `2026-07-22T20:51:42.559117+00:00` |
| Model | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Device and dtype | CUDA, bfloat16 |
| Task IDs | `666`, `795`, `944`, `653`, `857`, `884`, `878`, `822`, `687`, `820`, `920`, `771`, `869`, `851`, `728`, `704` |
| Aligned scored tokens | 798 |
| Bootstrap | 10,000 paired task resamples, seed 2339 |
| Repository and source files | clean and stable at start and end |
| Evidence verification | valid, with no verifier errors |

Independent rechecks reproduced every paired bootstrap contrast, source and
selector hash, task and token identity, byte count, quota, and transition
handshake. The protected `[8, 16)` loader guards also passed after the run.
This is an authenticated method failure, not an execution failure.

## Frozen-primary result

| Method | Role | Macro excess NLL | Mean KL | CVaR95 KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| `target_directional_fisher_difference_int4` | static Fisher | `0.5303981081` | `0.5286498712` | `3.3478351496` | `0.7586527802` |
| `adaptive_mse_target_directional_fisher_quota` | per-write row MSE | `0.4996191338` | `0.4905063435` | `3.0781028867` | `0.7698915936` |
| `query_ema32_weighted_mse_target_fisher_quota` | CQER-32 | **`0.3542503491`** | **`0.3641314693`** | **`2.4055756629`** | `0.8027831651` |
| `causal_observability_mse_target_fisher_quota` | raw CORA | `0.3661060743` | `0.3896415923` | `2.6112778820` | `0.8017726764` |
| `query_ema32_confirm2_mse_target_fisher_quota` | CQER-32+C2 | `0.3951669149` | `0.4165541902` | `2.7546808161` | **`0.8111176938`** |
| `causal_observability_confirm2_mse_target_fisher_quota` | frozen primary | `0.3915858567` | `0.4186489517` | `2.8813917115` | `0.8103738017` |

Comparator-minus-primary paired NLL contrasts were:

| Comparator | Mean improvement | Paired 95% interval | Interpretation |
| --- | ---: | ---: | --- |
| Static target-Fisher | `0.1388122514` | `[0.0436136126, 0.2466757520]` | 26.17% reduction; pass |
| Adaptive MSE | `0.1080332771` | `[0.0277902825, 0.1880076991]` | 21.62% reduction; descriptive pass |
| CQER-32 | `-0.0373355076` | `[-0.1060209079, 0.0192620480]` | 10.54% worse; fail |
| Raw CORA | `-0.0254797824` | `[-0.0689048139, 0.0198680538]` | C2 was 6.96% worse; fail |

CORA's transition-aware point estimate was also `3.35%` worse than CQER-32,
so the observability recurrence did not improve the strongest causal selector.
The uncertainty interval for raw CORA versus CORA-C2 crossed zero, but the
frozen advancement rule used the prespecified relative-worsening threshold as
well as the paired comparisons.

## Confirmation-2 result

Confirmation-2 behaved as a strong debouncer:

| Selector | Normalized committed churn |
| --- | ---: |
| Raw CORA | `0.0129384697` |
| CORA-C2 | `0.0025893428` |
| Relative reduction | **`79.9873%`** |

That stability was not free. CORA-C2 raised NLL from `0.366106` to `0.391586`.
The same pattern appeared in the CQER ablation: C2 raised NLL from `0.354250`
to `0.395167`. C2 improved top-1 agreement for both selectors, but its NLL and
tail-KL cost rules it out as the frozen advancement mechanism. This experiment
supports treating two-hit admission as a switching-quality trade-off, not as a
generally beneficial selector.

## Gate decision

| Frozen check | Observed | Result |
| --- | ---: | --- |
| Lower NLL than static, adaptive, and CQER | CQER `0.354250` < primary `0.391586` | **fail** |
| NLL reduction vs static | 26.17%, minimum 20% | pass |
| NLL reduction vs adaptive | 21.62%, minimum 5% | pass |
| NLL reduction vs CQER | -10.54%, minimum 5% | **fail** |
| Paired lower bound vs static | `0.043614 > 0` | pass |
| Paired lower bound vs CQER | `-0.106021 <= 0` | **fail** |
| Top-1 margin vs static/adaptive | primary was higher | pass |
| Top-1 not lower than CQER | `0.810374 >= 0.802783` | pass |
| CVaR95 margin vs static/adaptive | primary was lower | pass |
| Raw CORA NLL reduction vs CQER | -3.35%, minimum 3% | **fail** |
| C2 churn reduction vs raw CORA | 79.99%, minimum 50% | pass |
| C2 top-1 not lower than raw CORA | `0.810374 >= 0.801773` | pass |
| C2 NLL worsening vs raw CORA | 6.96%, maximum 1% | **fail** |
| Exact bytes, quotas, handshakes, finiteness, identity | all exact | pass |

The packed state used `2,564,096` bytes. CORA-C2 used another `152,064`
resident bytes for its FP32 observability diagonal and bit-packed prior raw
masks, for `2,716,160` bytes including selector state. This is not an
equal-total-memory, peak-memory, latency, throughput, or fused-kernel result.

## Interpretation and decision

The negative result isolates two useful facts:

1. The frozen diagonal causal-observability proxy was not better than the much
   simpler query-energy EMA on this development window.
2. Requiring two consecutive raw-mask hits removes most switching but is too
   sticky for the quality objective at this token scale.

Experiment 008 stops here. The independent numerical verifier is not built,
the protected `[8, 16)` window is not opened, and no formula, threshold, or
ablation will be tuned and relabelled as Experiment 008.

The underlying observability-weighted-error idea also has classical finite-
wordlength state-space precedent. CORA is best understood as an experimental
Gated DeltaNet adaptation of that principle, not a newly invented principle.
Any successor must use a new protocol and development identity, replace or
substantially strengthen the failed signal, and compare against CQER-32.

Experiment 008 makes no novelty, state-of-the-art, deployment, or breakthrough
claim.
