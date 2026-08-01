# Experiment 012 Stage-A result: StateLease-H5 survives falsification screen

> **Status: all frozen Stage-A checks passed on the one-task falsification screen.**
>
> StateLease-H5 beat the strongest fixed-replay schedule by `17.90%` excess NLL
> under the same byte contract on MBPP task `666`. It did not beat the two
> strongest equal-total-byte no-replay codecs. This is a one-task screening
> result, not development evidence, held-out confirmation, or a general
> advantage claim.

Date recorded: 2026-07-31

The immutable result and its frozen identity are documented in:

- [`../evidence/experiment012-statelease-stage-a-666.json`](../evidence/experiment012-statelease-stage-a-666.json)
- `research/EXPERIMENT_012_STATELEASE_PROTOCOL.md`
- `research/EXPERIMENT_012_STAGE_A_IDENTITY.md`

I also keep the full attempt history and pre-existing administrative records from
Experiments 010 and 011 in:

- `evidence/experiment010-statelease-stage-a-administrative-null.json`
- `evidence/experiment011-statelease-stage-a-administrative-null.json`

## Authenticated artifact

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment012-statelease-stage-a-666.json` |
| Artifact kind | `recurquant_experiment012_statelease_stage_a_falsification` |
| Clean H0 commit | `c3999c8ff7cc25b02a70da98b0d8faba388d3319` |
| One-run seal commit | `eeeab4b8d5962066e225ea856e83a5ccc24b7dfb` |
| Canonical evidence SHA-256 | `d4bd2c89bb265e5e1dab81a7bf89d97e71fa4a6822bef5d56142428e16b897c1` |
| Output file SHA-256 | `1e92b0bea176154496c7d5e45013bf051ef3f388352c1267d86910f81844fd22` |
| Output created at (UTC) | `2026-07-31T01:05:28.131816+00:00` |
| One-run marker | `RecurQuant-One-Run: experiment012-stage-a-task666-v1` |
| Model | `Qwen/Qwen3.5-0.8B-Base` revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Device and dtype | CUDA, bf16 (`torch.bfloat16`) |
| Task identity | MBPP task `666`, 69 prompt tokens, 39 code tokens, 38 scored tokens |
| Task-row SHA-256 | `b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9` |
| Stage-0 artifact | `artifacts/experiment012_stage0_production.pt` |
| Stage-0 artifact SHA-256 | `b6d40f126b9fca7578f3dd36d3bf26deeb20e81d270d3b1114dc9e27fd4a3551` |
| Forward passes | 429 |

The package verifier recomputes the stored per-token summaries, each method's
38-write-by-18-layer trajectory summary, physical tensor-schema totals, and
every gate field without importing the experiment runner:

```text
recurquant verify-statelease-stage-a evidence/experiment012-statelease-stage-a-666.json
```

## Frozen method set and screen rule

StateLease-H5 was evaluated as:

- equal-byte no-replay: `expanded_rht_q4_q8`, `rht_q4_q6_q8`, `rht_residual_q4`
- fixed-replay: `fixed_cc1`, `fixed_cc2`, `fixed_cc4`, `fixed_cc5`, `fixed_cut4_in5`
- historical anchor: `rht_cqer32`
- state lease candidate: `statelease_h5`

The one-shot gate required every method to be prespecified and storage-audited.
It required StateLease to beat `fixed_cc1` and remain within 5% of the strongest
fixed-replay schedule. It did not require StateLease to beat the no-replay
codecs; those become mandatory comparators at Stage B.

## Frozen one-task screen result

| Method | Excess NLL (38 tokens) | Top-1 agreement | Mean KL | CVaR95 KL | Max KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rht_q4_q6_q8` | `-0.000014` | `1.000000` | `0.005318` | `0.024342` | `0.026786` |
| `expanded_rht_q4_q8` | `0.002461` | `1.000000` | `0.009735` | `0.042403` | `0.061300` |
| `statelease_h5` | `0.023349` | `0.947368` | `0.027658` | `0.193988` | `0.267577` |
| `fixed_cut4_in5` | `0.028442` | `0.947368` | `0.030580` | `0.225663` | `0.323064` |
| `rht_residual_q4` | `0.034794` | `0.973684` | `0.012522` | `0.083783` | `0.090904` |
| `fixed_cc5` | `0.056124` | `0.947368` | `0.021947` | `0.076099` | `0.076178` |
| `fixed_cc4` | `0.093655` | `0.947368` | `0.077519` | `0.833122` | `0.960356` |
| `fixed_cc1` | `0.136328` | `0.868421` | `0.154371` | `1.022590` | `1.397867` |
| `rht_cqer32` | `0.136328` | `0.868421` | `0.154371` | `1.022590` | `1.397867` |
| `fixed_cc2` | `0.147195` | `0.947368` | `0.059956` | `0.392832` | `0.505890` |

StateLease-H5 was `17.90%` lower on excess NLL than the strongest fixed-replay
schedule in the gate definition and `82.87%` lower than `fixed_cc1`. It was
worse than both `rht_q4_q6_q8` and `expanded_rht_q4_q8` at the same total byte
budget. One task is too small for uncertainty or a general ranking.

![StateLease-H5 one-task excess NLL comparison](../assets/experiment012-stage-a-excess-nll.svg)

## Exact physical contract

StateLease-H5 and every fixed-replay comparator share the same allocated budget:

- 3,454,664 allocated persistent bytes in total
- 18,874,368 full FP32 recurrent-state bytes as the exact reference
- 2,564,096-byte physically packed checkpoint
- 2,485,760 q4/q8 payload bytes
- 73,728 FP16 scales bytes
- 4,608 precision-mask bytes
- 147,456 FP32 query-EMA selector bytes
- 743,112 allocated replay-buffer/count bytes; 289,032 bytes occupied at the
  end of this run
- 5.857109917534722 bits per state element

## Integrity and gate checks

All eight frozen Stage-A checks passed:

- all primary values were finite;
- exact allocation:
  `3454664` candidate persistent bytes vs `3454664` expected
- `fixed_cc1` improvement threshold:
  StateLease excess NLL was `0.112978` lower than `fixed_cc1`, or `82.87%`
- top-1 agreement:
  StateLease `0.947368` tied with best fixed comparator (`fixed_cc2`) and
  trail `0.0` with a `0.01` ceiling
- strongest-fixed disadvantage:
  relative disadvantage `-0.1790` (statelease better)
- trajectory-nmse:
  StateLease AUC `0.019805` < `fixed_cc1` AUC `0.050699`
- replay actions stayed within the frozen C4/C5 controller set, with ties
  deterministically assigned to C5; and
- the Stage-0 checkpoint, one-run seal, output artifact, diagnostics, and
  update-evidence integrity checks all passed.

The artifact was published in one monotonic two-phase step and no rerun occurred.

## Decision

This is a successful one-task falsification screen against the prespecified
fixed-replay gate. The stronger no-replay results make practical advantage an
open question. Any Stage-B run needs a new committed three-workload identity and
a genuine StateLease evaluator; Experiment 009 cannot be relabelled for that
purpose.

## Limitations

- This is one public MBPP calibration-task reference-code trace (38 scored
  teacher-forced tokens).
- The Stage-A pass does not establish cross-window, multi-seed, cross-code,
  latency, or speed claims.
- The correctness-first path still materializes floating-point workspaces and
  has no end-to-end fused recurrent kernel or speed result.
