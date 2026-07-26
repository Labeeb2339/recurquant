# Experiment 009 Stage-B result: RHT-CQER-32 advances

> **Status: all eight frozen Stage-B development gates passed.**
>
> On the authenticated 32-task MBPP development window, RHT-CQER-32 lowered
> task-macro aligned excess NLL by `52.73%` and aggregate local
> recurrent-state reconstruction SSE by `57.85%` relative to CQER-32 at the
> exact same packed-state and selector byte counts. This is development
> evidence for one pinned model and task window. It is not confirmation,
> novelty, a speed result, state-of-the-art evidence, or a breakthrough claim.

Date recorded: 2026-07-26

## Authenticated artifact

I preserved the complete raw result as a compressed GitHub release asset
instead of adding 168 MB of repeated state-error records to Git history. The
small committed
[release manifest](../evidence/experiment009-rht-cqer-stage-b-result-manifest.json)
binds the downloadable artifact, its checksums, the frozen identity, the
evaluation commit, and the verifier receipt.

| Field | Value |
| --- | --- |
| Artifact kind | `recurquant_rht_cqer32_stage_b_development` |
| Raw artifact | `experiment009-rht-cqer-stage-b-result-cdc603b.json` |
| Raw bytes | `167,987,192` |
| Raw file SHA-256 | `57b341d37871a52977b1ff89709864f3e6e0927154e5b2b9275b6f374953fe05` |
| Canonical evidence SHA-256 | `2b15c732e894510f0421a22fcca9435e035dd15c4d3b50e2fcb733c0d1df58a8` |
| Created at | `2026-07-26T05:46:49.671275+00:00` |
| Frozen evaluation commit | `8168c469b252bc9e707e51feaeccc3f940f190bb` |
| Corrected verifier commit | `2075154e642c39a14432adcc8ec32da679b534d3` |
| Model | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Development identity | ranked MBPP window `[32, 64)`, 32 tasks |
| Primary scored tokens | 1,956 aligned code tokens |

The artifact's repository commit, source hashes, runtime, model revision,
ordered task identity, token manifest, selector artifacts, row plan, sign
schedule, methods, thresholds, storage contract, per-task results, raw state
records, aggregate metrics, bootstrap interval, integrity decision, and eight
advancement checks all passed strict semantic validation. The
[verification receipt](EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md) records
the interrupted attempts and the post-write verifier correction without
rewriting the artifact or rerunning the model.

## Primary result

The primary metric is the unweighted mean of each task's aligned excess
next-token NLL above its matched FP32 recurrent-state reference.

| Metric | CQER-32 | RHT-CQER-32 | Change |
| --- | ---: | ---: | ---: |
| Task-macro aligned excess NLL | `0.323944` | `0.153129` | **52.73% lower** |
| Task-macro mean KL | `0.328890` | `0.193501` | lower |
| Task-macro CVaR95 KL | `2.046658` | `1.335989` | lower |
| Task-macro top-1 agreement | `0.826000` | `0.879499` | `+0.053498` |
| Aggregate local state SSE | `36,409.363073` | `15,345.844948` | **57.85% lower** |

![Authenticated Stage-B excess-NLL and state-SSE comparison.](../assets/experiment009-stage-b-overview.svg)

The paired task-macro improvement was `0.170815` nats/token. A 10,000-sample
paired task bootstrap with frozen seed `2339` produced a two-sided 95%
equal-tailed interval of `[0.116082, 0.229438]`. RHT-CQER-32 had strictly lower
excess NLL on 27 of 32 tasks, with no ties.

![Per-task paired CQER-32 minus RHT-CQER-32 excess-NLL differences.](../assets/experiment009-stage-b-paired.svg)

Five tasks favored CQER-32. The largest RHT-minus-CQER task disadvantage was
`0.077088` nats/token on task `607`, below the frozen `0.25` ceiling. The
paired chart keeps those negative cases visible instead of hiding them behind
the aggregate.

## Frozen gate decision

| Frozen advancement check | Required | Observed | Result |
| --- | ---: | ---: | :---: |
| Macro excess-NLL reduction | at least 20% | `52.7299%` | pass |
| Paired 95% lower bound | above 0 | `0.116082` | pass |
| Strict task-level wins | at least 20 of 32 | `27/32`, 0 ties | pass |
| Macro mean KL | lower | `0.328890 -> 0.193501` | pass |
| Macro CVaR95 KL | no higher | `2.046658 -> 1.335989` | pass |
| Macro top-1 disadvantage | at most 0.005 | `-0.053498` | pass |
| Maximum task NLL disadvantage | at most 0.25 | `0.077088` | pass |
| Aggregate local state-SSE reduction | at least 50% | `57.8519%` | pass |

All eight advancement checks and every prerequisite integrity check passed.
Under the frozen protocol, RHT-CQER-32 advances beyond this development stage.
The result does not authorize retuning the transform seed, thresholds, task
window, selector, or method name.

## Exact storage and numerical checks

CQER-32 and RHT-CQER-32 both used the same 1,976 promoted Q8 rows:

| Component | Bytes |
| --- | ---: |
| Q4/Q8 payloads | 2,485,760 |
| FP16 scales | 73,728 |
| precision masks | 4,608 |
| **packed recurrent state** | **2,564,096** |
| FP32 query-energy selector | 147,456 |
| **resident bytes including selector** | **2,711,552** |

The state-SSE comparison sums local codec reconstruction error over 35,784
matched recurrent-state records and 9,380,560,896 elements per method. It is a
write-micro reconstruction metric against each method's own pre-pack source
state, not an end-to-end peak-memory or latency measurement.

The independent dense NumPy reference matched the production right-RHT encode
within `9.5367e-7`, matched physical packing exactly, derived the exact frozen
sign schedule, and passed its preregistered thresholds. The production inverse
relative L2 was `1.1468e-7`, below `3e-7`.

## Decision boundary

This 32-task result is a positive, authenticated development result. It does
not replace the separate 500-task held-out v0.2 confirmation, and it is not a
new held-out confirmation for RHT-CQER-32. Ranked MBPP window `[8, 16)` stayed
outside application-level content, tokenization, model, and evaluation access.

Randomized Hadamard and rotation quantizers are established prior art. The
current Python implementation regenerates signs and materializes FP32
transform workspaces; it has no fused-kernel, latency, throughput, peak-memory,
long-context, cross-model, or independent external reproduction result. I
therefore report exactly the frozen development finding and no broader
novelty, deployment, state-of-the-art, speed, or breakthrough claim.
