# Experiment 009 Stage-A result: RHT-CQER advances

> **Status: all frozen Stage-A gates passed on the one-task falsification
> screen.**
>
> RHT-CQER-32 reduced closed-loop recurrent-state SSE by `59.97%`, aligned
> excess NLL by `58.59%`, and aligned mean KL by `31.19%` relative to CQER-32
> at the exact same packed-state and selector byte counts. Top-1 agreement was
> unchanged. This authorizes the separately frozen 32-task development stage;
> it is not confirmation, a speed result, or a breakthrough claim.

Date recorded: 2026-07-26

## Authenticated artifact

The method, task identity, sign schedule, implementation, byte contract, and
nine-condition gate were committed before model quality was observed.

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment009-rht-cqer-stage-a-666-5be8d48.json` |
| Artifact kind | `recurquant_rht_cqer32_stage_a_screen` |
| Clean implementation commit | `5be8d48369d94081e55aa389c25f63c303c7b0dd` |
| File SHA-256 | `98a432843dc438f2d5fde34f8704f154ebc3ee12c93ba7c469369acfedfb15b5` |
| Canonical evidence SHA-256 | `9e03a1e8cefb5801406a47a2e5e365686afb0a05e10e099a989cee616b505ed1` |
| Created at | `2026-07-26T03:34:27.699967+00:00` |
| Model | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Device and dtype | CUDA, bfloat16 |
| Task identity | MBPP task `666`, row SHA-256 `b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9` |
| Token identity | 69 prompt, 39 code, 38 aligned scored tokens |
| Evidence verification | valid, with no verifier errors |

The repository and all hashed source files were clean and stable from the
start of model loading through the atomic artifact write. Dataset identity was
authenticated before model weights were loaded. Ranked MBPP window `[8, 16)`
was not loaded, tokenized, or evaluated.

## Frozen screen result

| Metric | CQER-32 | RHT-CQER-32 | Change |
| --- | ---: | ---: | ---: |
| Closed-loop state SSE | `906.603876` | `362.875870` | **59.97% lower** |
| Aligned excess NLL | `0.329244` | `0.136350` | **58.59% lower** |
| Aligned mean KL | `0.224385` | `0.154389` | **31.19% lower** |
| Aligned CVaR95 KL | `1.276270` | `1.022663` | 19.87% lower |
| Aligned maximum KL | `1.563906` | `1.397917` | 10.61% lower |
| Aligned top-1 agreement | `0.868421` | `0.868421` | unchanged |

The full-code secondary metrics showed the same direction:

| Metric | CQER-32 | RHT-CQER-32 |
| --- | ---: | ---: |
| Excess NLL | `0.320802` | `0.132853` |
| Mean KL | `0.218631` | `0.150430` |
| Top-1 agreement | `0.871795` | `0.871795` |

## Exact physical contract

Both methods used:

| Component | Bytes |
| --- | ---: |
| Q4/Q8 payloads | 2,485,760 |
| FP16 scales | 73,728 |
| precision masks | 4,608 |
| **packed recurrent state** | **2,564,096** |
| FP32 query-energy selector | 147,456 |
| **resident bytes including selector** | **2,711,552** |

Every layer retained its frozen target-Fisher quota, totaling 1,976 Q8 rows.
All 39 recurrent-state writes and 107 observed query tokens completed the
stage/consume handshake exactly once. All logits and reported metrics were
finite.

Independent numeric evidence measured a right-RHT inverse relative L2 of
`1.1468e-7`, below the frozen `3e-7` threshold. Physical transformed packing
matched the independent transform-quantize-dequantize reconstruction exactly.
The sign schedule hash matched the preregistered value.

## Decision and limitation

Every frozen Stage-A check passed. Experiment 009 may now resolve and commit
the new ranked MBPP `[32, 64)` development identities before any Stage-B model
quality is observed.

This result contains one deliberately exposed task. It shows that the codec
survived a cheap falsification test; it does not estimate generalization,
statistical uncertainty, long-context behavior, another model size, peak
memory, latency, or fused-kernel performance. Randomized Hadamard and rotation
quantization are established prior art. No novelty, state-of-the-art,
deployment, or breakthrough claim follows from Stage A.
