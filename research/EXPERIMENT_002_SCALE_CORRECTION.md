# Experiment 002: stored-scale correction and packed-cache parity

Status: correction replay complete; public-dataset validation pending

Date: 2026-07-22

## What was wrong

RecurQuant v0.1 counted each group scale as 16 bits in its modeled storage but
performed the numerical quantize/dequantize calculation with an FP32 scale.
That made the byte estimate internally inconsistent with the simulated number
format. The discrepancy was found while implementing real integer residency.

The implementation now casts each stored scale to the declared FP16 or FP32
format before it is used, includes padded groups in the payload byte count, and
clamps FP16 scales to a finite representable range. INT4 and INT8 payloads can
also be stored physically rather than kept as dequantized floating-point state.

The original v0.1 artifacts remain in history as an audit trail. Their headline
numbers are superseded by the correction runs below; they are not silently
rewritten.

## Corrected diagnostic results

All runs use Qwen3.5-0.8B-Base revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, group size 128, deterministic
nearest rounding, and stored FP16 scales. These are still short diagnostic
traces rather than a public benchmark.

| Trace | Uniform INT4 CVaR95 KL | Layer 0 INT8, rest INT4 | Relative reduction |
| --- | ---: | ---: | ---: |
| Retrieval | 7.2744 | 1.2270 | 83.1% |
| Code | 5.4844 | 2.0482 | 62.7% |
| Multilingual correction replay | 5.7644 | 1.4277 | 75.2% |

The retrieval and code sensitivity sweeps still rank layer 0 as the best
single-layer INT8 promotion after the correction. The simple diagnostic
signals remain unsuitable as a general selector: their rank correlations vary
substantially by trace, including sign changes for forgetting activity.

## Realized resident storage

The packed cache stores INT4 codes two per byte, INT8 codes as signed bytes, and
the declared FP16 scale tensor. At batch one, the 18 Gated DeltaNet matrix
states occupy:

| Policy | Resident bytes | Ratio versus FP32 states |
| --- | ---: | ---: |
| FP32 recurrent states | 18,874,368 | 1.000x |
| Uniform INT4 plus FP16 scales | 2,433,024 | 7.758x smaller |
| Layer 0 INT8, rest INT4, plus FP16 scales | 2,564,096 | 7.361x smaller |

The packed multilingual replay produced exactly the same token-level metrics as
the corrected QDQ replay. This establishes implementation parity for that run,
not losslessness relative to the FP32-state model. The current Python path
materializes one floating-point recurrent state while each layer executes, so
it proves persistent resident-byte reduction but makes no speed or peak-memory
claim.

## Reproducibility records

- Corrected static retrieval evidence hash:
  `61abb54b16989d697f995c04807401bd5920b4c03c4911d1340a4e4082e7dadf`
- Corrected retrieval sweep evidence hash:
  `a077f18867c69e5581b4dd003e4e7ed6d03f9843720fbbe2e1522f837518f1de`
- Corrected code sweep evidence hash:
  `cbac481cd90c99e30bf550a8c6026e680471480a99ff60600bbae57daa4b9776`
- Corrected multilingual QDQ replay evidence hash:
  `38522abbd9b8fc593ee2b02d886f9a48a22d3c589d049d256fb8235800198931`
- Packed multilingual replay evidence hash:
  `62ac233fd51dcbb722850593dfbbb3c1bec54e189c3f8e3812d1aeaa251b3523`

## What this permits

This correction permits a public-dataset development evaluation and a useful
experimental packed-cache release. It does not permit “breakthrough,” novelty,
quality-preserving, end-to-end memory, or speed claims. Those remain gated on
the frozen public protocol and, for systems claims, a fused runtime kernel.
