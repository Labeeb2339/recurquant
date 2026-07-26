# Exact-byte Q4/Q6/Q8 reference design

> **Status: implemented and unit-tested as a correctness-first reference.**
>
> This is not yet connected to the Qwen3.5 cache evaluator and has no quality,
> latency, kernel, novelty, or deployment result.

Last updated: 2026-07-26

## Physical format

Every 128-value recurrent-state row uses one symmetric signed quantizer and
one FP16 scale:

| Precision | Code range | Payload per row |
| --- | ---: | ---: |
| Q4 | `[-7, 7]` | 64 bytes |
| Q6 | `[-31, 31]` | 96 bytes |
| Q8 | `[-127, 127]` | 128 bytes |

Q6 uses an exact little-endian bit stream: four two's-complement 6-bit codes
occupy three bytes. Separate Q4, Q6, and Q8 pools are indexed by a canonical
two-bit precision stream:

```text
0 = Q4
1 = Q6
2 = Q8
3 = invalid
```

The public reference object validates its complete shape, scale, code-stream,
pool, dtype, device, contiguity, and byte-count invariants when constructed.

## Correct equal-byte accounting

The pinned Qwen3.5 cache contains:

```text
18 layers * 16 heads * 128 rows = 36,864 row groups
```

The existing Q4/Q8 design spends one precision bit per row and promotes 1,976
rows to Q8:

```text
payloads: 2,485,760 bytes
scales:      73,728 bytes
mask:         4,608 bytes
total:    2,564,096 bytes
```

A general three-width design requires two bits per row, so its precision
stream costs 9,216 bytes. At the same total state budget, it has exactly 3,808
32-byte marginal steps:

```text
Q4 payload baseline: 2,359,296 bytes
FP16 scales:             73,728 bytes
two-bit codes:            9,216 bytes
3,808 steps * 32:       121,856 bytes
total:                 2,564,096 bytes
```

This is **not** 3,952 Q6 rows and not an exact two-times coverage result. A row
at Q6 consumes one step; a row at Q8 consumes two. The final counts depend on
the allocator.

To preserve every old layer's byte count, a layer with old Q8 quota `q` gets:

```text
K = 2q - 8
```

The eight-step subtraction pays for that layer's additional 256 precision-code
bytes. The frozen per-layer step budgets sum to 3,808 and reproduce the
2,564,096-byte total exactly.

## Exact reference allocator

For each row, the allocator receives complete weighted distortions
`D4`, `D6`, and `D8`. It minimizes total distortion subject to:

```text
sum(precision_code) = K
precision_code in {0, 1, 2}
```

The CPU reference uses dynamic programming over all complete row states. It is
globally optimal even when the Q6-to-Q8 improvement is larger than the
Q4-to-Q6 improvement, where naive independent marginal sorting can violate
precedence. Exact ties give higher precision to the earlier flattened row.

For `N` rows and step budget `K`, the reference complexity is:

```text
time: O(NK)
choice storage: O(NK) bytes
FP64 workspace: O(K)
```

This is an oracle/correctness path, not the final per-token runtime selector.
An optimized deployment path would need a GPU allocator and a packed-native
mixed dequantization/Gated DeltaNet kernel. INT6 has no native arithmetic path
on the tested GPU, so end-to-end benchmarks—not payload math—must decide
whether its additional bandwidth efficiency survives unpack cost.

## Claim boundary

Mixed-bit allocation, rate-distortion optimization, INT6 formats, and dynamic
precision are established ideas. RateQuant, MixKVQ, OuroMamba, and related
work are close prior art. The current contribution is limited to an auditable
physical format, exact byte parity, and a globally optimal reference allocator
for this recurrent-state geometry. A new frozen quality experiment is required
before the design can be recommended over Q4/Q8 CQER.
