# Experiment 003: HRR same-calibration diagnostic

Status: H32 horizon hypothesis rejected at the diagnostic stage; pivot opened

Date: 2026-07-22

## Decision

I tested whether looking farther ahead improved the row allocator before
freezing a development protocol. It did not. At the same physical byte budget,
`hrr_h32` produced higher macro excess NLL than the immediate-read `hrr_h1`
selector:

```text
hrr_h1  macro excess NLL = 0.6218181103467941
hrr_h32 macro excess NLL = 0.7428321987390518
H1 - H32                  = -0.12101408839225769
paired 95% interval       = [-0.21324796304106708, -0.04036934301257135]
```

The interval excludes zero in H1's favor, and H1 had lower excess NLL on seven
of the eight tasks. H32 also had worse macro mean KL, CVaR95 KL, and top-1
agreement. I therefore reject the H32 horizon hypothesis at this diagnostic
stage. I will not carry H32 forward as the primary candidate or tune the
horizon on these same examples.

## Scope and evidence boundary

This was a same-data implementation diagnostic, not a held-out evaluation.
Both row selection and quality evaluation used the same eight MBPP `train`
calibration examples in this order:

```text
945, 794, 657, 702, 651, 720, 903, 918
```

The quality run scored 650 teacher-forced code tokens. The selector trace had
642 pre-update decode positions because each sequence of `n` code tokens
provides `n - 1` recurrent transitions. The model and data were:

- model: `Qwen/Qwen3.5-0.8B-Base`
- model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- MBPP revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- MBPP manifest SHA256:
  `97d691a6d45ee29668f5b0151c1a0885629539ac9e7967e9baa30cfb5c97ae8b`
- selector seed: `2339`
- random-row baseline seed: `1101`
- paired bootstrap: 10,000 task-level resamples, seed `2339`

This scope can reveal a selector or cache integration failure. It cannot show
held-out generalization, cross-domain robustness, speed, or end-to-end memory
improvement.

## Reproducibility anchors

The artifacts were generated from a dirty experimental worktree rooted at Git
commit `23f07cfe5f0dfd2406bbc4f85f885280e467b533`; that commit alone does not
reconstruct the uncommitted experimental implementation.

| Record | SHA256 |
| --- | --- |
| Selector canonical evidence | `b3a2ee24d2a2dc8bbf9d2b81e9909b7d8844fe0e611cdc0dc51570b5db562b43` |
| Selector artifact file | `a0627fde2681c8085ff3667d3b7c21d7a7a882a01f62491704492f39303e30fb` |
| Quality canonical evidence | `8bac3d436597d0e6dee563c037c56fd4369597f11745dff67de98b5ebeffb9d2` |
| Quality artifact file | `fdd05416a696554c2c6450df6c7ee4f1090fed05cb3910df4ff66ad4e400b903` |

The ignored local files are `artifacts/hrr-diagnostic-8task.json` and
`artifacts/hrr-quality-diagnostic-8task.json`.

These hashes authenticate the two local files, not the unavailable dirty
source state that produced them. The numerical results in this document are
therefore locally verified but not release-reproducible from a clean public
commit. Before publication as reproducible evidence, rerun the diagnostic from
the clean committed implementation and publish the resulting artifacts. Until
then, the numbers remain historical exploratory observations only.

## Physical format and selectors

The row-format selectors used the same exact resident-state budget:

| Component | Bytes |
| --- | ---: |
| INT4/INT8 payload | 2,485,760 |
| 36,864 FP16 row scales | 73,728 |
| One precision bit for each row | 4,608 |
| **Total** | **2,564,096** |

Each row-format policy promoted exactly 1,976 of 36,864 rows from INT4 to
INT8. Each promotion added 64 payload bytes. The physical cache reported the
same `2,564,096` resident bytes for `hrr_h1`, `hrr_h32`, `row_mse`, and
`random_rows_s1101`. It also reported `2,564,096` bytes for the mask-free v0.2
static layout and `2,433,024` bytes for uniform INT4. The FP32 recurrent-state
reference occupied `18,874,368` bytes.

The evaluated selectors were:

- `hrr_h1`: actual INT4-to-INT8 marginal read-risk reduction at the immediate
  read, measured on frozen full-precision traces;
- `hrr_h32`: the same marginal score propagated across up to 32 future reads;
- `row_mse`: marginal row reconstruction-MSE reduction from INT4 to INT8;
- `random_rows_s1101`: equal-count random row promotions with fixed seed 1101;
- `v02_layer0_static`: layer 0 at INT8 and the other recurrent layers at INT4;
- `uniform_int4`: all recurrent-state rows at INT4.

H1 and H32 selected substantially similar policies: 1,872 promoted rows
overlapped, with Jaccard similarity `0.9`. The remaining 104 promotions in
each policy were enough to produce a measurable quality difference in this
small diagnostic.

## Numeric results

All values are task-macro aggregates over the same eight examples and 650
teacher-forced tokens. Lower excess NLL, mean KL, and CVaR95 KL are better;
higher top-1 agreement is better.

| Method | Excess NLL | Mean KL | CVaR95 KL | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| `hrr_h1` | 0.6218181103467941 | 0.635199373587966 | 3.388882279396057 | 0.7301077470183372 |
| `hrr_h32` | 0.7428321987390518 | 0.7663818132132292 | 4.135775178670883 | 0.7108453139662743 |
| `row_mse` | 0.7827504575252533 | 0.8211609497666359 | 3.675527900457382 | 0.6954880431294441 |
| `v02_layer0_static` | 0.8490711152553558 | 0.9121971130371094 | 4.451788902282715 | 0.6625703647732735 |
| `uniform_int4` | 2.6367427557706833 | 2.796179473400116 | 7.581509709358215 | 0.3663086034357548 |
| `random_rows_s1101` | 2.683250844478607 | 2.791328117251396 | 7.672792077064514 | 0.36465270072221756 |

The paired contrasts below are `baseline excess NLL - hrr_h32 excess NLL`.
Positive values favor H32; negative values favor the named baseline.

| Baseline | Mean paired contrast | Paired 95% interval |
| --- | ---: | ---: |
| `hrr_h1` | -0.12101408839225769 | [-0.21324796304106708, -0.04036934301257135] |
| `row_mse` | 0.03991825878620148 | [-0.05020202696323395, 0.12860547192394733] |
| `v02_layer0_static` | 0.10623891651630402 | [0.031215629354119343, 0.20246222615242004] |
| `uniform_int4` | 1.8939105570316315 | [1.5334751740098, 2.2339482568204403] |
| `random_rows_s1101` | 1.9404186457395554 | [1.6602401442825794, 2.197948945313692] |

H32 beat the coarse static, uniform, and single random baselines in this
same-data run. That does not rescue the horizon hypothesis: H1 was the stronger
equal-format selector, and row MSE was statistically unresolved against H32.

## Working diagnosis, not a finding

HRR scores one quantization error injected into a captured full-precision state
and propagates it through frozen full-precision `q`, `k`, `g`, and `beta`
values. The physical cache behaves differently: it quantizes after prefill and
after every recurrent update, so later states and activations come from an
already perturbed, repeatedly requantized trajectory.

One possible explanation is that a 32-read score accumulates effects that do
not survive, or are replaced by new error, under repeated requantization. H1
may align better because it asks only about the next read. This is a post-hoc
hypothesis, not a demonstrated cause. The current artifacts do not isolate
horizon length from repeated requantization, and the high H1/H32 policy overlap
makes causal interpretation especially weak.

To test this explanation, a future diagnostic would need controlled one-shot
versus repeated-quantization ablations and direct row interventions. I will not
describe the mismatch as the reason for failure until those tests exist.

## Pivot: empirical downstream loss sensitivity

The next selector will target measured downstream loss rather than extending a
hand-crafted read proxy farther through a frozen trajectory. The planned path
is:

1. Build a small single-row promotion oracle on the physical, repeatedly
   quantized cache and measure the actual change in downstream excess NLL.
2. Test a directional loss or empirical-Fisher score with respect to the
   recurrent state, using the real `Q8(S) - Q4(S)` promotion direction rather
   than weight magnitude or a generic parameter Hessian.
3. Compare that scalable score with the intervention oracle, H1, row MSE,
   static v0.2, and prespecified random seeds at the same counted byte budget.
4. Freeze a new development protocol only if the score is stable across
   calibration subsets and improves on the strongest baseline without tuning
   on its evaluation examples.

This is a new hypothesis. The present diagnostic rejects H32; it does not show
that a loss/Fisher selector will succeed.
