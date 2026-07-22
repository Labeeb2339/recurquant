# Calibration 002: frozen MBPP layer plan

Status: passed and frozen before development

Date: 2026-07-22

## Frozen inputs

- Repository commit: `cc35f4396ef4dd475908d8f96e05fe9c559f13be`
- Model: `Qwen/Qwen3.5-0.8B-Base`
- Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Dataset: `google-research-datasets/mbpp`, configuration `full`
- Dataset revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- Partition: the 128 training tasks fixed by
  `SHA256("rq-v0.2|{task_id}")` in the public protocol
- Dataset manifest hash:
  `129698d01bdf7f08989878ff7c980230456095c5b2cae42a010f567ecc49dc1a`
- Eligible single-token state-read steps: 7,134

The calibration artifact records Python, PyTorch, Transformers, Datasets, CUDA,
GPU, driver, command, clean-worktree status, task IDs, row hashes, token counts,
and both full 18-layer selector vectors.

## Frozen outcome

The normalized state-MSE baseline selected model layer `0`. Layer `0` also
ranked first under the diagnostic query-weighted read-risk measure. The primary
policy was already frozen to layer `0` before public calibration and was not
changed by this result.

Highest normalized-state-MSE layers:

| Rank | Model layer | Task-macro normalized state MSE |
| ---: | ---: | ---: |
| 1 | 0 | 0.175292 |
| 2 | 17 | 0.105471 |
| 3 | 18 | 0.091148 |
| 4 | 16 | 0.090392 |
| 5 | 21 | 0.081859 |

Highest diagnostic read-risk layers:

| Rank | Model layer | Task-macro read-risk |
| ---: | ---: | ---: |
| 1 | 0 | 0.397649 |
| 2 | 6 | 0.241932 |
| 3 | 5 | 0.228262 |
| 4 | 17 | 0.220239 |
| 5 | 16 | 0.213679 |

The exact-byte development plan is now fixed:

- `read_risk_l0_nearest`: layer 0 INT8, the other 17 GDN layers INT4.
- `mse_selected_nearest`: layer 0 INT8, the other 17 GDN layers INT4.
- Random controls: layers 18, 4, and 13 respectively at INT8.
- Uniform INT4 and INT8 quality anchors.
- Layer-0 mixed stochastic controls at base seeds 2339, 2340, and 2341.

Because the independently specified MSE baseline selected the same layout as
the primary policy, their outputs should be identical. Both names remain in the
artifact so that the prespecified method matrix is not rewritten after seeing
calibration.

## Reproduction check

An initial full calibration completed before repository metadata was added to
the artifact schema. It was treated as incomplete evidence rather than edited
afterward. The full run was repeated from clean commit `cc35f43`.

The two runs matched exactly on:

- all candidate layers;
- every value in both 18-layer score vectors;
- all per-task token counts;
- the 7,134 eligible steps; and
- the complete dataset manifest.

Accepted calibration evidence hash:
`7aa8227dd0b19bb7494963c0b590c8ec53cee29d3b696ccd4087c71a5ac461ee`

Accepted file SHA256:
`d3d2f9acf6113ad455cce78d1b957a265c6675236845564d855c7cb537267125`

## Split-isolation note

During an earlier one-task diagnostic smoke, the non-streaming Hugging Face
Datasets builder prepared local cache files for all four MBPP splits even though
the loader was called with `split="train"`. No validation or test rows were
returned to the calibration code, inspected for method outcomes, or scored. The
loader was then changed to request streaming explicitly, and both accepted full
calibration runs received only the pinned training split. The confirmation guard still
rejects partial access and requires an explicit acknowledgement token before
the test split can be returned.

## Decision

Stage 0 passes. The frozen evaluator may now run all 90 validation tasks once.
This calibration agreement is not a public quality result, a novelty result, or
permission to open the 500-task confirmation split. Confirmation remains gated
on every preregistered development condition in
`PUBLIC_EVAL_PROTOCOL_V02.md`.
