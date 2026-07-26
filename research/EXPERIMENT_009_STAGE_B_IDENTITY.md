# Experiment 009 Stage-B identity freeze

> **Status: the 32-task development identity is authenticated and frozen
> before model weights or quality metrics are opened.**
>
> This artifact fixes the data, tokenizer, row-allocation plan, runtime, and
> source-code identity for the one permitted Stage-B run. It is not
> performance, novelty, state-of-the-art, or breakthrough evidence.

Date recorded: 2026-07-26

## Authenticated artifact

| Field | Value |
| --- | --- |
| Artifact | `evidence/experiment009-rht-cqer-stage-b-identity-cdc603b.json` |
| Artifact kind | `recurquant_rht_cqer32_stage_b_identity` |
| Frozen source commit | `cdc603b6b8462e42e98495c5ec0610faa9473721` |
| File SHA-256 | `f94e60adb6d9ab51b5894e3ad8b3c0064a8e69721e3df24f90a81f0b4009fe19` |
| Canonical evidence SHA-256 | `4074db6860d3f36dff1a13b7945d0c5c5d619ee208f0fdb35108defb9512c343` |
| Created at | `2026-07-26T05:12:01.884112+00:00` |
| Ordered identity SHA-256 | `3a7f9f2e1b60321680082d93ea983425085f0e26afad03c49413b41e6e4b2ddc` |
| Content manifest SHA-256 | `7a66a3d1241baf27887f43f04129b9e6d8ce46f24e22835b9888cab53d172bcb` |
| Token manifest SHA-256 | `76b73079b704c48a5e87d479887e5b88d2a0a9c1516c1187acae22e0c71618a9` |
| Row-plan SHA-256 | `b480b6483bec2f07ef56388df27f5d78402b9b9545f2c039d332495de2a9fbde` |

The artifact loader independently recomputed its canonical hash and validated
the complete production schema. The committed copy is byte-identical to the
generated artifact.

## Frozen evaluation identity

The development set contains 32 ranked MBPP tasks from the preregistered
`[32, 64)` window:

```text
725, 616, 686, 722, 919, 636, 607, 950,
885, 646, 708, 840, 793, 860, 800, 867,
622, 677, 849, 756, 960, 741, 763, 924,
839, 870, 862, 877, 672, 894, 911, 907
```

| Token total | Count |
| --- | ---: |
| Prompt tokens | 4,079 |
| Code tokens | 1,988 |
| Aligned scored tokens | 1,956 |

The model and tokenizer are fixed to
`Qwen/Qwen3.5-0.8B-Base` at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The tokenizer is
`Qwen2Tokenizer` from Transformers `5.14.1`, with the exact prompt/code
special-token settings recorded in the artifact.

## Frozen runtime and row plan

The identity authenticated the same runtime used for Stage A before reading
dataset content:

| Component | Frozen value |
| --- | --- |
| Python | `3.11.15` |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | `12.8` |
| Transformers | `5.14.1` |
| Datasets | `4.8.5` |
| NumPy | `2.4.6` |
| Safetensors | `0.8.0` |

The compact selector plan promotes 1,976 of 36,864 recurrent-state groups.
It preserves the exact 2,564,096-byte packed-state target, including the
4,608-byte precision mask. Both source selector artifacts and their canonical
evidence hashes are bound inside the identity.

## Access and integrity boundary

The identity run:

- authenticated Stage A, the runtime, selector artifacts, source files, and
  repository commit before target content access;
- used a task-ID-only ranking pass and retained no row mappings from it;
- retained, canonicalized, formatted, and tokenized only the 32 Stage-B tasks;
- did not run a model forward pass;
- did not load model weights, logits, or quality metrics;
- did not access the protected ranked `[8, 16)` window at the application
  content, tokenization, model, or evaluation layers; and
- recorded repository-relative paths only.

The Hugging Face streaming transport may deserialize complete source records
before yielding them. RecurQuant itself inspected only `task_id` on
non-target rows; the artifact records both transport and application-level
counters.

## Safe runtime rejection

The first invocation used a different local virtual environment. Its runtime
check detected that `datasets` was absent and stopped before dataset access or
artifact creation. The successful invocation used the exact Stage-A runtime
listed above. No dependency, seed, threshold, task window, selector, or source
file was changed after the rejection.

## Historical authorization

At the time of this freeze, the identity authorized one Stage-B development
evaluation whose result had to be preserved whether the eight advancement
gates passed or failed. No threshold, seed, method, task, or source file could
be changed after observing the result.

That evaluation is now complete. The authenticated outcome is recorded in the
[Stage-B result](EXPERIMENT_009_STAGE_B_RESULT.md), and the immutable-artifact
audit is recorded in the
[verification receipt](EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md).
