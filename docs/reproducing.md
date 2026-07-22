# Reproducing and extending RecurQuant

This guide separates three different activities:

1. **Use the cache** to run the supported Qwen3.5 checkpoint. This checks that
   the integration works and reports resident recurrent-state bytes; it does
   not reproduce a quality result.
2. **Reproduce development evidence** on the 90-task MBPP validation split.
   This repeats an already inspected development experiment.
3. **Run or replicate confirmation** on the 500-task MBPP test split. An
   "untouched" confirmation has its policy, manifest, gates, and code frozen
   before outcomes are computed. Once those outcomes are public, later runs are
   replications rather than new untouched confirmations.

None of these workflows establishes generated-code correctness, lower latency,
lower whole-model or peak memory, cross-model generality, novelty, or a
breakthrough.

## Frozen reference

The v0.2 numerical evidence is scoped to:

- Python `3.11.15`;
- PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, and NVIDIA driver `592.15`;
- Transformers `5.14.1` and Datasets `4.8.5`;
- `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`;
- `google-research-datasets/mbpp`, configuration `full`, at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; and
- batch one, eager attention, BF16 model weights on CUDA, evaluation mode, and
  teacher-forced reference-code tokens.

The accepted run used an NVIDIA GeForce RTX 5070 Laptop GPU with 8 GB VRAM.
The 90-task development matrix took 4,198.6 seconds, about 70 minutes. The
500-task confirmation matrix is a long, multi-hour run. CPU execution uses
FP32 model weights and may be substantially slower; it is supported by the
scripts but is not the environment behind the published numerical evidence.
Treat a different software or hardware environment as a replication rather
than a byte-for-byte repeat.

Create the normal contributor environment from the repository root:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev,eval]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

The package pins Transformers `5.14.1`; the other exact reference versions
above are provenance pins, not the full range declared by `pyproject.toml`.
Record the versions actually resolved for a replication. On Linux or macOS,
replace `.venv\Scripts\python.exe` with `.venv/bin/python` below.

The scripts download the pinned model and dataset by default. Add
`--local-files-only` only when the model and tokenizer are already cached. It
does not apply to the streamed MBPP dataset loader.

## 1. Use the packed cache

The installed command runs the pinned checkpoint with the frozen v0.2 layout:
model layer 0 at INT8 and the other 17 recurrent layers at INT4, group size 128,
FP16 scales, and nearest rounding.

```powershell
.venv\Scripts\python.exe -m recurquant.cli qwen35 `
  --device cuda `
  --max-new-tokens 32 `
  --prompt "Explain recurrent-state quantization in two sentences."
```

Use `--device auto` to select CUDA when available and CPU otherwise. The output
includes resident packed bytes, the full-precision recurrent-state equivalent,
the largest single materialized recurrent state, and the resident-state
compression ratio. That largest-state field is not a peak-workspace or peak
CUDA-memory measurement.

For application code, use `create_qwen35_v02_mixed_cache(model)`. Use
`create_qwen35_packed_cache(...)` with `layer_specs` only for an explicitly
custom policy. The model must be in evaluation mode, loaded with eager
attention on one materialized device, and every forward must run inside
`torch.inference_mode()` or `torch.no_grad()`. See
[`compatibility.md`](compatibility.md) for unsupported generation and runtime
modes.

## 2. Reproduce the development evidence

The committed calibration and prepared manifest are sufficient to repeat the
frozen 90-task method matrix:

```powershell
.venv\Scripts\python.exe scripts\evaluate_mbpp.py `
  --phase development `
  --calibration-artifact evidence\mbpp-v02-calibration.json `
  --prepared-manifest evidence\mbpp-v02-development-manifest.json `
  --device cuda `
  --checkpoint artifacts\replication-development.checkpoint.json `
  --output artifacts\replication-development.json
```

To regenerate each input instead of trusting the committed copies, run the
full calibration and then prepare an outcome-free token manifest:

```powershell
.venv\Scripts\python.exe scripts\calibrate_mbpp_layers.py `
  --device cuda `
  --output artifacts\replication-calibration.json

.venv\Scripts\python.exe scripts\evaluate_mbpp.py `
  --phase development `
  --calibration-artifact artifacts\replication-calibration.json `
  --manifest-only `
  --device cuda `
  --output artifacts\replication-development-manifest.json
```

Then substitute those two artifact paths in the first command. A full
development or confirmation run requires `--prepared-manifest`. Do not use
`--limit`, change `--group-size 128` or `--bootstrap-samples 10000`, or add
`--skip-qdq-preflight` when claiming a protocol-eligible replication. The
evaluator rejects those changes for a full public run.

## 3. Replicate the confirmation phase

The explicit lock is an acknowledgement token, not a credential. It prevents
accidental access to the test split. Run this only after freezing your own
policy and analysis plan:

```powershell
.venv\Scripts\python.exe scripts\evaluate_mbpp.py `
  --phase confirmation `
  --calibration-artifact evidence\mbpp-v02-calibration.json `
  --prepared-manifest evidence\mbpp-v02-confirmation-manifest.json `
  --confirmation-lock recurquant:unlock-mbpp-confirmation:rq-v0.2 `
  --device cuda `
  --checkpoint artifacts\replication-confirmation.checkpoint.json `
  --output artifacts\replication-confirmation.json
```

At the commit represented by this guide, the committed confirmation file is a
prepared manifest with `outcomes_computed: false`; it is not a result artifact.
Do not inspect partial candidate metrics, retune on test tasks, or rerun an
unfavourable outcome. A rerun is valid only for a documented infrastructure or
evaluator failure defined without reference to which method looked better.

## Manifests, checkpoints, and hashes

Prepared manifests record the dataset rows and tokenization alongside the
model, dataset, calibration, candidate plan, and repository commit before a
long run. The evaluator recomputes and compares the phase, model revision,
dataset-manifest hash, token-manifest hash, and calibration-evidence hash.
Checkpoints are written atomically after each task. Their run signature binds
the phase, model revision, current repository commit, calibration evidence
hash, prepared-manifest hash, token-manifest hash, group size, and current
candidate plan. Resume by repeating the exact command. A checkpoint from
changed code or settings is rejected instead of silently reused.

The committed integrity anchors are:

| Artifact | Tasks | File SHA256 | Canonical evidence SHA256 |
|---|---:|---|---|
| `evidence/mbpp-v02-calibration.json` | 128 | `d3d2f9acf6113ad455cce78d1b957a265c6675236845564d855c7cb537267125` | `7aa8227dd0b19bb7494963c0b590c8ec53cee29d3b696ccd4087c71a5ac461ee` |
| `evidence/mbpp-v02-development-manifest.json` | 90 | `7d51f732f9d0147c485d3bd3214e9bfcd16132de07df342f0c6b9696160aac3f` | `2b13dfb1799472b2fd0006cb87cea60c67a7af37825a87f1686a6fe3d8e38d7c` |
| `evidence/mbpp-v02-development.json` | 90 | `5980fd58aa0933ad97deb896d4901fcd37350c4a57d8a80022ab218aaf77e727` | `301c52e194bbd23059a0040a8e94aeac97dc33de1100f13edbf17dc877755488` |
| `evidence/mbpp-v02-confirmation-manifest.json` | 500 | `c6a7d0db6ef7577a66ac19fbbc0be166279488f6a6be432b364bd9eb6833f7b0` | `21a6d18c6a0887b1499d156a3d610d4bfafdd59d3557713485b62038e263b96a` |

Their recorded source commits are respectively
`cc35f4396ef4dd475908d8f96e05fe9c559f13be`,
`3a3c4a2a11c0822f6c456a74327127f294ce67e1`,
`20a5ea95a8ed692600ee1645d2913f3a4b8a6795`, and
`44d75a2776fa36441e17cc688965c9825c4c1a1c`.

Validate a committed artifact without loading the model or dataset:

```powershell
.venv\Scripts\python.exe -m recurquant.cli verify-artifact `
  evidence\mbpp-v02-development.json `
  --expect-file-sha256 5980fd58aa0933ad97deb896d4901fcd37350c4a57d8a80022ab218aaf77e727 `
  --expect-canonical-evidence-sha256 301c52e194bbd23059a0040a8e94aeac97dc33de1100f13edbf17dc877755488
```

The command exits nonzero if the JSON is malformed, the recorded canonical
hash is wrong, or either expected anchor does not match. Its report is JSON so
the same check can run in CI.

Cross-machine invariant manifest hashes are:

| Phase | Dataset manifest SHA256 | Token manifest SHA256 |
|---|---|---|
| Calibration | `129698d01bdf7f08989878ff7c980230456095c5b2cae42a010f567ecc49dc1a` | not produced by the calibration script |
| Development | `8fed3da0aae864f4e30c70ad70b0269f759d3592dccb9ab87f24444fa24d65dc` | `3c19f37f3c35cb22f17e66a5438bb9968b214e0aff64067930bc3a7124f63f5c` |
| Confirmation | `060aaff7117dc47af6c01253a912f34b6956241c336bbc7216e73bca8624d2d4` | `199a8836489af9bd0af3fec027e85d57df356bd9919492b24015de51d143f525` |

`canonical_evidence_sha256` is SHA256 over `artifact["evidence"]` serialized
with sorted keys, two-space indentation, standard JSON ASCII escaping,
`allow_nan=False`, and one trailing newline. It protects the evidence payload
but includes environment and command provenance, so a legitimate replication
on another commit or machine need not match the accepted canonical hash. The
dataset and token manifest hashes should match when the frozen inputs and
tokenizer do.

## Extend with a new policy

Do not edit the committed v0.2 evidence files or reinterpret its confirmation.
Start with a separate diagnostic artifact. The existing smoke script can test
one or more promoted recurrent layers without touching MBPP:

```powershell
.venv\Scripts\python.exe scripts\run_qwen35_smoke.py `
  --cache-mode packed `
  --low-bits 4 `
  --high-bits 8 `
  --upgrade-layers 6 `
  --prompt-profile code `
  --device cuda `
  --output artifacts\policy-layer6-diagnostic.json
```

Use `--sensitivity-sweep` instead of `--upgrade-layers` to evaluate every Gated
DeltaNet layer on one of the script's synthetic prompt profiles. These traces
are diagnostics, not public-task or confirmation evidence.

If a policy remains promising:

1. add it through `layer_specs` and add focused unit/integration tests;
2. write a new versioned protocol with its own calibration, development, and
   confirmation partitions or dataset;
3. freeze new candidate definitions and outcome-free manifests before scoring;
4. write artifacts to new versioned filenames rather than replacing anything
   under `evidence/mbpp-v02-*`; and
5. report failed gates and negative controls with successful results.

Changing `candidate_definitions()` in `scripts/evaluate_mbpp.py` creates a new
experiment; it does not extend the frozen v0.2 claim. Follow
[`CONTRIBUTING.md`](../CONTRIBUTING.md), keep the pull request to one testable
change, and include the exact command, environment, artifact, metric definition,
and claim boundary. Read the complete frozen design in
[`PUBLIC_EVAL_PROTOCOL_V02.md`](../research/PUBLIC_EVAL_PROTOCOL_V02.md) and the
current restrictions in [`CLAIM_BOUNDARY.md`](../research/CLAIM_BOUNDARY.md).
