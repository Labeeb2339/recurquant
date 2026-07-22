# Contributing to RecurQuant

RecurQuant is an alpha research package with a deliberately narrow supported
surface. Contributions are most useful when they include a small reproducer, a
regression test, and an explicit statement of what the result does **not** show.

## Set up and check a change

The repository baseline is Python 3.11. On Windows, the same setup used by the
project is:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev,eval]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

On Linux or macOS, replace `.venv\Scripts\python.exe` with
`.venv/bin/python` in the last three commands.

The full-model experiments are separate from the unit suite and download public
model weights. Do not run them merely to validate a small code change. If a
change affects Qwen3.5 integration, also run the smallest relevant smoke path
documented in [compatibility and support](docs/compatibility.md), and report the
exact model revision, device, dtype, attention implementation, and Transformers
version.

Before submitting a change:

- add or update a focused test for changed behavior;
- run the complete unit suite and Ruff command above;
- keep generated evidence separate from hand-written interpretation; and
- do not include access tokens, authentication files, private prompts,
  proprietary model data, or local machine secrets in code, logs, artifacts, or
  issues.

## Evidence and claim boundary

Use the terminology in [the project claim boundary](research/CLAIM_BOUNDARY.md).
In particular:

- report packed **resident recurrent-state bytes** separately from model
  weights, attention KV caches, activations, allocator overhead, and the one
  state materialized during a layer call;
- do not turn byte accounting into a speed, latency, peak-CUDA-memory, or
  whole-model-memory claim;
- do not describe recurrent-state quantization itself as new, and do not call a
  result a breakthrough from a diagnostic trace or a development split;
- attach the command, pinned revisions, environment, metric definition, and
  machine-readable artifact for a new numerical claim; and
- preserve failed gates and negative results instead of silently replacing
  them.

The current supported package boundary is recorded in
[docs/compatibility.md](docs/compatibility.md). A new model, Transformers
release, execution backend, generation mode, or hardware target is unsupported
until it has a regression test and clearly scoped full-model evidence.

## Keep pull requests reviewable

Keep each change to one testable behavior. In the description, state the
problem, the exact verification commands, any new evidence file, and the claim
boundary affected. Documentation-only corrections should identify the source
artifact or code path that makes the replacement wording factual.
