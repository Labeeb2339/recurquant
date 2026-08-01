<p align="center">
  <img src="assets/recurquant-hero.png" width="100%" alt="RecurQuant - packed recurrent state for Qwen3.5">
</p>

<p align="center">
  <a href="https://github.com/Labeeb2339/recurquant/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Labeeb2339/recurquant/ci.yml?branch=main&amp;label=tests" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-6cc5bd" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.11-3776ab" alt="Python 3.11 or newer">
  <img src="https://img.shields.io/badge/transformers-5.14.1-ffd21e" alt="Transformers 5.14.1">
  <a href="https://colab.research.google.com/github/Labeeb2339/recurquant/blob/main/notebooks/recurquant_qwen35_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> |
  <a href="#verified-storage-fidelity-frontier"><b>Trade-off</b></a> |
  <a href="#what-is-physically-smaller"><b>Storage</b></a> |
  <a href="#held-out-confirmation"><b>v0.2 evidence</b></a> |
  <a href="#experiment-009-stage-b-development"><b>Stage B</b></a> |
  <a href="docs/compatibility.md"><b>Compatibility</b></a> |
  <a href="docs/reproducing.md"><b>Reproduce</b></a>
</p>

I wrote RecurQuant to test one narrow question:
can the recurrent memory of a linear-attention model keep generation quality with much less memory if we quantize only that path?

In practice, it is an alpha Python package that physically packs the persistent
recurrent matrix states used by Qwen3.5 Gated DeltaNet layers. You pass its cache
into ordinary eager Transformers model calls so those states remain grouped INT4 or
INT8 payloads between calls.

Its frozen v0.2 layout passed a 500-task held-out MBPP teacher-forced
recurrent-state fidelity protocol. Relative to uniform INT4, task-macro excess
NLL above the matched FP32-state reference was 72.75% lower while the packed
persistent recurrent state, including payloads, FP16 scales, and precision
masks, occupied exactly 2,564,096 resident bytes.

The experimental v0.3 RHT-CQER-32 method has now also passed its separate
32-task development gate: aligned excess NLL was 52.73% lower than CQER-32 at
the same packed-state and selector byte counts. That newer result is
development evidence, not held-out confirmation.

It currently targets
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base).
RecurQuant does not quantize model weights or ordinary attention KV caches, and
its current Python path dequantizes one recurrent state while that layer runs.

I built and maintain RecurQuant as an open research project.
[Muhammad Labeeb Aryan](https://github.com/Labeeb2339). Licensed under
[Apache-2.0](LICENSE).

## Verified storage-fidelity frontier

Among the nearest-rounding quantized layouts evaluated in the held-out
protocol, three form the non-dominated storage-fidelity frontier. Each trades
more resident recurrent-state storage for lower teacher-forced excess NLL. The
frozen v0.2 layout is the middle point: it adds `131,072` bytes (`5.39%`) over
uniform INT4 while lowering task-macro excess NLL by `72.75%`.

![Scatter plot of the held-out storage-fidelity frontier: uniform INT4 at 2.320 MiB and 2.9497 excess NLL, frozen v0.2 mixed precision at 2.445 MiB and 0.8037, and uniform INT8 at 4.570 MiB and 0.0172.](assets/mbpp-confirmation-pareto.svg)

The chart is regenerated directly from the authenticated
[500-task confirmation record](evidence/mbpp-v02-confirmation.json), and CI
rejects stale generated assets. It compares exact resident recurrent-state
bytes with teacher-forced fidelity only. The matched FP32 state reference is
off-plot at `18,874,368` bytes and zero excess NLL by definition; these are not
speed, peak-memory, whole-model-memory, or generated-code results.

## Quickstart

This installs the public v0.2 alpha from its frozen tag. The first model-backed
run downloads the pinned model and tokenizer. Python 3.11 and a CUDA GPU match
the evaluated path; `recurquant demo` uses synthetic states and does not
download a model. RHT-CQER-32 remains an experimental research path rather
than the default package policy.

Windows PowerShell:

```powershell
git clone --branch v0.2.0a1 --depth 1 https://github.com/Labeeb2339/recurquant.git
cd recurquant
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\recurquant.exe qwen35 --max-new-tokens 16
```

macOS or Linux:

```bash
git clone --branch v0.2.0a1 --depth 1 https://github.com/Labeeb2339/recurquant.git
cd recurquant
python3.11 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/recurquant qwen35 --max-new-tokens 16
```

To verify the installation without downloading a model, run `recurquant demo`
with the platform-specific executable path above. It performs a deterministic
synthetic state round-trip and reports physical payload bytes, compression
ratio, and quantization error.

The installed command and
[`examples/qwen35_quickstart.py`](examples/qwen35_quickstart.py) call the same
implementation. The default is the frozen v0.2 mixed policy: layer 0 at INT8
and the remaining recurrent layers at INT4. Uniform INT4 is retained only as an
explicit stress baseline via `--policy uniform-int4-stress`. Add `--json` for
one machine-readable result containing the generated text, pinned model
provenance, selected policy, and raw storage counters. Read the
[compatibility contract](docs/compatibility.md) before using a different model,
Transformers version, device layout, or generation mode.

### Reproducible research commands

I keep Stage-B experiments reproducible with explicit, auditable artifacts.
These commands are wrappers around the existing RHT-CQER Stage-B pipeline with
stateful naming for StateLease.

```powershell
# Identity pass: resolves only artifact contracts, no model loading
.\.venv\Scripts\recurquant.exe resolve-statelease-stage-b-identity ^
  --output evidence/statelease-stage-b-identity.json

# Development evaluation pass: consumes the stage artifacts and writes a result artifact
.\.venv\Scripts\recurquant.exe evaluate-statelease-stage-b ^
  --stage-a-artifact artifacts\experiment009-rht-cqer-stage-a-666-5be8d48.json ^
  --identity-artifact evidence/statelease-stage-b-identity.json ^
  --output evidence/statelease-stage-b-result.json ^
  --device cuda ^
  --local-files-only
```

Both runs are strict about file kinds, evidence hashes, and source-path
provenance. They are for development evidence only; they are not a production
quantization claim by themselves.

## Use it in Python

This example uses the reusable frozen v0.2 helper, which keeps Gated DeltaNet
layer 0 at INT8 and the other 17 recurrent layers at INT4. The generic
`create_qwen35_packed_cache()` factory remains available for controlled policy
experiments.

```python
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recurquant import create_qwen35_v02_mixed_cache

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    dtype = torch.float32
elif torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    warnings.warn(
        "CUDA BF16 is unavailable; falling back to FP16. RecurQuant's public "
        "full-model fidelity evidence has not been validated for FP16 weights.",
        RuntimeWarning,
        stacklevel=2,
    )
    dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    dtype=dtype,
    attn_implementation="eager",
).to(device)
model.eval()

cache = create_qwen35_v02_mixed_cache(model)
inputs = tokenizer("Explain recurrent-state quantization simply.", return_tensors="pt")
inputs = inputs.to(device)
continuation = []

with torch.inference_mode():
    output = model(**inputs, past_key_values=cache, use_cache=True)
    for step in range(32):
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        continuation.append(next_token)
        reached_eos = (
            tokenizer.eos_token_id is not None
            and bool((next_token == tokenizer.eos_token_id).all().item())
        )
        if reached_eos or step == 31:
            break
        output = model(input_ids=next_token, past_key_values=cache, use_cache=True)

generated_ids = torch.cat(continuation, dim=1)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print(cache.storage_summary())
```

`create_qwen35_v02_mixed_cache()` and `create_qwen35_packed_cache()` reject
unsupported Transformers versions, non-eager attention, training mode,
multi-device placement, and incompatible Qwen configurations early. The
returned cache exposes exact live tensor byte accounting through
`storage_summary()`.

## What is physically smaller

For batch-one Qwen3.5-0.8B-Base recurrent states, the frozen mixed layout stores
2,564,096 resident bytes instead of 18,874,368 FP32-state bytes. Uniform INT4
stores 2,433,024 bytes. These figures include packed payloads, FP16 group
scales, and padding.

![Horizontal bars showing 18,874,368 bytes for FP32 recurrent states, 2,433,024 for uniform INT4, and 2,564,096 for the mixed layer-0 INT8 layout.](assets/recurrent-state-storage.svg)

This is recurrent-state storage only. It is not a whole-model, peak-CUDA-memory,
latency, or throughput result.

## Held-out confirmation

The frozen layer-0 mixed layout passed every preregistered v0.2 quality and
integrity gate on the untouched MBPP test split. Task-macro excess negative
log-likelihood above the FP32-state reference fell from `2.949743` for uniform
INT4 to `0.803713`: a **72.75% reduction**.

![Horizontal bars showing held-out MBPP task-macro excess NLL of 2.9497 for uniform INT4 and 0.8037 for the mixed layer-0 INT8 layout.](assets/mbpp-confirmation-fidelity.svg)

The confirmation covers 500 paired tasks and 30,244 teacher-forced reference-
code tokens. The paired mixed-versus-uniform improvement was `2.1460`
nats/token with a 95% bootstrap interval of `[2.0922, 2.1999]`. Against the
mean of three exactly same-byte random high-precision layer placements, the
paired improvement was `2.0332` with a 95% interval of `[1.9802, 2.0861]`.

| Token-weighted measure | Uniform INT4 | Mixed L0 INT8 |
|---|---:|---:|
| Mean KL | 3.149969 | 0.914580 |
| Worst-5% KL | 9.002207 | 4.839139 |
| Top-1 agreement | 0.321155 | 0.665190 |

The earlier 90-task development result was a 74.14% reduction; it remains
available in
[`evidence/mbpp-v02-development.json`](evidence/mbpp-v02-development.json) and
[`DEVELOPMENT_002.md`](research/DEVELOPMENT_002.md).

Important boundaries:

- The accepted result is in
  [`evidence/mbpp-v02-confirmation.json`](evidence/mbpp-v02-confirmation.json),
  with the full decision and interruption record in
  [`CONFIRMATION_002.md`](research/CONFIRMATION_002.md).
- Tokens were scored teacher-forced. Candidate-generated code was not fed back,
  executed, or graded for correctness.
- The MSE selector also chose layer 0, so it is the exact same candidate and not
  independent evidence that the read-risk selector is novel or superior.
- This supports one pinned recurrent-state fidelity and resident-byte result.
  It does not support generated-code quality, speed, peak memory, whole-model
  memory, cross-model generality, or a breakthrough claim.

## Experiment 009 Stage-B development

RHT-CQER-32 applies a deterministic right-side randomized Hadamard transform
inside each existing recurrent-state row group before the same physical Q4/Q8
packing used by CQER-32. The transform does not change the frozen 1,976-row
precision allocation or storage contract: both methods use 2,564,096 packed
state bytes and 2,711,552 resident bytes including the query-energy selector.

On the preregistered 32-task ranked MBPP `[32, 64)` development window,
RHT-CQER-32 passed all eight frozen advancement checks. Task-macro aligned
excess NLL fell from `0.323944` to `0.153129`, a **52.73% reduction**.
Aggregate local recurrent-state reconstruction SSE fell from `36,409.363073`
to `15,345.844948`, a **57.85% reduction**.

![Authenticated Experiment 009 Stage-B excess-NLL and state-SSE comparison.](assets/experiment009-stage-b-overview.svg)

RHT-CQER-32 had lower excess NLL on 27 of 32 tasks, with no ties. The paired
CQER-minus-RHT improvement was `0.170815` nats/token and its frozen
10,000-sample paired 95% bootstrap interval was `[0.116082, 0.229438]`.

![Per-task paired CQER-32 minus RHT-CQER-32 excess-NLL differences.](assets/experiment009-stage-b-paired.svg)

This is positive development evidence on one pinned model and task window, not
held-out confirmation for RHT-CQER-32. Randomized Hadamard quantization is
prior art, and the current Python implementation has no fused-kernel, latency,
peak-memory, cross-model, or independent external-reproduction result. See the
[full Stage-B result](research/EXPERIMENT_009_STAGE_B_RESULT.md),
[verification receipt](research/EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md),
and
[machine-readable release manifest](evidence/experiment009-rht-cqer-stage-b-result-manifest.json).

## Scope

The supported public surface is deliberately narrow:

- Python `>=3.11` and exactly `transformers==5.14.1` for this alpha;
- text-only Qwen3.5 hybrid models with `linear_attention` and `full_attention`
  layer types;
- physical INT4 or INT8 recurrent-state payloads; FP16 scales are the evaluated
  default, while FP32 scales are supported as an experimental, unevaluated
  option;
- eager, evaluation-only, single-device inference; and
- explicit `past_key_values=cache` model calls.

See [`docs/compatibility.md`](docs/compatibility.md) for the validated software,
hardware, model revision, generation paths, and unsupported modes.

## Claim boundary

Quantizing recurrent state is not new. I built RecurQuant to test one narrower
question: can sensitivity-guided mixed precision preserve Gated DeltaNet
recurrent-state fidelity better than simple equal-byte placements? For the
pinned Qwen3.5-0.8B-Base teacher-forced MBPP protocol, the frozen policy passed
the held-out confirmation and beat all three tested same-byte random placements.

That is a confirmed case study, not proof of novelty or general superiority.
Q-Mamba already studies 4-bit persistent Mamba2 states, Quamba2 quantizes cached
SSM states, and other mixed-precision and replay systems overlap parts of this
design space. Experiment 009 adds a positive 32-task development result for a
known right-RHT codec composed with CQER-32, but it is not a new confirmation
result or evidence that Hadamard quantization is new. RecurQuant currently has
no fused packed kernel or measured speed claim. I therefore do not present this
release as a breakthrough, a whole-model memory reduction, or a cross-model
result. See the
[claim boundary](research/CLAIM_BOUNDARY.md) and
[prior-art review](research/PRIOR_ART.md) for the exact comparison.

## Research record

- Independently validate the held-out decision with
  `recurquant verify-confirmation`; the
  [reproduction guide](docs/reproducing.md) pins both committed hashes and
  explains optional raw-checkpoint reconstruction.
- [Held-out MBPP confirmation report](research/CONFIRMATION_002.md) and
  [machine-readable evidence](evidence/mbpp-v02-confirmation.json)
- [Frozen public evaluation protocol](research/PUBLIC_EVAL_PROTOCOL_V02.md)
- [MBPP development report](research/DEVELOPMENT_002.md)
- [Current research status](research/STATUS.md)
- [CORA-C2 development result](research/EXPERIMENT_008_RESULT.md)
- [Experiment 009 frozen protocol](research/EXPERIMENT_009_RHT_CQER_PROTOCOL.md)
- [Experiment 009 Stage-A result](research/EXPERIMENT_009_STAGE_A_RESULT.md)
- [Experiment 009 Stage-B identity freeze](research/EXPERIMENT_009_STAGE_B_IDENTITY.md)
- [Experiment 009 Stage-B result](research/EXPERIMENT_009_STAGE_B_RESULT.md)
- [Experiment 009 verification receipt](research/EXPERIMENT_009_STAGE_B_VERIFICATION_RECEIPT.md)
- [Experiment 009 release manifest](evidence/experiment009-rht-cqer-stage-b-result-manifest.json)
- [Claim boundary](research/CLAIM_BOUNDARY.md) and
  [prior-art review](research/PRIOR_ART.md)
- [Failed proxy signals and empirical-sensitivity pivot](research/EXPERIMENT_001_SIGNAL_PIVOT.md)
- [Scale-format correction and packed/QDQ parity](research/EXPERIMENT_002_SCALE_CORRECTION.md)
- [Earlier pilot protocol](research/PILOT_PROTOCOL.md) and
  [v0.1 diagnostic confirmation archive](research/CONFIRMATION_001.md)

## Contributing

I welcome reproducible compatibility reports, additional model-family adapters,
and work toward a fused packed recurrent kernel. Open an
[issue](https://github.com/Labeeb2339/recurquant/issues) with a minimal
reproducer and `cache.storage_summary()`; never include access tokens, private
prompts, or authentication files.

