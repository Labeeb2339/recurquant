# RecurQuant

RecurQuant is an experimental drop-in cache for **physically packed persistent
recurrent states in Gated DeltaNet language models**.

> **Current status:** INT4 and INT8 recurrent-state payloads now remain packed
> between layer calls. On Qwen3.5-0.8B-Base, the frozen mixed layout occupies
> 2,564,096 resident state bytes instead of 18,874,368 FP32-state bytes. This is
> a recurrent-state-only result; the Python path still materializes one state
> while its layer executes and makes no speed or whole-model memory claim.

After correcting v0.1 to emulate its declared FP16 scale storage, retaining only
Gated DeltaNet layer 0 at INT8 and using INT4 for the other 17 layers reduced
worst-5% token KL by 83.1% on a retrieval trace, 62.7% on a code trace, and
75.2% on a multilingual correction replay relative to uniform INT4. These are
short diagnostics, not a public benchmark or a breakthrough claim. The frozen
[MBPP public-evaluation protocol](research/PUBLIC_EVAL_PROTOCOL_V02.md) is the
next credibility gate.

## Research question

Can sub-8-bit storage of Gated DeltaNet's fixed recurrent matrix state allocate
precision from query-weighted read sensitivity to preserve difficult
long-context behavior better than uniform quantization at the same modeled bit
budget?

[Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) is the first
target. Its language model repeats three Gated DeltaNet layers followed by one
full-attention layer, giving 18 persistent recurrent states and six ordinary KV
caches across 24 layers.

## What this repository measures

- Deterministic grouped INT8, INT6, and INT4 state round trips.
- Physical INT4 nibble packing and INT8 payload storage with FP16/FP32 scales.
- A `transformers` cache that keeps Gated DeltaNet states packed between calls.
- Per-layer state size and numerical error.
- Paired token-level KL divergence and top-1 agreement against an FP32-state run.
- Tail error rather than only average perplexity.
- State-update magnitude for later sensitivity analysis.
- Query-weighted recurrent-read error, which measures the effect of state error
  on the actual `q^T S` read.
- Exact resident payload and scale bytes, including group padding.

The current implementation reduces the resident tensors used for persistent
recurrent-state storage. It dequantizes one state for the unmodified layer
kernel, so it **does not prove lower peak CUDA memory or faster inference**. A
fused quantized recurrent kernel is still required for those systems claims.

## Use the packed cache

```python
from recurquant import PackedRecurrentStateCache, QuantizationSpec

cache = PackedRecurrentStateCache(
    model.config,
    spec=QuantizationSpec(bits=4, group_size=128),
    layer_specs={0: QuantizationSpec(bits=8, group_size=128)},
)
output = model(input_ids, past_key_values=cache, use_cache=True)
print(cache.storage_summary())
```

This v0.2 development release targets
`transformers>=5.14.1,<5.15` because it integrates with that version's linear
attention cache contract. The default cache does not retain per-token evidence,
so its bookkeeping remains bounded with sequence length.

## Claim boundary

Quantizing recurrent states is not new. Existing SSM work includes
[Quamba2](https://arxiv.org/abs/2503.22879), while newer systems use quantized
state checkpoints, stochastic rounding, and replay. Gated DeltaNet work also
uses update residuals to manage auxiliary memory. RecurQuant therefore does not
claim to be the first recurrent-cache quantizer, update-aware memory method, or
state-replay system.

The narrower hypothesis under investigation is **precision allocation for the
persistent Gated DeltaNet matrix state**, conditioned on Gated DeltaNet dynamics
and compared at an equal bit budget. See
[the claim boundary](research/CLAIM_BOUNDARY.md) and
[pilot protocol](research/PILOT_PROTOCOL.md). The documented experiment trail
preserves the [failed signals and replacement](research/EXPERIMENT_001_SIGNAL_PIVOT.md)
and the [untouched confirmation](research/CONFIRMATION_001.md).
The later [scale-format correction and packed parity record](research/EXPERIMENT_002_SCALE_CORRECTION.md)
supersedes the original numerical headline while preserving its history.

The user-suggested
[Gated DeltaNet-2 paper](https://arxiv.org/abs/2605.22791) reinforces why erase,
write, and decay behavior should be analyzed separately. Its non-commercial
reference code is not a RecurQuant dependency, and the initial target remains
Apache-2.0 Qwen3.5.

## Local setup

Windows with an NVIDIA GPU:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev,eval]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\recurquant.exe demo --bits 4 --group-size 128
```

The model experiment is intentionally separate from the unit-test suite because
it downloads approximately 1.75 GB of public model weights.

## Reproduce the frozen confirmation

The script pins the model revision and records the environment, token digest,
state layout, metrics, and canonical evidence hash:

```powershell
.venv\Scripts\python.exe scripts\run_qwen35_smoke.py `
  --upgrade-layers 0 --low-bits 4 --high-bits 8 `
  --group-size 128 --rounding nearest `
  --cache-mode packed `
  --prefill-tokens 32 --decode-tokens 32 `
  --prompt-profile multilingual `
  --output artifacts\multilingual-confirmation.json
```

This reruns the already disclosed confirmation profile; it is a reproducibility
check, not a new held-out test. The recorded result and its limitations are in
[Confirmation 001](research/CONFIRMATION_001.md).

## Research discipline

- Model and tokenizer revisions are pinned in evidence artifacts.
- Calibration, development, and confirmation prompts must remain separate.
- Static baselines run before any adaptive policy is tuned.
- Simple averages of decay, write, update norm, and residual magnitude are kept
  as negative pilot evidence; they did not predict layer sensitivity reliably.
- Real latency is reported only after a packed kernel exists.
- Failed gates and negative results remain visible.
- Derived checkpoints must retain the base model's name, license, and lineage.

## License

RecurQuant code is licensed under Apache-2.0. Referenced papers, models, datasets,
and repositories retain their own licenses.
