# Compatibility and support

RecurQuant `0.2.0a1` is an alpha research package. The supported surface is
intentionally narrow because `PackedRecurrentStateCache` subclasses an internal
Transformers cache layer. A passing import or smoke test is not evidence of
model quality, speed, or production readiness.

## Validated compatibility snapshot

| Area | Current evidence | Support boundary |
|---|---|---|
| Python | The full MBPP development and confirmation runs used Python `3.11.15`. Packaging CI is configured to smoke-test Python `3.11` and `3.13` on Linux, plus Python `3.11` on Windows. | Python `>=3.11` is declared. Full-model numerical evidence is limited to `3.11.15`. |
| Transformers | `5.14.1` | The alpha dependency is deliberately pinned to `transformers==5.14.1`. Every other release is unsupported until its internal cache contract is tested. |
| Model | `Qwen/Qwen3.5-0.8B-Base` at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | No other checkpoint, revision, model family, or recurrent architecture has full-model evidence. |
| Model execution | BF16 weights, batch size one, evaluation mode, `trust_remote_code=False`, eager attention | Use `attn_implementation="eager"`. Flash, SDPA, and other attention implementations are not validated. |
| Full-run environment | PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, NVIDIA driver `592.15`, NVIDIA GeForce RTX 5070 Laptop GPU, recorded platform `Windows-10-10.0.26200-SP0` | CPU and other accelerator support is limited to unit or API smoke tests; the public numerical result was not replicated there. |
| Packed formats | Physical INT4 and INT8 recurrent-state payloads. FP16 scales are the evaluated default. | FP32 scales are supported for experiments but have no full-model fidelity evidence. The packed cache does not accept other payload widths. |

The latest full-model evidence and exact provenance are recorded in
[`evidence/mbpp-v02-confirmation.json`](../evidence/mbpp-v02-confirmation.json),
with the human-readable result and claim boundary in
[`research/CONFIRMATION_002.md`](../research/CONFIRMATION_002.md). The earlier
development record remains available in
[`evidence/mbpp-v02-development.json`](../evidence/mbpp-v02-development.json).

## Installed quickstart

`recurquant qwen35` and `examples/qwen35_quickstart.py` use one shared
implementation. Both default to the frozen v0.2 mixed policy: model layer 0 at
INT8 and every other recurrent layer at INT4, with group size 128 and FP16
scales. `--policy uniform-int4-stress` is retained only for reproducing the
uniform INT4 stress baseline. On the current branch,
`--policy statelease-h5` selects the exact Experiment 012 row identity and keeps
the causal StateLease observer active across prefill and decode.

The command downloads the pinned model and tokenizer unless
`--local-files-only` is supplied. It performs manual greedy decoding; the
full-checkpoint quality of free-running generations has not been evaluated.

## Generation and cache modes

The following paths have direct evidence:

- Explicit eager `model(...)` calls with
  `past_key_values=PackedRecurrentStateCache(...)` and `use_cache=True`.
- Batch-one prompt prefill followed by one-token teacher-forced decode calls on
  the pinned full model across all 500 frozen MBPP confirmation tasks and
  30,244 scored reference-code tokens.
- Prefill plus a short multi-token continuation on a tiny randomly initialized
  Qwen3.5 configuration in the unit suite.
- Batch-one greedy generation and beam search through `model.generate(...)` on
  a tiny randomly initialized Qwen3.5 CPU configuration. These are integration
  smoke tests only; free-running output quality was not evaluated.

The local generation smoke used Python `3.11.15`, PyTorch `2.13.0+cpu`, and
Transformers `5.14.1`. It generated three new tokens in greedy and two-beam
modes without downloading model weights.

The release-candidate full-checkpoint smoke used the pinned Qwen3.5 revision,
Python `3.11.15`, PyTorch `2.11.0+cu128`, BF16 weights, and eager CUDA decoding.
It generated a two-token continuation through the installed `recurquant qwen35`
path and reported 2,564,096 resident bytes, 18,874,368 FP32-reference bytes, a
1,048,576-byte largest single state materialization, and physical reduction
realized. This is a functional integration check, not a quality or latency
benchmark.

The StateLease-H5 installed-path smoke used the same pinned model revision,
Python `3.11.15`, PyTorch `2.13.0+cpu`, Transformers `5.14.1`, and eager CPU
decoding. A two-token continuation completed with all 18 layers observed, 36
committed observations, and 18 checkpoints. An eight-token continuation also
exercised the controller: 144 committed observations, 18 c4 decisions, and 36
checkpoints. Both runs reported exactly `3,454,664` resident bytes including the
packed checkpoint (which includes its precision mask), query EMA, and replay
capacity. These are functional
integration checks only; they are not CUDA validation, new Experiment 012
evidence, or latency/quality benchmarks.

The following paths are unsupported or not yet validated:

- sampling, beam sampling, diverse or constrained beam search, assisted or
  speculative decoding, and externally batched generation;
- training, gradient-based use, distributed or sharded inference, cache
  serialization, and cache offload/prefetch workflows;
- non-eager attention implementations; and
- `torch.compile`. Packed cache layers explicitly set `is_compileable=False`.

An unsupported mode may appear to run, but it is outside the alpha compatibility
contract until a regression test and a full-model check are added.

## Memory and performance boundary

RecurQuant measures the live tensor bytes used by packed persistent recurrent
states. The current Python implementation dequantizes one recurrent state while
its layer executes. It therefore makes **no claim** of:

- faster inference or lower latency;
- lower whole-model or peak CUDA memory;
- a fully quantized recurrent kernel; or
- improved generated-code correctness.

The reported compression ratio applies only to resident recurrent-state
payloads and scales, not model weights, ordinary attention KV caches,
activations, allocator overhead, or transient materialization.

## Reporting compatibility problems

Open an issue at <https://github.com/Labeeb2339/recurquant/issues> with:

1. the RecurQuant, Python, PyTorch, and Transformers versions;
2. the exact model ID and revision;
3. device, dtype, attention implementation, batch size, and generation mode;
4. a minimal reproducer and complete traceback; and
5. `cache.storage_summary()` when the failure concerns byte accounting.

Do not include access tokens, local authentication files, private prompts, or
proprietary model data. Because the package is pre-release, compatibility may
change between development versions and such changes will be documented.
