# Claim boundary

Last reviewed: 2026-07-22

## What is already established

- Weight-only and activation quantization for language models are mature fields.
- SSM persistent/cached states have already been quantized, including 4-bit
  Mamba2 state caches in
  [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) and 8-bit
  cached-state kernels in [Quamba2](https://arxiv.org/abs/2503.22879).
- Stochastic rounding, periodic state checkpoints, activation/update replay, and
  fewer persistent-state writes exist in current recurrent-model systems work.
- Update residuals have been used as importance signals for bounded auxiliary
  memory in Gated DeltaNet.
- Mixed precision and memory-budget allocation are broad existing ideas.

## Scoped v0.2 confirmation finding

On the pinned `Qwen/Qwen3.5-0.8B-Base` revision, the frozen layer-0 INT8 plus
17-layer INT4 policy passed every preregistered quality gate on all 500 untouched
MBPP test tasks and 30,244 teacher-forced reference-code tokens. Relative to
uniform INT4, task-macro excess NLL fell from 2.949743 to 0.803713 nats/token, a
72.75% reduction. The paired improvement was 2.146030 nats/token with a 95%
bootstrap interval of `[2.092249, 2.199866]`. The policy physically stored its
persistent recurrent-state payloads and FP16 scales in exactly 2,564,096 bytes.

This confirms the frozen quality hypothesis only for the model, dataset
construction, and teacher-forced metric in
[`CONFIRMATION_002.md`](CONFIRMATION_002.md). It is not evidence of novelty,
generated-code correctness, lower latency, lower whole-model or peak memory, or
cross-model generality. Mean decay, write gate, state-update magnitude, and
residual magnitude remain rejected diagnostic selectors from the earlier pilot.

## Claims prohibited without new evidence

- "First recurrent-state quantization method."
- "First sub-8-bit recurrent model."
- "First update-aware recurrent cache."
- "Lossless" unless exact output equivalence is demonstrated in the stated mode.
- "Reduces total model memory by 7.36x." The measured ratio covers resident
  recurrent-state payloads and scales only.
- "Speeds up inference" without passing the separate latency gate under the
  stated benchmark.
- "Breakthrough," "state of the art," or "novel" based on the v0.2 evidence.
- Authorship of Qwen3.5 or any other base model.

## What this project contributes

RecurQuant packages a physical low-bit recurrent-state cache, a frozen per-layer
allocation policy, exact-byte controls, an independently checkable confirmation
verifier, a preregistered evaluation protocol, and labelled evidence artifacts.
Its scoped held-out result is reproducible without expanding the claim. The
current Python path is not a fused low-bit recurrence kernel. RecurQuant does
not claim authorship of Qwen3.5 or its architecture, which remain credited to
the Qwen team.
