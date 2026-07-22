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

## Hypothesis that remains testable

At an equal modeled persistent-state bit budget, Gated DeltaNet-specific
precision allocation may reduce tail output divergence relative to uniform
state quantization. The current candidate uses query-weighted recurrent-read
error. Mean decay, write gate, state-update magnitude, and residual magnitude
were tested and rejected as selectors in the diagnostic pilot.

This is a hypothesis, not a novelty or performance claim. A broader literature
audit and confirmatory experiments are required before stronger wording.

## Claims prohibited without new evidence

- "First recurrent-state quantization method."
- "First sub-8-bit recurrent model."
- "First update-aware recurrent cache."
- "Lossless" unless exact output equivalence is demonstrated in the stated mode.
- "Reduces memory" while only storing dequantized PyTorch tensors.
- "Speeds up inference" without a packed kernel and wall-clock comparison.
- "Breakthrough," "state of the art," or "novel" based only on the pilot.
- Authorship of Qwen3.5 or any other base model.

## What this project contributes

RecurQuant evaluates a frozen per-layer allocation policy and packages its
physical cache implementation, exact-byte controls, preregistered evaluation
protocol, and labelled evidence artifacts. The current Python path is not a
fused low-bit recurrence kernel. RecurQuant does not claim authorship of
Qwen3.5 or its architecture, which remain credited to the Qwen team.
