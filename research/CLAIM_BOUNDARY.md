# Claim boundary

Last reviewed: 2026-07-22

## What is already established

- Weight-only and activation quantization for language models are mature fields.
- SSM persistent/cached states have already been quantized, including 8-bit
  cached-state work in [Quamba2](https://arxiv.org/abs/2503.22879).
- Stochastic rounding, periodic state checkpoints, activation/update replay, and
  fewer persistent-state writes exist in current recurrent-model systems work.
- Update residuals have been used as importance signals for bounded auxiliary
  memory in Gated DeltaNet.
- Mixed precision and memory-budget allocation are broad existing ideas.

## Hypothesis that remains testable

At an equal modeled persistent-state bit budget, Gated DeltaNet-specific
precision allocation may reduce tail output divergence relative to uniform
state quantization. Candidate signals may include per-layer decay, write gate,
state-update magnitude, and observed error accumulation.

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

## Authorship if the project succeeds

The defensible contribution would be the allocation method, implementation,
evaluation protocol, and any clearly labelled derivative artifact. Qwen3.5 and
its architecture remain credited to the Qwen team.
