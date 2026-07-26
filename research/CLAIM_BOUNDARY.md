# Claim boundary

Last reviewed: 2026-07-26

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
- Query magnitude combined with quantization difficulty has already been used
  for mixed-precision KV-cache channel selection in
  [MixKVQ](https://arxiv.org/abs/2512.19206).
- Mixed precision and memory-budget allocation are broad existing ideas.
- Randomized Hadamard transforms and learned rotations are established
  quantization outlier-suppression techniques. RecurQuant cannot claim to have
  invented rotation-based or Hadamard-based quantization.

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

## Current v0.3 research boundary

Experiment 005 permanently failed its frozen numerical prerequisite before
holdout (`13/16 = 0.8125` sign agreement versus a required `0.95`). Its later
same-calibration adaptive-MSE postmortem was promising but achieved only a
7.93% descriptive excess-NLL reduction against the strongest static empirical
selector, with a paired 95% interval crossing zero. The ranked `[8, 16)`
holdout remained unopened.

Experiment 006 implemented deterministic rank fusion of offline
target-directional sensitivity and causal per-write reconstruction benefit at
the same exact byte budget. Its authenticated same-calibration GPU diagnostic
rejected the frozen equal-rank primary: macro excess NLL was `0.514873` versus
`0.493302` for plain adaptive MSE, and the paired interval crossed zero. Its
3.90% point improvement over the strongest static method was below the frozen
20% requirement, again with an interval crossing zero. Better predeclared
ablations remain exploratory and cannot be relabelled as the primary. The
candidate stopped before its numerical prerequisite and the ranked `[8, 16)`
holdout remained unopened. Experiment 006 supports no improvement,
generalization, novelty, deployment, or breakthrough claim.

Experiment 007 implemented CQER-32, a causal normalized-query-energy EMA times
exact INT4-to-INT8 reconstruction benefit for physically packed Gated DeltaNet
recurrent-state rows. Its authenticated same-calibration diagnostic improved
macro excess NLL by 6.18% over plain adaptive MSE and 13.62% over static
target-Fisher, but both paired intervals crossed zero. It failed the frozen
20% static-reduction requirement and the top-1 non-inferiority margin. The
packed state used exactly `2,564,096` bytes, while persistent selector EMA
state increased the resident total to `2,711,552` bytes. The FP64 prerequisite
was not reached and ranked `[8, 16)` remained unopened. Experiment 007 supports
no generalization, novelty, latency, deployment, state-of-the-art, or
breakthrough claim.

Experiment 008 tested CORA-C2 on a separate frozen `[16, 32)` development
window. CORA-C2 lowered macro excess NLL by 26.17% relative to static
target-Fisher and 21.62% relative to adaptive MSE, but it was 10.54% worse than
CQER-32. Raw CORA was 3.35% worse than CQER-32. Confirmation-2 reduced
normalized committed-mask churn by 79.99%, but worsened raw-CORA NLL by 6.96%,
above the frozen 1% limit. Five advancement checks failed, the independent
verifier was not reached, and ranked `[8, 16)` remained unopened. Experiment
008 therefore supports neither improvement over CQER-32 nor a general benefit
from its observability or two-hit mechanisms.

Observability-weighted quantization error is not a new principle. Classical
finite-wordlength state-space analysis expresses output noise using the
quantization-noise covariance and observability Gramian. CORA's diagonal
observability score times measured row distortion belongs to that same broad
mathematical family. Its Gated DeltaNet adaptation may still be a useful
engineering hypothesis, but the failed development result does not establish
it as a contribution.

Experiment 009 freezes RHT-CQER-32, a deterministic right-side randomized
Hadamard codec composed with CQER-32. Its one-task Stage-A screen passed every
frozen gate, reducing state SSE by `59.97%` and aligned excess NLL by `58.59%`
at equal packed-state and selector bytes. That result authorizes a new
development run; it does not establish generalization or confirmation. The
codec regenerates SHA-256-derived signs and materializes FP32 transform
workspaces in the current Python path. The result does not establish novelty or
speed because randomized Hadamard and rotation quantizers are prior art and the
implementation has no fused-kernel benchmark.

## Claims prohibited without new evidence

- "First recurrent-state quantization method."
- "First sub-8-bit recurrent model."
- "First update-aware recurrent cache."
- "Lossless" unless exact output equivalence is demonstrated in the stated mode.
- "Reduces total model memory by 7.36x." The measured ratio covers resident
  recurrent-state payloads and scales only.
- "Speeds up inference" without passing the separate latency gate under the
  stated benchmark.
- "Breakthrough," "state of the art," or "novel" based on the current evidence.
- Authorship of Qwen3.5 or any other base model.

## What this project contributes

RecurQuant packages a physical low-bit recurrent-state cache, a frozen per-layer
allocation policy, exact-byte controls, an independently checkable confirmation
verifier, a preregistered evaluation protocol, and labelled evidence artifacts.
Its scoped held-out result is reproducible without expanding the claim. The
current Python path is not a fused low-bit recurrence kernel. RecurQuant does
not claim authorship of Qwen3.5 or its architecture, which remain credited to
the Qwen team.
