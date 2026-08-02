# Prior-art audit

Last searched: 2026-08-02

This is a living claim boundary, not proof that no related work exists. Repeat
the search before any paper or novelty statement.

## Persistent recurrent-state quantization

- [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) evaluates
  `W8A8H4`, including 4-bit persistent Mamba2 state caches with decoupled
  state/channel scaling and selectivity reconstruction. RecurQuant cannot
  claim the first 4-bit or sub-8-bit persistent SSM-state quantizer.
- [Quamba2](https://arxiv.org/abs/2503.22879) and its
  [official implementation](https://github.com/enyac-group/Quamba) load and
  store calibrated Mamba cached states at INT8 in sequence kernels. Quamba2
  also searches mixed W4A8/A16 blocks, although that search is not the same as
  RecurQuant's per-recurrent-layer cache layout. RecurQuant cannot claim the
  first INT8 SSM cache, broad sensitivity-guided SSM quantization, or a first
  packed recurrent kernel.
- [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) studies recurrent rounding
  accumulation and uses FP16 stochastic rounding in its released deployment
  path.
- [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) evaluates FP16, INT8, and
  FP8 recurrent states with block scaling, stochastic rounding, and periodic
  quantized-state checkpointing plus cached-activation replay. It reports a
  fixed checkpoint period `CC = 8` on Nemotron 3 Super; those checkpoint
  experiments were emulated while optimized 8-bit kernels were still under
  development. Fixed checkpoint/replay is prior art and a mandatory
  Experiment 010 stability baseline.

## Existing Gated DeltaNet systems work

- [When Good Enough Is Optimal](https://arxiv.org/abs/2606.06034) uses low-
  precision approximations inside quantized Gated DeltaNet computation. Its
  focus differs from persistent-cache storage, but it precludes a broad "first
  quantized Gated DeltaNet" claim.
- [KVBuffer](https://arxiv.org/abs/2605.19049) buffers Gated DeltaNet updates to
  reduce state writes.
- [ReplaySSM](https://tridao.me/blog/2026/replayssm/) applies checkpoint plus
  cached-input replay to Mamba-2 and Qwen3.5 Gated DeltaNet; its Gated DeltaNet
  algorithm buffers the post-correction update, normalized key, and log decay
  as `(u, k, g)`. Its
  [implementation is public](https://github.com/Johnny-Liou/ReplaySSM).
  RecurQuant cannot claim the first Gated DeltaNet update buffer,
  checkpoint/replay cache, reduced state-write frequency, or Qwen3.5 replay
  implementation.
- [HOLA](https://arxiv.org/abs/2607.02303) combines Gated DeltaNet state with a
  bounded exact KV cache and uses committed update magnitude as an importance
  signal. Update residuals and hybrid state/KV memory are therefore not novel.
- [AVMP](https://arxiv.org/abs/2605.22416) dynamically rebalances KV and SSM
  memory pools.

## Transition- and observability-weighted error

- Classical finite-wordlength state-space analysis already propagates
  quantization-noise covariance to output noise through an observability
  Gramian. In particular,
  [Hilaire, Menard, and Sentieys (EUSIPCO 2007)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/b2l-g05.pdf)
  writes the state-noise contribution as `tr(Psi_X * W_o)`. With diagonal
  error covariance, this is the same broad algebraic family as weighting each
  row's distortion by a diagonal observability term. RecurQuant cannot claim
  to have invented observability-weighted quantization error.
- [WriteSAE](https://arxiv.org/abs/2605.12770) derives the Gated DeltaNet cache-
  write perturbation transition and its later-query/logit effect. It is an
  interpretability method rather than a quantizer, but it precludes claiming
  that the transition or downstream-influence principle is new.
- [TQS-PTQ](https://arxiv.org/abs/2606.13300) treats quantized rollout error as
  a dynamical-system trajectory-sensitivity problem and uses the score for
  mixed-precision allocation. It targets offline time-series model tensors,
  not causal Gated DeltaNet cache rows.
- Nemotron 3 Super and Ultra propagate recurrent-cache errors through products
  of state transitions. They do not use CORA's online row allocator, but they
  preclude a broad "first transition-aware recurrent-cache quantizer" claim.

## Mixed-precision cache allocation

- [KVTuner](https://arxiv.org/abs/2502.04420),
  [KVmix](https://arxiv.org/abs/2506.08018), and
  [Quantize What Counts](https://arxiv.org/abs/2502.15075) already use
  layer-wise sensitivity or loss signals to allocate cache precision under
  hardware or memory constraints.
- [RateQuant](https://arxiv.org/abs/2605.06675),
  [PM-KVQ](https://arxiv.org/abs/2505.18610), and
  [SpectrumKV](https://arxiv.org/abs/2606.08635) cover fixed-average-bit,
  progressive, or per-token mixed-precision KV policies.
- [MixKVQ](https://arxiv.org/abs/2512.19206) combines query relevance and
  intrinsic quantization difficulty for budgeted mixed-precision key-cache
  channels. It is the closest published analogue to multiplying a read-
  importance signal by physical quantization benefit.
- [Kitty](https://arxiv.org/abs/2511.18643) dynamically promotes a fixed quota
  of key-cache channels using runtime sensitivity and magnitude, with Triton
  kernels. It targets transformer KV caches rather than recurrent matrices.
- [OuroMamba](https://arxiv.org/abs/2503.10959) updates outlier-channel choices
  at every timestep and assigns higher precision dynamically for Vision-Mamba
  activations. It precludes a broad "first online dynamic mixed-precision SSM
  channel" claim.

These works target transformer KV caches rather than Gated DeltaNet recurrent
matrices, but they preclude broad claims that sensitivity-guided, per-layer,
dynamic, or equal-budget cache precision allocation is new.

## Runtime adaptation and residual correction

- [Runtime-Certified Bounded-Error Quantized
  Attention](https://arxiv.org/abs/2605.20868) computes online per-head,
  per-step error bounds for a tiered quantized KV cache and uses them for
  adaptive precision selection and deterministic fallback. It targets
  attention rather than a Gated DeltaNet state, but it precludes a broad first
  claim for runtime-checked adaptive cache precision or fallback.
- [Don't Waste Bits!](https://arxiv.org/abs/2604.04722) uses a learned
  token-level controller to choose among 2-bit, 4-bit, 8-bit, and FP16 KV-cache
  precision during decode. Dynamic per-token precision is prior art.
- [GEAR](https://arxiv.org/abs/2403.05527) combines low-bit KV-cache
  quantization with low-rank and sparse reconstruction-error correction.
- [TurboQuant](https://arxiv.org/abs/2504.19874) combines randomized rotation
  and scalar quantization with a quantized residual correction for inner
  products.

These methods do not establish Experiment 010's result, but they prohibit
broad claims that online risk selection, adaptive fallback, rotation, or
residual correction is new.

## Rotation and Hadamard codecs

- [QuIP#](https://arxiv.org/abs/2402.04396) uses incoherence processing with
  randomized Hadamard transforms for low-bit language-model quantization.
- [QuaRot](https://arxiv.org/abs/2404.00456) uses computational invariance and
  Hadamard rotations to remove activation outliers.
- [SpinQuant](https://arxiv.org/abs/2405.16406) learns rotations for improved
  LLM quantization, and
  [TurboQuant](https://arxiv.org/abs/2504.19874) develops fast randomized
  transforms for outlier suppression.
- [MambaQuant](https://arxiv.org/abs/2501.13484) applies rotation-based
  outlier suppression to an architecture adjacent to RecurQuant's recurrent
  state setting.

These methods make the transform principle established prior art.
RHT-CQER-32's narrow hypothesis is whether a fixed, reproducible right-side
transform composes usefully with physical causal row selection for one Gated
DeltaNet cache layout. A positive result cannot support a broad first,
rotation, Hadamard, or outlier-suppression novelty claim.

## Architecture evidence

- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) separates channel-wise
  erase and write gates. That distinction motivates analyzing
  the two state axes separately but does not establish precision allocation.
  Its [official code](https://github.com/NVlabs/GatedDeltaNet-2) uses the NVIDIA
  Source Code License-NC and is not copied into this Apache-2.0 repository.
- [Sparse Delta Memory](https://arxiv.org/abs/2607.07386) changes the architecture
  to use sparse reads and writes over a larger state. RecurQuant instead studies
  post-training storage of an existing model's state.

## Audited gap, not a novelty claim

The scan did not locate a published implementation combining all four:

1. sub-8-bit **persistent** Gated DeltaNet cache storage;
2. a causal transition-derived diagonal future-read score;
3. physical row-distortion allocation with confirmation-gated admissions; and
4. a packed end-to-end runtime kernel with measured quality and latency.

That negative search is not proof of firstness, and the current RecurQuant
alpha does not satisfy item 4: it physically stores the cache in packed form
but materializes one recurrent state for each layer call and has no fused
recurrence kernel or latency result.

Experiment 008 also failed its frozen development gate: CORA-C2 and raw CORA
were both worse on macro excess NLL than CQER-32. Exact-combination novelty
would not rescue a method that has not demonstrated the required quality.

Experiment 010's StateLease-H5 protocol therefore treats checkpoint/replay,
`(u, k, g)` buffering, rotations, dynamic precision, and residual correction
as prior art. Its frozen contribution question is narrower: whether choosing
between legal c4 and c5 handoff boundaries by direct local handoff distortion
improves reference-aligned trajectory drift and excess NLL for a physically
packed RHT-CQER Gated DeltaNet checkpoint at exactly `3,454,664` resident
bytes. No Experiment 010 quality result exists at protocol freeze.

The frozen v0.2 protocol compares uniform INT4, one uniform INT8 reference,
three same-byte random layer placements, MSE-selected placement, nearest and
stochastic rounding, and the prespecified layer-0 policy. It does **not**
compare against Q-Mamba's DSQ method, Nemotron checkpoint/replay, ReplaySSM,
KVBuffer, per-head/per-block allocation, or fused-kernel baselines. A later
claim of novelty or general superiority would require those comparisons,
multiple models and tasks, a repeated prior-art search, and measured end-to-end
latency.
