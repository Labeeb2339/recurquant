# Prior-art audit

Last searched: 2026-07-22

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
  FP8 recurrent states with block scaling, stochastic rounding, checkpoints,
  and cached-input replay. Its reported INT8/FP8 checkpoint experiments were
  emulated while optimized 8-bit kernels were still under development. These
  methods remain prior art and mandatory stability baselines for RecurQuant.

## Existing Gated DeltaNet systems work

- [When Good Enough Is Optimal](https://arxiv.org/abs/2606.06034) uses low-
  precision approximations inside quantized Gated DeltaNet computation. Its
  focus differs from persistent-cache storage, but it precludes a broad "first
  quantized Gated DeltaNet" claim.
- [KVBuffer](https://arxiv.org/abs/2605.19049) buffers Gated DeltaNet updates to
  reduce state writes.
- [ReplaySSM](https://tridao.me/blog/2026/replayssm/) applies update/correction
  replay to Mamba-2 and Qwen3.5 Gated DeltaNet; its
  [implementation is public](https://github.com/Johnny-Liou/ReplaySSM).
- [HOLA](https://arxiv.org/abs/2607.02303) combines Gated DeltaNet state with a
  bounded exact KV cache and uses committed update magnitude as an importance
  signal. Update residuals and hybrid state/KV memory are therefore not novel.
- [AVMP](https://arxiv.org/abs/2605.22416) dynamically rebalances KV and SSM
  memory pools.

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

These works target transformer KV caches rather than Gated DeltaNet recurrent
matrices, but they preclude broad claims that sensitivity-guided, per-layer,
dynamic, or equal-budget cache precision allocation is new.

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

The scan did not locate a published implementation combining all three:

1. sub-8-bit **persistent** Gated DeltaNet cache storage;
2. Gated DeltaNet-specific read/sensitivity-based precision allocation; and
3. a packed end-to-end runtime kernel with measured quality and latency.

That negative search is not proof of firstness, and the current RecurQuant
alpha does not satisfy item 3: it physically stores the cache in packed form
but materializes one recurrent state for each layer call and has no fused
recurrence kernel or latency result.

The frozen v0.2 protocol compares uniform INT4, one uniform INT8 reference,
three same-byte random layer placements, MSE-selected placement, nearest and
stochastic rounding, and the prespecified layer-0 policy. It does **not**
compare against Q-Mamba's DSQ method, Nemotron checkpoint/replay, ReplaySSM,
KVBuffer, per-head/per-block allocation, or fused-kernel baselines. A later
claim of novelty or general superiority would require those comparisons,
multiple models and tasks, a repeated prior-art search, and measured end-to-end
latency.
