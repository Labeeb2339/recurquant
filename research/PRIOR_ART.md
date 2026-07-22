# Prior-art audit

Last searched: 2026-07-22

This is a living claim boundary, not proof that no related work exists. Repeat
the search before any paper or novelty statement.

## Persistent recurrent-state quantization

- [Quamba2](https://arxiv.org/abs/2503.22879) stores calibrated grouped Mamba
  cached states at INT8. RecurQuant cannot claim the first recurrent-state or
  first INT8 SSM cache quantizer.
- [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) studies recurrent rounding
  accumulation and stochastic rounding.
- [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) evaluates FP16, INT8, and
  FP8 recurrent states with block scaling, stochastic rounding, checkpoints,
  and activation replay. These are mandatory stability baselines, not new ideas
  available for RecurQuant to claim.

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

## Architecture evidence

- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) separates channel-wise
  erase and write gates. That distinction motivates analyzing
  the two state axes separately but does not establish precision allocation.
  Its [official code](https://github.com/NVlabs/GatedDeltaNet-2) uses the NVIDIA
  Source Code License-NC and is not copied into this Apache-2.0 repository.
- [Sparse Delta Memory](https://arxiv.org/abs/2607.07386) changes the architecture
  to use sparse reads and writes over a larger state. RecurQuant instead studies
  post-training storage of an existing model's state.

## Narrow lane under test

The scan did not locate a published implementation combining all three:

1. sub-8-bit **persistent** Gated DeltaNet cache storage;
2. Gated DeltaNet-specific read/sensitivity-based precision allocation; and
3. a packed end-to-end runtime kernel with measured quality and latency.

That negative search is not a novelty claim. A defensible result would still
need to beat uniform per-head/per-block quantization, Quamba2-style calibration,
Nemotron stochastic/checkpoint baselines, ReplaySSM/KVBuffer, magnitude and
update-residual scoring, and ordinary KV-cache quantization.
