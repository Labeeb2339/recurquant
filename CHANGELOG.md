# Changelog

This file records user-visible package changes. RecurQuant is pre-release
software, so compatibility can still change between development versions.

## [0.2.0a1] - 2026-07-22

### Added

- Physical INT4 nibble packing and INT8 payload storage for persistent Gated
  DeltaNet recurrent states, with grouped FP16 or FP32 scales.
- `PackedRecurrentStateCache` for keeping recurrent states packed between layer
  calls, with opt-in evidence recording and exact resident-byte accounting.
- A guarded Qwen3.5 cache factory, pinned Qwen3.5 quickstart, and compatibility
  checks for the tested Transformers release and eager, single-device
  inference path.
- A reusable frozen v0.2 mixed-policy cache helper and an installed
  `recurquant qwen35` workflow shared with the source-tree quickstart. Uniform
  INT4 remains available only as an explicitly named stress baseline.
- Machine-readable `recurquant qwen35 --json` output, a one-click Colab
  notebook, and a hash-anchored reproduction and extension guide.
- Offline `recurquant verify-artifact` checks for strict JSON, whole-file
  SHA256 anchors, and canonical evidence hashes.
- `recurquant verify-confirmation` independently checks the frozen MBPP
  manifest, artifact, quality gates, and optional raw checkpoint arrays. It
  distinguishes verified pass/fail outcomes from unanchored evidence.
- Frozen MBPP calibration and development evaluation workflows with pinned
  dataset/model revisions, prepared token manifests, canonical evidence hashes,
  equal-byte baselines, and resumable per-task checkpoints.
- Held-out confirmation evidence on all 500 frozen MBPP test tasks and 30,244
  teacher-forced reference-code tokens. The mixed layer-0 INT8 plus 17-layer
  INT4 layout reduced task-macro excess NLL by 72.75% relative to uniform INT4
  while using exactly 2,564,096 resident recurrent-state bytes; every
  preregistered quality gate passed.
- A compressed release attachment containing the retained raw confirmation
  checkpoint for independent reconstruction of every reported gate.
- Unit coverage for packing parity, byte accounting, cache integration, public
  data split discipline, and the supported Qwen3.5 factory path.

### Changed

- Scale-storage emulation now uses the declared physical FP16 or FP32 format;
  superseding diagnostic results retain the earlier record instead of rewriting
  it.
- The Transformers dependency is pinned to exactly `5.14.1` while the alpha
  package depends on that release's internal linear-attention cache contract.
- FP16 scale storage is identified as the evaluated default. FP32 scale storage
  remains supported for experiments but is not covered by full-model evidence.

### Fixed

- Beam-cache reordering now permutes packed INT4/INT8 payloads and scales
  directly instead of dequantizing and requantizing the recurrent state.

### Known limitations

- The current Python path materializes one recurrent state while its layer
  executes. It does not establish faster inference, lower whole-model memory,
  or lower peak CUDA memory.
- Full-model evidence is currently limited to the pinned
  `Qwen/Qwen3.5-0.8B-Base` revision and the environment described in
  [docs/compatibility.md](docs/compatibility.md).
- The held-out result is limited to teacher-forced likelihood on the pinned
  Qwen3.5 checkpoint and MBPP construction. It does not establish generated-code
  correctness, novelty, cross-model generality, or a breakthrough.
