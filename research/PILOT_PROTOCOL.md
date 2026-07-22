# Diagnostic pilot protocol v0.1

Status: frozen before model-level candidate tuning

Date: 2026-07-22

Purpose: determine whether a Gated DeltaNet-specific adaptive quantizer has
enough measurable headroom to justify implementation.

This is a diagnostic pilot, not the final confirmatory preregistration.

## Target

- Model: `Qwen/Qwen3.5-0.8B-Base`
- Revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Architecture expectation: 24 language layers, of which 18 are Gated DeltaNet.
- Expected state per linear-attention layer at batch one: `[1, 16, 128, 128]`
  in FP32, or 1 MiB.

Any mismatch stops the run until the protocol and adapter are reviewed.

## Fixed baseline order

1. FP32 persistent-state reference.
2. Static symmetric INT8, group size 128, round-to-nearest.
3. Static symmetric INT8, group size 128, stochastic rounding.
4. Static symmetric INT6, group size 128, round-to-nearest.
5. Static symmetric INT4, group size 128, round-to-nearest.
6. INT4 group-size ablation: 64 and 256.

Each state is quantized once after prefill and again after every single-token
teacher-forced decode step. Model weights, input tokens, convolution state, and
ordinary attention KV caches remain unchanged.

## Diagnostic inputs

Use three disjoint prompt groups:

- Calibration: tune no more than normalization and candidate thresholds.
- Development: select one candidate policy.
- Confirmation: untouched until the policy and pass gates are committed.

The first smoke artifact may use a short public-domain or synthetic sequence,
but it cannot support a research claim.

For candidate v0.1, the profiles are fixed as:

- `retrieval`: calibration.
- `code`: development.
- `multilingual`: untouched confirmation.

## Recorded metrics

- Mean, p95, p99, maximum, and worst-5% mean token KL.
- Top-1 token agreement.
- Reference and candidate negative log-likelihood.
- Per-layer relative state error and update ratio.
- Actual FP32 cache bytes and estimated packed bytes including FP16 scales.
- Model revision, package versions, device, seed, token hash, and artifact hash.

No speed or physical-memory result is recorded during QDQ emulation.

## Gate A: integration validity

All conditions must pass:

- Exactly 18 recurrent states are captured with the expected shape.
- A repeated deterministic baseline run produces identical canonical evidence.
- All output and state metrics are finite.
- INT8 round-to-nearest has no lower mean state error than INT6 or INT4 due to
  an implementation mistake on the same state.

## Gate B: allocation headroom

Before designing the adaptive policy, run a per-layer sensitivity oracle that
upgrades one layer at a time from INT4 to INT8.

Continue to a candidate only if, on at least two prompt groups:

- an equal-budget oracle allocation (mean no more than 4.5 payload bits/state
  element) reduces worst-5% token KL by at least 15% versus uniform INT4; and
- the direction of layer sensitivity is not reversed across every prompt group.

If this fails, stop the Gated DeltaNet-specific allocation hypothesis and move
to the separately planned CliffQuant project. Do not tune the threshold on the
confirmation prompts.

## Candidate phase

Gate B passed on the calibration and development traces. Candidate v0.1 is now
frozen before the multilingual confirmation run:

1. On the FP32-state calibration trace, emulate INT4 state storage.
2. For each Gated DeltaNet layer and cached decode token, compute
   `||(Q4(S) - S)^T q||_2 / max(||S^T q||_2, 1e-12)` using the normalized query.
3. Average this score over the calibration tokens.
4. Select exactly one layer with the highest score; exact ties select the lower
   layer index.
5. Store that layer at INT8 and all other Gated DeltaNet states at INT4, using
   group size 128, FP16 scales, and deterministic nearest rounding.

Calibration selected layer 0. Development independently ranked layer 0 highest.
The deployed confirmation plan is therefore fixed at layer 0 INT8 plus 17 layers
INT4, for an average 4.2222 payload bits per state element.

Simple mean decay, write-gate, update-norm, and committed-residual signals are
not part of candidate v0.1. Their correlations were weak or inconsistent in the
pilot and remain negative evidence.

## Gate C: untouched confirmation

Run the fixed plan once on the `multilingual` profile with 32 prefill and 32
teacher-forced decode tokens. Do not run a layer sweep on that profile first.

All conditions must pass versus uniform INT4:

- worst-5% token KL improves by at least 15%;
- mean token KL is lower;
- top-1 agreement is not lower;
- all metrics are finite; and
- a deterministic repeat has the same canonical evidence hash.

Passing Gate C permits a larger public-dataset preregistration. It does not by
itself validate novelty, real memory reduction, or speed.
