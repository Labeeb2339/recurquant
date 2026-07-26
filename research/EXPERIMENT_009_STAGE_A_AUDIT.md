# Experiment 009 Stage-A post-result audit

> **Status: the authenticated Stage-A result is unchanged; its verifier and
> Stage-B handoff were hardened before any Stage-B identity, tokenizer, model,
> or quality access.**

Recorded: 2026-07-26

## Immutable result

The Stage-A artifact remains byte-for-byte unchanged:

```text
evidence/experiment009-rht-cqer-stage-a-666-5be8d48.json
file SHA-256:
98a432843dc438f2d5fde34f8704f154ebc3ee12c93ba7c469369acfedfb15b5
canonical evidence SHA-256:
9e03a1e8cefb5801406a47a2e5e365686afb0a05e10e099a989cee616b505ed1
```

The audit did not change the method, sign seed, task, thresholds, metrics,
model, tokenizer, selector artifacts, row quotas, or recorded measurements.
It rechecked the existing evidence and strengthened what future code must
recompute before accepting it.

## Findings and corrections

The first verifier trusted several internally recorded summaries more than it
should have. The hardened verifier now independently recomputes and
cross-checks:

- candidate-minus-reference excess NLL and the shared reference baseline;
- KL ordering, probability bounds, NLL non-negativity, and finiteness;
- aggregate, per-write, and per-layer state SSE, MSE, element counts, layer
  order, write order, and exact recurrent-state shape;
- FP32 recurrent source-state provenance;
- per-layer Q8 quotas, selected-row counts, exact selection method, and every
  stage/consume handshake;
- final physical precision-mask hashes against selector diagnostics; and
- exact packed-state and selector-aware byte accounting.

Mutation tests now demonstrate that coherent changes to these fields are
rejected instead of being accepted through a self-consistent wrapper hash.
The historical artifact still passes the hardened verifier.

The original phrase “independent numeric evidence” was also too strong for a
production implementation checking itself. That check is now described as
deterministic production-code self-consistency. A separate dense NumPy
implementation independently derives the SHA-256 sign schedule, constructs
the Hadamard transform, applies the FP16-scale Q4/Q8 quantizer, decodes the
state, and compares its outputs with the production implementation.

## Data-access clarification

The public Hugging Face streaming transport may deserialize complete source
records before yielding them. The enforceable boundary is therefore
application-level: RecurQuant reads only `task_id` on non-target rows and does
not retain, canonicalize, format, tokenize, model, evaluate, or report their
content. Ranked MBPP window `[8, 16)` remained outside every protected
application content set.

The immutable Stage-A artifact predates this wording and contains a legacy
field using the broader word `loaded`. Its exact interpretation is fixed in
[`EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md`](EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md);
the artifact was not silently rewritten.

## Runtime precision

The model weights and forward passes used CUDA bfloat16. The recurrent
source/reference state used for quantization and state-error measurement was
FP32. Both facts are now stated separately in the result and checked by the
handoff.

## Decision boundary

The one-task Stage-A pass remains a falsification-screen result. It authorizes
only the separately frozen Stage-B development run. It is not confirmation,
novelty, speed, deployment, state-of-the-art, or breakthrough evidence.
