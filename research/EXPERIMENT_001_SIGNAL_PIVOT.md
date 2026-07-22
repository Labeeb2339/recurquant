# Experiment 001: from gate averages to read-error sensitivity

Date: 2026-07-22

Status: original hypothesis rejected; replacement passed diagnostic confirmation

## Question

Can inexpensive Gated DeltaNet dynamics identify which persistent recurrent
states require higher precision?

## Original hypothesis

Layers with larger average write gate, forgetting, state-update norm, or
committed update residual would benefit most when upgraded from INT4 to INT8.

The hypothesis was tested against an exhaustive single-layer intervention:
start with all 18 Gated DeltaNet states at INT4, upgrade one layer to INT8, then
measure the change in worst-5% token KL versus the FP32-state reference.

## Negative result

The simple signals did not rank measured sensitivity reliably.

| Signal | Retrieval Spearman | Code Spearman | Decision |
|---|---:|---:|---|
| Mean write gate `beta` | -0.104 | -0.133 | Reject |
| Mean forget activity | +0.216 | -0.309 | Reject |
| Relative state-update norm | -0.259 | -0.053 | Reject |
| Committed residual RMS | +0.121 | -0.408 | Reject |

The signs changed or remained weak. These signals are therefore not used in
candidate v0.1. HOLA-style update magnitude is also existing prior art, so
retuning it until it matched this pilot would be both methodologically weak and
unlikely to establish a distinct contribution.

## Diagnosis

Those statistics describe how the state is updated, but the model consumes the
state through a query-dependent read. A state can have modest update activity
yet be highly sensitive if its quantization error aligns with the next query.
Raw state MSE also treats every matrix direction as equally important.

## Replacement hypothesis

Measure the error at the Gated DeltaNet read boundary:

```text
read_risk = ||(Q4(S) - S)^T q||_2 / max(||S^T q||_2, 1e-12)
```

Here `S` is the FP32 persistent state, `Q4` is the fixed INT4 QDQ baseline, and
`q` is the normalized query used by the recurrent read. The score is measured
without mutating the reference cache and averaged over calibration tokens.

## Pilot evidence after the pivot

| Evidence | Retrieval | Code |
|---|---:|---:|
| Read-risk Spearman vs tail-KL improvement | +0.459 | +0.505 |
| Highest-risk layer predicted | 0 | 0 |
| Best layer from exhaustive intervention | 0 | 0 |
| Tail-KL reduction from layer 0 at INT8 | 79.78% | 62.16% |

The replacement consistently selected the oracle-best layer and correlated
better than raw state error. This repairs the immediate selector failure at the
calibration/development level; it does not yet prove generalization.

## Alternatives considered after the negative result

The failed gate proxies do not justify switching blindly to methods designed
for a different tensor class:

- **Empirical sensitivity:** retained. The exhaustive INT4-to-INT8 layer
  interventions measure downstream logit sensitivity directly, rather than
  approximating it from a parameter Hessian. A state-Jacobian, Fisher, or
  Taylor approximation is a useful future way to reduce the cost of that
  oracle, but it must differentiate with respect to the recurrent state—not
  the frozen model weights.
- **AWQ, GPTQ, and AutoRound:** useful weight-quantization baselines and design
  references, but not drop-in selectors for a persistent runtime state. They
  do not replace the current state intervention experiment.
- **Token-wise or block-wise mixed precision:** relevant follow-up work if the
  static candidate generalizes. It adds metadata, packing, and scheduling
  costs, so it is not introduced after seeing the pilot and before the frozen
  confirmation.
- **A learned precision controller:** deferred. Training a controller on these
  two short traces would add parameters and create an obvious overfitting path.
  It becomes defensible only with a larger calibration set and a separately
  held-out evaluation.

These choices preserve a simple falsifiable candidate while keeping the most
relevant ideas as registered follow-ups.

## Frozen candidate v0.1

- Calibration profile: retrieval.
- Select exactly one layer by mean read risk; ties favor the lower layer index.
- Selected layer: 0.
- Storage plan: layer 0 INT8, other 17 Gated DeltaNet layers INT4.
- Group size: 128; FP16 scale overhead modeled; nearest rounding.
- Average state payload: 4.2222 bits per element.
- Confirmation profile: untouched multilingual trace, 32 prefill plus 32 decode
  tokens.

## What would count as fixed

The pilot issue is considered fixed only if the frozen plan passes every Gate C
condition on the untouched profile and the canonical evidence reproduces. The
research question remains open until the method is evaluated on public tasks,
longer contexts, another model or checkpoint, stronger baselines, and a packed
runtime.

Candidate v0.1 passed those diagnostic conditions on 2026-07-22, and its repeat
matched the canonical evidence hash. The immediate selector failure is therefore
fixed for this pilot; the broader research question remains open. See
[Confirmation 001](CONFIRMATION_001.md).
