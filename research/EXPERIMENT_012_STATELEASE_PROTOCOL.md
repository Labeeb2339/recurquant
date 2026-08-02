# Experiment 012: StateLease-H5 protocol

> **Status: frozen before Experiment 012 task access, tokenization,
> model-weight loading, forward passes, or quality metrics.**
>
> I carry the complete Experiment 011 scientific protocol forward unchanged.
> Experiment 012 creates a new administrative one-run identity, corrects one
> fail-closed runtime type attestation so it matches the fixed-replay factory
> that was already frozen, and closes pre-freeze authentication gaps without
> changing a scientific input, calculation, or gate.

Protocol frozen: 2026-07-31

## Experiment 011 disposition

Experiment 011 is complete as an infrastructure administrative null, not as a
quality result. I preserve its seal and receipt without reset, deletion,
reinterpretation, resume, or rerun:

- H0 commit:
  `827bcadacd6231e521f9e2f2ea92582dd4d68cef`;
- one-run seal commit:
  `0b236c4b46d54ece36f9518ef791a90cf113f0fe`;
- identical H0 and seal tree:
  `5596ed305246750da1bacc576002aae828acc045`;
- raw failed-attempt receipt SHA-256:
  `f7c7f68adf5078cbbe24b47d17f17fa1e2fdbbd4a9f6fc8229f8d2c7a5dcb9b4`;
  and
- privacy-safe failure-detail SHA-256:
  `4e714f51191d838888188dbfbb33eeb98ae54eaf5383f4ec89a9c05f3b093573`.

The authenticated access ledger records task-row access, tokenization,
model-weight loading, and evaluation entry. Evaluation did not durably return.
The forward-pass count and quality-result state are therefore unknown, not
zero or false. No result artifact or authenticated quality result exists, and
no rerun occurred or is authorized.

The failure was an infrastructure mismatch: the runtime attestation expected
method-specific subclass names, while the already-frozen factory constructs
the exact base `FixedReplayRecurrentStateCache` and distinguishes each
comparator through its policy and selection identity. This null is neither a
pass nor a scientific negative about StateLease-H5.

The finalized privacy-safe administrative-null records are:

- `evidence/experiment011-statelease-stage-a-administrative-null.json`;
- `research/EXPERIMENT_011_STAGE_A_ADMINISTRATIVE_NULL.md`;
- canonical evidence SHA-256:
  `c3ab1763502e3fac91166337dbc9fb536bd66258fbb17128f617911a2d6db387`;
- evidence-file SHA-256:
  `9a50de42c5e0eabad97798d90459aee938be2521a53a503c62159204fba308a2`;
  and
- narrative-file SHA-256:
  `0aace5364b92b06232526bacc70d30bc15c0fde2910d171c971052e42b337376`.

Experiment 012 is a new authenticated identity. It is not an Experiment 011
resume, replacement receipt, or second attempt under the old marker.

## Normative carry-forward

The complete Experiment 011 protocol is incorporated by reference from
`research/EXPERIMENT_011_STATELEASE_PROTOCOL.md`, file SHA-256
`29ad6a7d6c6eec243191a0d444a748219ed2ed12ab42f48e01af7316c8ab2737`.
Its Stage-A identity record is incorporated from
`research/EXPERIMENT_011_STAGE_A_IDENTITY.md`, file SHA-256
`9a1a855df14ba96e05bc948d016d1f360dadcdb5a510a15f02b87f26e4390536`.

Those authenticated files remain normative for every field not overridden
explicitly here. In particular, I do not change:

- the `StateLease-H5` method, c4-versus-c5 controller, exact c5 tie rule,
  replay record, RHT-CQER-32 checkpoint codec, causal query EMA, quota, seed,
  rounding, or prefill/decode boundary;
- Qwen/Qwen3.5-0.8B-Base revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, batch size one, recurrent-state
  geometry, or BF16/FP32 buffer dtypes;
- the exact `3,454,664`-byte StateLease allocation and `5.857110` resident
  bits per state element;
- the five equal-allocation fixed-replay comparators, three equal-total-byte
  no-replay comparators, historical anchor, or off-budget references;
- MBPP task 666, prompt, target, formatting, token alignment, or protected
  boundary;
- any metric, aggregation, uncertainty rule, comparator-selection rule,
  threshold, advancement gate, interruption rule, or falsification rule;
- fail-closed runtime readiness, authenticated local Arrow transport,
  two-phase reservation and result publication, monotonic access-ledger
  semantics, evidence recomputation, source authentication, or privacy
  boundary; or
- the Stage-B, Stage-C, Stage-D, prior-art, and public-claim boundaries.

Experiment-number, source-path, schema, provenance-bundle, receipt, and output
substitutions are administrative only. They create and authenticate the new
identity but may not change scientific inputs, calculations, or gates.

## Scientific evaluator correction

Beyond the administrative substitutions above, the only scientific
evaluator-logic amendment permitted by this protocol is the fixed-replay
runtime attestation. It must match the complete factory contract already
frozen in Experiment 011.

For each fixed-replay method in `fixed_cc1`, `fixed_cc2`, `fixed_cc4`,
`fixed_cc5`, and `fixed_cut4_in5`, the evaluator must require conjunctively:

1. `type(cache) is FixedReplayRecurrentStateCache`;
2. `type(cache.policy) is FixedReplayPolicy`;
3. `cache.policy == fixed_replay_policy(method)`, including its exact mode,
   checkpoint period, replay capacity, tail-retention, equal-allocation, and
   off-budget fields;
4. `cache.selection_method ==`
   `f"{method}_right_rht_query_ema32_weighted_mse_fisher_quota"`;
5. `cache.experiment_identity_sha256 ==`
   `EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256`;
6. `experiment010_statelease_effective_plan_sha256(cache.plan) ==`
   `EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256`;
7. `cache.plan` has exact `ExactBudgetRowPlan` type and is field-for-field and
   canonical-payload identical to the authenticated preflight selector plan;
   and
8. every recurrent layer has exact `FixedReplayLinearAttentionLayer` and
   `_StateLeaseStateView` types, the exact full frozen policy and replay
   capacity, the same exact selection identity as its cache, and
   `high_precision_group_indices` exactly equal to the sorted groups for that
   layer in the authenticated selector plan.

The check is exact and fail-closed. A subclass, proxy, alternate policy field,
altered plan or experiment identity, within-layer selected-row swap, layer
group-tuple drift, or alternate cache or layer selection string is rejected.
The effective-plan hash remains a storage-and-quota identity; full canonical
plan equality is the independent row-location identity. The existing exact
StateLease candidate type,
recurrent-layer type, state-view type, packed-tensor type, geometry, dtype,
allocation, storage-closure, and no-hidden-mirror checks stay unchanged.

This correction changes no cache construction, state value, checkpoint
schedule, replay capacity, token, logit, comparator, metric, threshold, or
gate. It only attests the runtime object identity that the frozen factory
actually returns. Any other scientific evaluator change requires another
protocol identity.

## Pre-freeze authentication hardening

The Experiment 012 implementation also closes integrity gaps discovered before
H0. These checks cannot select, tune, or change an evaluated value:

1. every loaded `recurquant` module name must resolve to its canonical,
   authenticated in-repository source path, every required core module must be
   present, and helper scripts must load only from their exact repository
   paths;
2. every repository snapshot must reject hidden index flags, unsafe local Git
   configuration, replacement or alternate object views, and any change to
   the authenticated local-config digest;
3. the Experiment 011 administrative-null note, identity, protocol, evidence,
   raw receipt, H0, seal, and absent result must remain independently
   hash-bound; and
4. any filesystem entry at a path that must be absent, including a dangling
   symlink, counts as present and fails closed.

The exact selector-plan and per-layer group-tuple checks apply to StateLease
itself as well as every fixed-replay comparator. They authenticate the objects
created from the already-frozen plan; they do not select a new plan or alter an
evaluated value. Each candidate factory receives an independent copy while
storage attestation retains a separate expected-plan snapshot. Artifact
construction must then require the exact-plan verification flag and the
canonical digest derived from the authenticated preflight plan.

These are authentication rules, not new scientific degrees of freedom. A
failure stops execution before it can be interpreted as model-quality
evidence.

## New H0 and Stage 0

Experiment 011's Stage-0 artifact cannot authenticate Experiment 012. Before
the Stage-A seal, I must commit a clean Experiment 012 H0 containing the
complete evaluator, verifier, tests, protocol, identity, and finalized
Experiment 011 administrative-null records.

The H0 commit and tree can be established only after those exact source and
evidence bytes are committed. The new Stage-0 artifact and pre-seal receipt
must bind the resulting identities; this document does not guess them in
advance.

The new production artifact path and schema are:

```text
artifacts/experiment012_stage0_production.pt
artifacts/experiment012_stage0_production.pt.sha256
recurquant.experiment012.stage0.production.v1
```

Stage A accepts only those exact resolved repository artifact and sidecar paths
as regular, non-symlink files. Copies, alternate filenames, alternate
sidecars, and symlink targets fail closed even if their bytes are identical.

The artifact and independent verifier must authenticate that future clean H0,
the complete observed source closure, exact HEAD blobs, canonical Git view,
runtime manifest, synthetic algebra, exact codec and comparator behavior,
transaction and privacy rules, and exactly 162 StateLease-owned persistent
tensor storages totaling `3,454,664` bytes with zero unexplained storage.

Stage 0 remains synthetic. It may not access task 666, instantiate or call the
tokenizer, load pinned Stage-A weights, run a quality-model forward, or compute
a candidate metric. All Experiment 011 pre-seal readiness, local-resource
hashes, package versions, Git sanitization, stable-file verification, and
independent verification requirements remain unchanged.

## Stage-A identity and one-run boundary

Stage A may access only the inherited MBPP task 666 identity:

- dataset-row SHA-256:
  `b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9`;
- prompt-text SHA-256:
  `b6f0f93b9d15b96ac42bbabbdb349a09d2d24e57667d47cafe900c1ea91fd64b`;
- code-text SHA-256:
  `d2701e79ccd968c9e5af78474af16256f3bbf39cdfedbec2199ac92e1a4f397e`;
- prompt-token-ID SHA-256:
  `729215c4c99cdf96b13ad73f6ac7b537ddf9e882409b77e479d609aee046bffa`;
- code-token-ID SHA-256:
  `a920370c4892513c8a5cdb9f88a33fd95d4c90201af39fdb7d517f3ad42a9d9a`;
  and
- token counts: 69 prompt, 39 code, and 38 aligned scored tokens.

The new exact one-run marker is:

```text
RecurQuant-One-Run: experiment012-stage-a-task666-v1
```

The only primary Stage-A result and attempt-receipt paths are:

```text
artifacts/experiment012-statelease-stage-a-666.json
artifacts/experiment012-statelease-stage-a-666.attempt.json
```

The seal must have the future clean Experiment 012 H0 as its parent and the
identical tree. The evaluator must prove that the exact Experiment 012 marker
does not already exist in any ref or reflog before two-phase reservation.
Experiment 010 and 011 markers remain preserved and do not count as the new
seal.

Reservation, post-seal access ordering, monotonic receipt updates, raw-evidence
recomputation, end-integrity checks, two-phase result publication, interrupted
publication handling, and no-rerun rules remain exactly those incorporated
from Experiment 011. In particular, a hook already entered may never later be
reported as false or zero without evidence; an unknown state remains unknown.

## Unchanged Stage-A falsification gates

Stage A compares StateLease-H5 with FP32, original RHT-CQER, all five fixed
replay schedules, and all three equal-total-byte no-replay codecs. All of the
following remain conjunctive:

1. every Stage-0 and artifact-integrity check passes;
2. exact allocated bytes equal `3,454,664`;
3. every StateLease interval is four or five and every tie selects c5;
4. aligned excess NLL is at least 10% lower than equal-allocation `fixed_cc1`;
5. aligned excess NLL is no more than 5% worse than the strongest fixed replay
   comparator;
6. reference-aligned trajectory NMSE AUC is lower than `fixed_cc1`;
7. task-macro top-1 agreement trails the best fixed comparator by no more than
   `0.01`; and
8. all primary values are finite.

If a comparator's excess NLL is zero or negative, the relative-improvement
gate against it fails closed. A failure is an authenticated Stage-A
falsification result for the unchanged candidate. A pass permits only the next
already-defined development-identity step; it is not confirmation.

## Claim boundary

This protocol contains no Experiment 012 quality observation. Stage A is a
single-task falsification screen. Whether it passes or fails, it cannot support
an improvement, novelty, deployment, speed, state-of-the-art, or breakthrough
claim.
