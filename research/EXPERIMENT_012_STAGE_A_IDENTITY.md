# Experiment 012 Stage-A identity

Date: 2026-07-31

I freeze Experiment 012 as a new one-run administrative identity for the
unchanged StateLease-H5 Stage-A falsification screen. It is not a resume or
rerun of Experiment 011 and does not erase or supersede that experiment's
infrastructure administrative null.

## Scientific lineage

I incorporate the complete Experiment 011 records without scientific change:

- `research/EXPERIMENT_011_STATELEASE_PROTOCOL.md`, SHA-256
  `29ad6a7d6c6eec243191a0d444a748219ed2ed12ab42f48e01af7316c8ab2737`;
  and
- `research/EXPERIMENT_011_STAGE_A_IDENTITY.md`, SHA-256
  `9a1a855df14ba96e05bc948d016d1f360dadcdb5a510a15f02b87f26e4390536`.

The method, model, task, formatting, token alignment, storage allocation,
comparators, metrics, thresholds, tie rules, gates, transaction semantics, and
claim boundaries are unchanged. Experiment-number, source-path, schema,
provenance-bundle, receipt, and output substitutions are required to create
the new administrative identity. Beyond those substitutions, the sole
scientific evaluator-logic correction is the exact fixed-replay runtime
attestation described below. Pre-freeze source, Git, prior-record, path, and
absence checks are strengthened only to fail closed.

## Experiment 011 boundary

I preserve the following authenticated Experiment 011 facts:

- H0 commit:
  `827bcadacd6231e521f9e2f2ea92582dd4d68cef`;
- one-run seal commit:
  `0b236c4b46d54ece36f9518ef791a90cf113f0fe`;
- identical H0 and seal tree:
  `5596ed305246750da1bacc576002aae828acc045`;
- raw failed-attempt receipt SHA-256:
  `f7c7f68adf5078cbbe24b47d17f17fa1e2fdbbd4a9f6fc8229f8d2c7a5dcb9b4`;
  and
- failure-detail SHA-256:
  `4e714f51191d838888188dbfbb33eeb98ae54eaf5383f4ec89a9c05f3b093573`.

Task 666, the tokenizer, and the pinned model weights were accessed.
Evaluation entered but did not durably return. Forward-pass count and
quality-result state are unknown. No result artifact or authenticated quality
result exists, and no rerun occurred or is authorized.

The finalized tracked administrative-null records are:

- `evidence/experiment011-statelease-stage-a-administrative-null.json`
  canonical evidence SHA-256:
  `c3ab1763502e3fac91166337dbc9fb536bd66258fbb17128f617911a2d6db387`;
- the same file's SHA-256:
  `9a50de42c5e0eabad97798d90459aee938be2521a53a503c62159204fba308a2`;
  and
- `research/EXPERIMENT_011_STAGE_A_ADMINISTRATIVE_NULL.md` SHA-256:
  `0aace5364b92b06232526bacc70d30bc15c0fde2910d171c971052e42b337376`.

## Scientific evaluator correction

The already-frozen fixed-replay factory returns the exact base
`FixedReplayRecurrentStateCache` for every parameterized fixed comparator. It
does not return the method-specific subclasses that the Experiment 011
runtime check incorrectly expected.

For each method in `fixed_cc1`, `fixed_cc2`, `fixed_cc4`, `fixed_cc5`, and
`fixed_cut4_in5`, Experiment 012 must attest the complete frozen factory
contract:

```text
type(cache) is FixedReplayRecurrentStateCache
type(cache.policy) is FixedReplayPolicy
cache.policy == fixed_replay_policy(method)
cache.selection_method ==
    f"{method}_right_rht_query_ema32_weighted_mse_fisher_quota"
cache.experiment_identity_sha256 ==
    EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256
experiment010_statelease_effective_plan_sha256(cache.plan) ==
    EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256
cache.plan == authenticated_selector_plan
canonical_json_bytes(asdict(cache.plan)) ==
    canonical_json_bytes(asdict(authenticated_selector_plan))

for every fixed recurrent layer:
    type(layer) is FixedReplayLinearAttentionLayer
    type(layer.policy) is FixedReplayPolicy
    layer.policy == fixed_replay_policy(method)
    layer.replay_capacity == fixed_replay_policy(method).replay_capacity
    layer.selection_method == cache.selection_method
    layer.high_precision_group_indices ==
        tuple(sorted(authenticated_selector_plan.groups_for_layer(layer_index)))
```

The full policy equality covers mode, checkpoint period, replay capacity,
tail-retention behavior, equal-allocation status, and off-budget status.
The inherited effective-plan hash authenticates storage fields and per-layer
quotas; it is not treated as a full row-location identity. Exact dataclass and
canonical-payload equality separately bind every selected `(layer, head, row)`
location. Subclasses, alternate modes or schedules, altered plan identities,
within-layer row swaps, and alternate cache or layer selection identities fail
closed.

Candidate and comparator construction, state updates, checkpoint and replay
schedules, allocations, storage calculations, logits, metrics, and gates are
scientifically unchanged. Experiment 012 administrative receipts and
provenance fields necessarily differ because they bind a new identity; they
do not change a scientific input or calculation.

## Pre-freeze authentication hardening

Before H0, Experiment 012 also closes audit-discovered integrity gaps:

- every loaded `recurquant` module and evaluator helper script must resolve to
  its canonical authenticated repository path, with the complete required
  module set present;
- StateLease and every fixed-replay comparator must carry the exact
  authenticated selector plan, exact factory selection and experiment
  identities, and exact per-layer high-precision group tuples; the evaluator
  snapshots the expected plan independently before construction and
  reauthenticates its digest when building the serialized result;
- every repository snapshot rejects hidden index flags and unsafe local Git
  configuration and binds the local-config digest;
- the complete Experiment 011 administrative-null lineage is checked against
  its declared file hashes; and
- an output, receipt, or historical result path is considered occupied when
  any filesystem entry exists there, including a dangling symlink.

These checks cannot alter a state, token, logit, comparator, metric, threshold,
or gate. They only prevent unauthenticated execution or ambiguous absence from
being recorded as scientific evidence.

## Frozen task, model, and allocation

Stage A may access only MBPP task 666:

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

The token-ID hashes retain the Experiment 011 canonicalization, formatter,
Qwen2Tokenizer behavior, special-token settings, and scoring alignment.

The model and allocation remain:

- model: `Qwen/Qwen3.5-0.8B-Base`;
- revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`;
- batch size: 1;
- state geometry: 18 recurrent layers, 16 value heads per recurrent layer,
  128 key rows per head, and value width 128;
- persistent StateLease allocation: `3,454,664` bytes; and
- resident precision: `5.857110` bits per state element.

The exact runtime-package manifest, model files, tokenizer files, local MBPP
Arrow files, byte lengths, and hashes are unchanged from the incorporated
Experiment 011 identity. Preflight remains local-only and must not access the
protected task row, instantiate the tokenizer, deserialize weights, run a
quality-model forward, or compute a candidate metric before sealing.

## New H0, Stage 0, seal, and outputs

Experiment 012 requires a new clean H0 and a newly generated, independently
verified Stage-0 artifact. The H0 commit, tree, and artifact hash must be
derived from the final bytes and bound by Stage 0 and the pre-seal receipt;
they are not guessed in this identity record.

- Stage-0 schema: `recurquant.experiment012.stage0.production.v1`;
- Stage-0 artifact:
  `artifacts/experiment012_stage0_production.pt`; and
- Stage-0 sidecar:
  `artifacts/experiment012_stage0_production.pt.sha256`.

Stage A accepts only those exact resolved repository paths as regular,
non-symlink files. A copied artifact, alternate filename, alternate sidecar,
or symlink target fails closed even when its bytes match.

The exact one-run marker is:

```text
RecurQuant-One-Run: experiment012-stage-a-task666-v1
```

The only primary Stage-A result and attempt-receipt paths are:

```text
artifacts/experiment012-statelease-stage-a-666.json
artifacts/experiment012-statelease-stage-a-666.attempt.json
```

The seal must have the final clean Experiment 012 H0 as parent and an
identical tree. Two-phase reservation, post-seal access ordering, monotonic
receipt facts, independent raw-evidence recomputation, end-integrity checks,
two-phase result publication, interruption handling, and the no-rerun rule are
unchanged from Experiment 011.

No task row, tokenizer, model weights, quality-model forward, or candidate
metric may be accessed before the new seal. No failed, prepared, or published
state may be downgraded to hide an entered hook or an available result.

## Claim boundary

This identity contains no Experiment 012 quality observation. Stage A remains
a single-task falsification screen. It cannot support an improvement, novelty,
deployment, speed, state-of-the-art, or breakthrough claim.
