# Experiment 008 data-access clarification

> **Status: committed before any Experiment 008 model load, forward pass, or
> quality metric.**

The MBPP source is a streaming public split rather than a server-side keyed
database. Locating already-pinned task IDs therefore requires the dataset
loader to stream source records. A non-target record can transiently pass
through the loader even when its content is never retained or used.

For Experiment 008, "protected and unopened" has this enforceable meaning:

- ranked window `[8, 16)` is never selected, retained, canonicalized, hashed,
  formatted, tokenized, passed to the model, or evaluated;
- the evaluator loads only the exact pinned selector-prefix IDs and `[16, 32)`
  development IDs into application state;
- non-target stream records are inspected only for `task_id` and discarded;
- the legacy Experiment 005 route to `[8, 16)` is locked; and
- any task-ID, row-content, token-manifest, ordering, repository, or source-file
  mismatch fails before model loading.

This clarification fixes an implementation ambiguity in the frozen wording. It
does not reveal a protected prompt, token, logit, loss, or quality result and
does not change the method, tasks, thresholds, or advancement rule.

## Metric aggregation clarification

The protocol's `token CVaR95 KL` is computed within each task over that task's
scored tokens, then averaged with equal task weight. This is the
`macro_cvar95_kl` contract already used by Experiments 005-007 and prevents long
reference programs from dominating the task-macro evaluation. The artifact
also records the maximum KL over all task maxima as `maximum_kl`. The frozen
`0.10` CVaR margin applies to `macro_cvar95_kl`.
