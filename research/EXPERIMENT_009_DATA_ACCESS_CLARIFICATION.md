# Experiment 009 data-access clarification

> **Status: recorded after Stage A and before any Stage-B identity resolution,
> tokenizer load, model load, forward pass, or quality metric.**
>
> This clarification narrows ambiguous transport wording. It does not change
> the method, ranked windows, task count, metrics, thresholds, or advancement
> rule.

Recorded: 2026-07-26

## Why this is necessary

The pinned MBPP source is exposed through Hugging Face streaming, not a
server-side keyed API. The transport can deserialize a complete source record
before RecurQuant's loader receives its mapping. RecurQuant can control which
fields its code inspects and which rows enter application state, but it cannot
truthfully prove that the dataset transport never deserialized a non-target
record.

The Stage-A artifact contains the legacy field:

```text
protected_window_8_16_loaded_tokenized_or_evaluated = false
```

Its word `loaded` is too broad if read as transport-level deserialization. The
authenticated artifact remains immutable. For Experiment 009, that field is
interpreted only as the enforceable application-level boundary below.

## Enforceable protected-window boundary

Ranked MBPP window `[8, 16)` must never be:

- selected as a target row;
- retained as a row mapping in RecurQuant application state;
- canonicalized or content-hashed;
- formatted into a prompt/code example;
- tokenized;
- passed to a model;
- evaluated; or
- included in a quality or diagnostic artifact.

For non-target streaming records, RecurQuant may inspect only `task_id` and
must discard the record immediately. Dataset transport may already have
deserialized other fields; those fields must not be read by RecurQuant code.
Task IDs alone may be used to compute the frozen rank ordering.

This is the practical boundary required by a public streaming source:
protected task content and tokens remain unseen by the experiment, while
source traversal is acknowledged rather than hidden.

## Stage-B identity algorithm

The `[32, 64)` identity step must use two content-separated passes:

1. Stream the pinned training split and inspect only `task_id`.
2. Rank task IDs with the frozen `rq-v0.2` SHA-256 key and retain only the 32
   target IDs at ranks `[32, 64)`.
3. Stream again through the task-ID loader. Inspect only `task_id` for
   non-target records and canonicalize only the 32 target rows.
4. Verify their exact rank order, canonical row hashes, and content manifest.
5. Load the pinned tokenizer and tokenize only those 32 target rows.
6. Write and authenticate the identity artifact without loading model weights
   or computing logits, losses, state errors, or any other quality metric.

The implementation must have a fail-closed test mapping that raises if any
non-target field is read. It must also assert that selected, canonicalized,
formatted, tokenized, and evaluated task-ID sets are disjoint from ranked
window `[8, 16)`.

## Claim correction

The precise Stage-A statement is:

> Task `666` was the only row selected, retained, canonicalized, formatted,
> tokenized, passed to the model, or evaluated. RecurQuant code inspected only
> `task_id` on non-target stream records. The dataset transport may have
> deserialized complete non-target records.

No protected prompt, code, row hash, token sequence, logit, loss, or state
metric was retained or reported by Stage A.
