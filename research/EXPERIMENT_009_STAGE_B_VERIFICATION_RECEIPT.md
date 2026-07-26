# Experiment 009 Stage-B verification receipt

> **Status: the original Stage-B artifact is immutable and passes the
> corrected strict semantic verifier.**
>
> The correction changed only the validator's interpretation of canonical
> manifest order. It did not change the artifact, method, data, model run,
> metrics, thresholds, bootstrap, or gate decision.

Recorded: 2026-07-26

## Artifact under verification

```text
raw artifact:
experiment009-rht-cqer-stage-b-result-cdc603b.json

raw file SHA-256:
57b341d37871a52977b1ff89709864f3e6e0927154e5b2b9275b6f374953fe05

canonical evidence SHA-256:
2b15c732e894510f0421a22fcca9435e035dd15c4d3b50e2fcb733c0d1df58a8

evaluation commit:
8168c469b252bc9e707e51feaeccc3f940f190bb

corrected verifier commit:
2075154e642c39a14432adcc8ec32da679b534d3
```

The
[release manifest](../evidence/experiment009-rht-cqer-stage-b-result-manifest.json)
binds these identifiers to the compressed public artifact and its checksum.

## Run and interruption record

I kept three non-result events separate from the authenticated result:

1. An initial invocation used the wrong local Python environment. The frozen
   runtime contract rejected it before dataset access, tokenizer/model loading,
   or artifact creation.
2. A later shell invocation had a 14-second terminal timeout, which terminated
   the partial evaluator process. I verified that no Python process remained,
   no output artifact existed, and no quality metric had been exposed. I then
   restarted the identical frozen command as a hidden background process.
3. The completed evaluator atomically wrote the result artifact, then its
   post-write strict loader falsely rejected the valid canonical content
   manifest. The artifact already existed at the raw and canonical hashes
   above. I did not rerun the model or alter that artifact.

The wrong-runtime rejection and timed-out partial process revealed no result
that could be used for tuning or retry selection. The successful model
evaluation ran against the committed identity and exact frozen source at
commit `8168c469b252bc9e707e51feaeccc3f940f190bb`.

## Validator finding and correction

The artifact carries two intentional orders:

- task result records stay in the exact frozen ranked-window order; and
- the canonical content-manifest rows are sorted by integer task ID before
  hashing.

The original post-write loader incorrectly expected the canonical manifest in
ranked-window order. That contradicted the resolver's canonicalization rule
and caused a false rejection after a valid artifact had already been written.

Commit `2075154e642c39a14432adcc8ec32da679b534d3` corrected that one validator
assumption. The loader now independently requires:

- exact ranked-window order for task records;
- exact task-ID-sorted order for canonical manifest rows;
- exact `(task_id, row_sha256)` projection between those two representations;
- exact content-manifest, token-manifest, and ordered-identity hashes; and
- rejection of missing, duplicate, extra, reordered, or hash-mutated rows.

The fix therefore narrows acceptance to the protocol's actual canonical form;
it does not relax identity authentication.

## Independent audit

The corrected verifier authenticated the immutable artifact without executing
another model forward pass. The audit:

- recomputed the canonical wrapper hash and complete production schema;
- authenticated the Stage-A and Stage-B identity artifacts;
- compared every one of the 37 recorded source-file hashes with the exact file
  bytes at evaluation commit `8168c469b252bc9e707e51feaeccc3f940f190bb`;
- authenticated all imported repository modules and the clean, stable
  evaluation commit;
- revalidated the runtime, model revision, dataset access boundary, ordered
  tasks, token counts, row plan, selector bindings, sign schedule, physical
  storage, quotas, and state-record coverage;
- recomputed task and macro metrics, the 10,000-sample paired bootstrap, local
  state-SSE aggregates, all integrity checks, and all eight advancement checks;
  and
- passed the focused canonical-order and tamper test suite.

The strict loader accepts the original raw file SHA-256
`57b341d37871a52977b1ff89709864f3e6e0927154e5b2b9275b6f374953fe05`
and canonical evidence SHA-256
`2b15c732e894510f0421a22fcca9435e035dd15c4d3b50e2fcb733c0d1df58a8`.
No evidence field was reconstructed, deleted, reordered, or rewritten to make
the artifact pass.

## Verification boundary

This receipt establishes that the published artifact matches the frozen
Experiment 009 Stage-B protocol and that its positive gate decision is
recomputable from its recorded evidence. It does not turn a 32-task
development result into confirmation, establish novelty, measure speed, or
support a state-of-the-art, deployment, or breakthrough claim.
