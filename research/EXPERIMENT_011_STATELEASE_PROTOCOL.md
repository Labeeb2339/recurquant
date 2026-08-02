# Experiment 011: StateLease-H5 protocol

> **Status: frozen before Experiment 011 task access, tokenization, model-weight
> loading, forward passes, or quality metrics.**
>
> I carry the scientific design of Experiment 010 forward unchanged. This
> document amends only execution and evidence safeguards: fail-closed runtime
> readiness, authenticated local Arrow transport, two-phase reservation and
> result publication, and a monotonic access ledger.

Protocol frozen: 2026-07-31

## Experiment 010 disposition

Experiment 010 is complete as an administrative null, not a quality result. Its
one-run seal was created before an optional `datasets` import failed. The
failure occurred before `load_dataset` was called and before MBPP task 666 was
read, tokenized, passed to the model, or scored.

I preserve the following record without reset, deletion, reinterpretation, or
reuse:

- H0 commit:
  `0e3dbcec2cb9cca1cdb062ec2491954ae052d7b9`;
- one-run seal commit:
  `c0ef99c924121b981d7bbda8ba4b9b76d3b14f51`;
- administrative-null evidence:
  `evidence/experiment010-statelease-stage-a-administrative-null.json`;
- canonical evidence SHA-256:
  `c5f779ed4fd5a48284e212dfaead9146cbd2bb0b53404a5628fd49bc74ee31f3`;
- evidence-file SHA-256:
  `2baa25005d4220f99ea784d21bce1c869311987b7ecc56cb9338f76c14b36d12`;
  and
- raw failed-attempt receipt SHA-256:
  `f53cbb53f043180d40e472cacda64397014b8a60ec065fabcb5c0738d53adc15`.

Experiment 011 is a new authenticated experiment identity. It is not an
Experiment 010 resume, replacement receipt, or second attempt under the old
marker.

## Normative scientific carry-forward

The complete scientific protocol is incorporated by reference from
`research/EXPERIMENT_010_STATELEASE_PROTOCOL.md` as it existed at H0
`0e3dbcec2cb9cca1cdb062ec2491954ae052d7b9`, with file SHA-256
`1c1517bb11939cbef4673f7a5890055d8092d519743b118543a3615a3a7c8208`.
The Stage-A identity clarification is incorporated from
`research/EXPERIMENT_010_STAGE_A_IDENTITY.md` at the same H0, with file
SHA-256
`0bab7c8f416ce238071b9a87ed6b6dda6450d0e21265ee06ce5e47b1be36deb6`.

Those authenticated files remain normative for every scientific field not
overridden explicitly in this document. In particular, I do not change:

- the `StateLease-H5` method or
  `statelease_cut4_cut5_right_rht_query_ema32_weighted_mse_fisher_quota`
  implementation identity;
- the c4-versus-c5 controller, exact c5 tie rule, replay record, RHT-CQER-32
  checkpoint codec, causal query EMA, quota, seed, rounding, or prefill/decode
  boundary;
- Qwen/Qwen3.5-0.8B-Base revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, batch size one, recurrent-state
  geometry, or BF16/FP32 buffer dtypes;
- the exact `3,454,664`-byte allocation and `5.857110` resident bits per state
  element;
- the five equal-allocation fixed replay comparators, three equal-total-byte
  no-replay comparators, historical anchor, or off-budget references;
- MBPP task 666 or its prompt, target, formatting, token alignment, and
  protected-window boundary;
- any Stage-A or Stage-B metric, aggregation, uncertainty rule, comparator
  selection rule, threshold, advancement gate, interruption rule, or
  falsification rule; or
- the Stage-C confirmation, Stage-D kernel, prior-art, and public-claim
  boundaries.

Experiment-number substitutions are administrative only. They may name the new
seal, artifacts, receipts, and future Experiment 011 identity records, but they
must not change any literal domain, seed, digest rule, ordering rule, dataset
selection, scientific calculation, or gate incorporated above. If an
administrative substitution would change selected scientific data, the
authenticated Experiment 010 rule remains literal.

## New H0 and Stage 0

Experiment 010's Stage-0 artifact cannot authenticate Experiment 011. Before
the Stage-A seal, I must commit a new clean H0 containing the complete
Experiment 011 evaluator, verifier, tests, protocol, and identity records.

A newly generated Stage-0 artifact and sidecar must:

1. authenticate that exact H0 commit, tree, every required producer dependency,
   and the complete observed mapping from loaded `recurquant` module names to
   regular, non-symlink files inside the authenticated repository;
2. establish per-file HEAD-blob equality and a clean worktree, and independently
   reject `skip-worktree`, `assume-unchanged`, or any other non-canonical tracked
   index tag that could conceal drift from `git status`;
3. independently reproduce the synthetic algebra, storage inventory, codec,
   comparator, replay, transaction, privacy, and protected-window checks
   required by the incorporated protocol;
4. reconcile exactly 162 method-owned persistent tensor storages and
   `3,454,664` allocated bytes for StateLease-H5;
5. reject hidden FP16, BF16, FP32, FP64, integer, aliased, split, cache-level,
   or empty-view-backed persistent mirrors outside the declared schema; and
6. record the pinned runtime manifest defined below; and
7. write an exact LF-terminated SHA-256 sidecar, hash and deserialize the same
   immutable bytes read from one stable regular-file handle, and reauthenticate
   unchanged artifact and sidecar bytes after deserialization and again after
   independent semantic verification.

Stage 0 remains synthetic. The inherited integration audit initializes only
the fixed tiny random Qwen configuration and runs its two fixed synthetic
forwards; those forwards use no pretrained checkpoint, task text, target
tokens, or candidate quality data. Stage 0 may not access task text, tokenize
task 666, load pretrained or pinned Stage-A weights, run a Stage-A or
quality-model forward pass, or calculate a candidate quality metric. Any
Stage-0 or source-integrity failure stops Experiment 011 before sealing.

## Pre-seal dependency and runtime readiness

The final readiness check must execute in the same interpreter and environment
that will create the seal and continue into Stage A. It must be local-only and
must finish successfully before the one-run marker is committed.

The pinned package manifest is:

```json
{"datasets":"4.8.5","fsspec":"2026.2.0","huggingface-hub":"1.26.0","numpy":"2.4.6","pyarrow":"25.0.0","safetensors":"0.8.0","tokenizers":"0.22.2","torch":"2.11.0+cu128","transformers":"5.14.1"}
```

Its SHA-256 is
`2466ad25043894fcd1604c97c373e5d5680061fdb7637f861b83d5c9465c31fe`,
computed over UTF-8
`json.dumps(packages, sort_keys=True, separators=(",", ":")) + "\n"`.

Before sealing, the evaluator must:

1. import `datasets`, `fsspec`, `huggingface_hub`, `numpy`, `pyarrow`,
   `safetensors`, `tokenizers`, `torch`, and `transformers`, then fail unless
   every exact distribution version matches the manifest;
2. resolve `datasets.load_dataset` and fail unless it is callable, without
   calling it;
3. require `torch.cuda.is_available()`, at least one visible CUDA device, and
   BF16 support on the selected device;
4. resolve the pinned model revision from local cache only;
5. parse the local model configuration without loading model weights and fail
   unless the incorporated model type and recurrent geometry match; and
6. verify the presence, readability, and nonzero size of every locally cached
   tokenizer, configuration, index, and model-weight resource required by
   Stage A, reject every alternate recognized tokenizer or weight filename and
   shard set, and bind later loading to the authenticated snapshot and
   safetensors index, without instantiating the tokenizer, deserializing weight
   tensors, or deserializing a safetensors payload; and
7. hash, without decoding or iterating, the exact cached MBPP revision
   metadata and training Arrow resource required by the already-frozen row
   loader.

The required local model resources are:

| Resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `model.safetensors.index.json` | 50,900 | `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a` |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |
| `tokenizer_config.json` | 16,712 | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
| `tokenizer.json` | 12,807,196 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

The required privacy-safe MBPP cache identities under revision
`4bb6404fdc6cacfda99d4ac4205087b89d32030c` are:

| Resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `dataset_info.json` | 1,069 | `141cbe58ff5cb6fe53772f36a41520c1f7f3adda9f773848e11fa7a5bd40123c` |
| `mbpp-train.arrow` | 178,448 | `dbd85255cf0fad7b11f3b39233045a0ab1799c4fe51846ec57946e0abe59ed70` |

The readiness check may inspect import metadata, CUDA capabilities, the local
model configuration, cache paths, filenames, file metadata, and resource
presence. It may hash complete cached resource bytes to authenticate them, but
may not decode an Arrow row, inspect safetensors metadata, or deserialize a
weight tensor. It may not:

- call any dataset loader or iterate, inspect, canonicalize, or hash a dataset
  row;
- read task 666 or any protected MBPP row;
- instantiate or call the tokenizer, produce token IDs, or regenerate task
  hashes;
- deserialize, map, or otherwise load model weights;
- run a forward pass, inspect logits, reconstruct a model trajectory, or
  calculate any candidate metric; or
- reserve the one-run marker before every readiness condition has passed.

A pre-seal readiness failure exposes no scientific result and consumes no
one-run seal. It may be corrected only before sealing. After the final
readiness pass, the evaluator must recheck H0, Stage-0 integrity, source hashes,
and the runtime manifest immediately before atomically reserving the seal. A
runtime, source, dependency, model-cache, method, task, or identity change
after that point is not the authenticated Experiment 011 run.

Every Git subprocess used for authentication, commit construction, history
inspection, or compare-and-swap must receive a sanitized environment that
removes repository-routing, index, object-alternate, replacement-ref, and
configuration overrides. The evaluator must bind and repeatedly verify the
resolved repository top-level, Git directory, common directory, worktree,
object directory, and replacement-ref view before the seal and at end
integrity. Privacy-safe canonical path hashes must remain identical across
critical phases. Any mismatch fails closed.

## Stage-A identity and one-run outputs

Stage A uses only the already-open MBPP task 666 with the following immutable
identity:

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
RecurQuant-One-Run: experiment011-stage-a-task666-v1
```

The only primary Stage-A result and attempt-receipt paths are:

```text
artifacts/experiment011-statelease-stage-a-666.json
artifacts/experiment011-statelease-stage-a-666.attempt.json
```

The one-run reservation is a fail-closed two-phase transaction. The evaluator
must first create the proposed same-tree seal commit without moving a ref,
then exclusively create and `fsync` a canonical
`prepared_before_head_cas` attempt receipt. That receipt must bind the exact
H0, proposed seal commit, tree, seal-message hash, marker, output path, source
closure, Stage-0 identity, Experiment 010 administrative-null provenance,
runtime readiness, model configuration, and final pre-seal freshness receipt.
It must record zero completed tasks, no exposed quality aggregate, and no
automatic rerun authorization.

Only after authenticating that prepared receipt may the evaluator
compare-and-swap `HEAD` from H0 to the exact proposed seal. It may then
atomically promote only the receipt status to
`reserved_before_quality_data_or_model_weights`. Task-row access,
tokenization, weight loading, forward passes, and metrics remain forbidden
until the post-CAS seal and receipt validate together.

An interruption after prepared-receipt creation is not silently retryable. If
`HEAD` equals the proposed seal, the attempt is consumed even when status
promotion did not finish; the prepared receipt remains machine-readable
zero-result evidence. If `HEAD` still equals H0, no scientific run was
consumed, but normal execution must still stop. Recovery may only roll forward
the same authenticated proposed seal through a separately audited procedure or
freeze a new experiment identity; it may never create a different seal or
automatically run Stage A. The seal message independently binds the
pre-seal-freshness, output, zero-result, and claim-boundary fields so Git
preserves a second durable record.

Result publication is also fail-closed and two-phase. After all end-integrity,
privacy, finiteness, gate, and independent artifact-verification checks pass,
but before making the output public, the evaluator must durably replace the
attempt receipt with `result_prepared_before_output_publish`. That receipt
must bind the exact output-file and canonical-evidence hashes, a canonical hash
of the gate evidence without embedding its decision or quality scalars, task
ID, completed-task list, intended output path, and the fact that a quality
result has been computed. The evaluator may then publish the output
exclusively, reauthenticate its bytes and canonical evidence against the
prepared completion receipt, and atomically promote the receipt to
`completed_with_authenticated_stage_a_result`.

Before gate evaluation or publication, the evaluator must independently
recompute every gated aligned-NLL, top-1, and trajectory aggregate from the
complete authenticated per-token and per-layer/write record sets. The
recomputed values must agree exactly with their supplied summaries under the
incorporated FP32 `fidelity_summary` semantics for aligned metrics and the
FP64 compensated `TrajectoryNmseAccumulator` semantics for trajectory.
Presence, row count, ordering, and finiteness alone are insufficient. The gate
must consume these recomputed values, not a second unchecked summary.

The failure handler may never downgrade a prepared-completion receipt, a
completed receipt, or an existing independently valid output to
`failed_without_authenticated_stage_a_result`. If output publication succeeds
but final receipt promotion is interrupted, the valid output remains the
scientific record and the receipt must remain or be promoted to an explicit
interrupted-promotion state that records the result as available. If
publication fails first, the prepared completion receipt records a computed
but unpublished result. Neither condition authorizes another Stage-A run.

Every post-seal attempt receipt must also carry a monotonic access ledger for
task-loader entry and completion, tokenizer entry and completion, model-weight
loader entry and completion, evaluation entry and return, and finalization
entry. A failure record may claim that task data, tokenization, weights,
forwards, or a quality result were absent only when the ledger proves the
corresponding hook was never entered. After entry, the receipt must preserve
the completed fact or conservatively record attempted, in-progress, unknown,
or an evidence-backed lower bound; it may never reset that field to `false` or
zero. The original all-zero receipt boundary is valid only before task-loader
entry.

The seal commit must have the new H0 as its parent and the identical tree. The
evaluator must scan all refs and reflogs for the exact Experiment 011 marker
before creating it. Experiment 010's marker remains preserved and cannot be
renamed or counted as the Experiment 011 seal.

After sealing, the evaluator may access only task 666, revalidate every copied
text and token hash, load the pinned weights, and execute the one authenticated
Stage-A quality run. Artifact preparation, privacy checks, source and seal
revalidation, transactional publication, and interruption handling remain
exactly as incorporated from Experiment 010.

The post-seal row loader must consume only the already-authenticated local
`mbpp-train.arrow` resource. It must use a local PyArrow predicate on
`task_id == 666`, materialize exactly one matching row, and then apply the
incorporated row/text hashes. It may not call `datasets.load_dataset`, create a
streaming dataset, contact the Hub, or fall back to another cache or revision.
Zero or multiple matches fail closed.

## Unchanged Stage-A falsification gates

Stage A compares StateLease-H5 with FP32, original RHT-CQER, all five fixed
replay schedules, and all three equal-total-byte no-replay codecs. All of the
following are conjunctive and unchanged:

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
gate against it fails closed. A failure authenticates a Stage-A falsification
result for the unchanged candidate. A pass permits only the next frozen
development-identity step under the incorporated protocol.

## Claim boundary

Stage A is a cheap falsification screen. Whether it passes or fails, it cannot
support a public improvement, novelty, deployment, speed, state-of-the-art, or
breakthrough claim.

I will report only what the authenticated artifact establishes. I will not
describe the unchanged StateLease-H5 hypothesis as a breakthrough unless later
disjoint, multi-workload, cross-model evidence and the incorporated claim gates
actually support a narrower statement.
