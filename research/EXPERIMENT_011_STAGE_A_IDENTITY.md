# Experiment 011 Stage-A identity

Date: 2026-07-31

I freeze Experiment 011 as a new one-run identity for the unchanged
StateLease-H5 Stage-A falsification screen. It is not a resume of Experiment
010 and it does not erase or supersede that experiment's administrative null.

## Scientific lineage

The complete Experiment 010 StateLease-H5 protocol at H0
`0e3dbcec2cb9cca1cdb062ec2491954ae052d7b9` is incorporated without scientific
change:

- protocol file SHA-256:
  `1c1517bb11939cbef4673f7a5890055d8092d519743b118543a3615a3a7c8208`;
- Stage-A identity-note SHA-256:
  `0bab7c8f416ce238071b9a87ed6b6dda6450d0e21265ee06ce5e47b1be36deb6`.

The method, model, task, formatting, token alignment, storage allocation,
comparators, metrics, thresholds, tie rules, gates, and claim boundaries are
unchanged. `EXPERIMENT_011_STATELEASE_PROTOCOL.md` amends only execution and
evidence safeguards: pre-seal readiness, authenticated local Arrow transport,
two-phase reservation and result publication, and a monotonic access ledger.

Experiment 010 remains sealed at
`c0ef99c924121b981d7bbda8ba4b9b76d3b14f51`. Its tracked
`evidence/experiment010-statelease-stage-a-administrative-null.json` has:

- canonical evidence SHA-256:
  `c5f779ed4fd5a48284e212dfaead9146cbd2bb0b53404a5628fd49bc74ee31f3`;
- file SHA-256:
  `2baa25005d4220f99ea784d21bce1c869311987b7ecc56cb9338f76c14b36d12`;
  and
- raw receipt SHA-256:
  `f53cbb53f043180d40e472cacda64397014b8a60ec065fabcb5c0738d53adc15`.

That evidence records no task-row access, tokenization, model-weight load,
forward pass, metric, aggregate, or quality result.

## Frozen task identity

Stage A may access only the already-open MBPP task 666:

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

The token-ID hashes use SHA-256 over
`recurquant.evidence.canonical_json_bytes(list[int])`. The formatter,
Qwen2Tokenizer behavior, special-token settings, and scoring alignment remain
exactly those authenticated by the incorporated Experiment 010 identity note.
No new task identity may be resolved.

## Frozen model and runtime identity

- Model: `Qwen/Qwen3.5-0.8B-Base`
- Revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Batch size: 1
- State geometry: 18 recurrent layers, 16 value heads per recurrent layer,
  128 key rows per head, and value width 128
- Persistent StateLease allocation: `3,454,664` bytes

The exact package versions are:

| Distribution | Version |
| --- | --- |
| datasets | `4.8.5` |
| fsspec | `2026.2.0` |
| huggingface-hub | `1.26.0` |
| numpy | `2.4.6` |
| pyarrow | `25.0.0` |
| safetensors | `0.8.0` |
| tokenizers | `0.22.2` |
| torch | `2.11.0+cu128` |
| transformers | `5.14.1` |

The canonical package-manifest SHA-256 is
`2466ad25043894fcd1604c97c373e5d5680061fdb7637f861b83d5c9465c31fe`.

The cached model resources are frozen by name, byte length, and SHA-256:

| Resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `model.safetensors.index.json` | 50,900 | `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a` |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |
| `tokenizer_config.json` | 16,712 | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
| `tokenizer.json` | 12,807,196 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

The cached MBPP revision is frozen without decoding task content:

| Resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `dataset_info.json` | 1,069 | `141cbe58ff5cb6fe53772f36a41520c1f7f3adda9f773848e11fa7a5bd40123c` |
| `mbpp-train.arrow` | 178,448 | `dbd85255cf0fad7b11f3b39233045a0ab1799c4fe51846ec57946e0abe59ed70` |

Before sealing, the evaluation process must import and version-check every
listed distribution, confirm that `datasets.load_dataset` is callable, confirm
CUDA and BF16 support, parse the pinned local configuration, and verify all
required cached model and MBPP resources. It must reject alternate recognized
weight and tokenizer resources in the authenticated snapshot so the loaders
can consume only the pinned tokenizer files, safetensors index, and shard. The
check may hash complete resource bytes but may not decode dataset rows,
instantiate the tokenizer, inspect safetensors metadata, or deserialize
tensors. This check is local-only.

The preflight may not call a dataset loader, read task 666, inspect a protected
row, instantiate or call the tokenizer, produce token IDs, load model weights,
run a forward pass, inspect logits, or calculate any quality metric. Those
operations remain ordered after the seal.

After the seal, the evaluator must select `task_id == 666` directly from the
authenticated local `mbpp-train.arrow` with PyArrow and materialize exactly
one row. Network access, `datasets.load_dataset`, streaming, cache fallback,
and alternate revisions are forbidden.

## H0, seal, and outputs

The eventual Experiment 011 H0 is not borrowed from Experiment 010. A new
Stage-0 artifact must authenticate the exact clean H0 that contains this
identity, the Experiment 011 protocol, evaluator, verifier, and tests. The
Stage-0 artifact must bind the full source closure and the runtime manifest
before preflight can authorize sealing.

The exact one-run marker is:

```text
RecurQuant-One-Run: experiment011-stage-a-task666-v1
```

The only Stage-A result and attempt-receipt paths are:

```text
artifacts/experiment011-statelease-stage-a-666.json
artifacts/experiment011-statelease-stage-a-666.attempt.json
```

Reservation is two-phase and fail-closed. Before moving `HEAD`, the evaluator
must durably write an exclusive `prepared_before_head_cas` receipt bound to the
exact proposed seal and all authenticated pre-seal identities. The prepared
receipt records zero completed tasks, no quality aggregate, and no automatic
rerun authorization. Only then may `HEAD` be compare-and-swapped from H0 to
that seal and the receipt status atomically promoted. Any interruption leaves
enough Git-plus-receipt evidence to classify the state without accessing task
text or silently creating another attempt.

Completion uses the same discipline. Before exclusive output publication, a
prepared-completion receipt must bind the already-verified result hashes and a
non-revealing canonical gate hash. Every gated aggregate must first be
recomputed from the authenticated per-token and per-layer/write records and
match its supplied summary under the original FP32 fidelity-summary semantics
for aligned NLL/top-1 and FP64 compensated semantics for trajectory. After
publication, the evaluator must reauthenticate the output before promoting the
receipt to completed. No failure path may relabel a prepared or published
valid result as a no-result failure, and no interrupted publication state
authorizes another Stage-A run.

All Git authentication, seal construction, and compare-and-swap operations
must use a sanitized process environment and the exact repository top-level,
Git directory, worktree, common directory, object directory, and replacement-
ref view authenticated before sealing. Privacy-safe canonical path hashes must
remain identical across critical phases. Repository-routing or object-view
environment overrides are forbidden.

Post-seal receipts use a monotonic access ledger. Once a task, tokenizer,
weight-loading, evaluation, or finalization hook is entered, later failure
records must preserve that fact or report a conservative unknown/lower bound;
they may not inherit the original all-zero access boundary.

The seal must have the new H0 as parent and an identical tree. The result must
be withheld until the complete authenticated run passes artifact-integrity,
privacy, and finiteness checks. An infrastructure receipt is not a scientific
result.

## Claim boundary

This identity record contains no StateLease quality observation. Stage A is
only a falsification screen and cannot support an improvement, novelty,
deployment, speed, state-of-the-art, or breakthrough claim.
