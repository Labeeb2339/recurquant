# Experiment 013: static RHT-Q468 packed-native adoption protocol

> **Status: fifth replacement-H0 candidate after a preserved pre-model,
> pre-dataset-row Fisher-smoke failure caused by an ambiguous RULER directory
> argument; not yet re-preregistered.**
>
> This replacement working copy becomes the next frozen Experiment 013
> preregistration only when its exact bytes and dependencies are committed in a
> clean source commit H0 and that H0 is bound before any further identity
> resolution, model staging or loading, policy fitting, protected
> materialization, or quality measurement. A dirty or unbound working copy is
> not a frozen protocol.
>
> The amendment history is retained below, but its rules remain candidate rules
> until that H0 binding. Prior identities, token spans, tokenizer-file hashes,
> and content hashes remain preserved as superseded evidence under the disclosed
> reuse rule; the next replacement identity has not yet been resolved or
> promoted. An identity candidate is not authorization to stage or load model
> weights.

Protocol draft initiated: 2026-08-02

Current fifth replacement-H0 candidate amended: 2026-08-15

Pre-resolution audit amendment: 2026-08-02. The amendment corrects a
cache-exposed-span off-by-one, binds the Stage-A calibration chain by exact
artifact-file hashes, and replaces an invalid first-output-only RULER target
with the complete task-aware serialization below. No model weights or quality
results had been opened.

Second pre-resolution audit amendment: 2026-08-02. Launcher v2 generated all
16 calibration receipts but failed before the first Stage-A receipt when
Windows attempted to encode non-ASCII prompt text as `cp1252` on the isolated
tokenizer subprocess stdin. That partial batch is rejected. A first sender-only
UTF-8 correction, launcher v3, was also rejected on its first receipt because
the isolated child still decoded stdin under the Windows locale and therefore
failed the independent-length check. Launcher v4 fixes both ends: strict UTF-8
on the parent pipe plus Python `-X utf8` in the isolated child. It is
authenticated by a new source hash. No identity had been promoted and no model
weights or quality results had been opened.

Third pre-resolution audit amendment: 2026-08-02. The first hardened launcher
v4 run stopped before publishing a receipt because Python `-I` ignored the
`PYTHONDONTWRITEBYTECODE` environment variable and imported bytecode changed the
otherwise authenticated staged-source inventory. The first v5 bootstrap smoke
then stopped nonzero because the isolated import path omitted the task script's
authenticated sibling directory. Both failed attempts are rejected. Launcher
v5 adds explicit `-B` and prepends only the authenticated task directory and
data root. A live single-receipt regression then passed every generator-side
source, runtime, command, raw-row, and tokenizer check. No identity was promoted
and no model weights or quality results were opened.

Fourth pre-resolution audit amendment: 2026-08-02. Launcher v5 subsequently
generated all 20 required receipts, but an independent producer-to-consumer
audit found that its runtime manifest could never pass identity capture: the
producer emitted isolation `flags` but omitted `machine`, while the strict
consumer required `machine` and rejected `flags` as an extra field. The entire
v5 batch is therefore rejected and is not evidence. Launcher v6 and RULER
runtime-manifest schema v3 bind both non-empty `machine` identity and the exact
isolated flags (`ignore_environment=1`, `isolated=1`, `no_user_site=1`). A
producer-to-consumer regression and a new complete 20-receipt batch are required
before identity resolution. No identity was promoted and no model weights or
quality results were opened.

Fifth pre-resolution audit amendment: 2026-08-02. A startup audit showed that
`-I -B` alone still permits `site`, virtual-environment `.pth` hooks, and reads
of unbound bytecode. Launcher v6 therefore uses `-I -S -B`, UTF-8 mode, and a
fresh verified-empty bytecode prefix; stages a complete authenticated Python
standard-library and adjacent-DLL tree; stages only RECORD-declared package
files; excludes the two exact verified virtual-environment startup files; and
binds the tokenizer and both Punkt resource layouts before each child. The
producer and consumer now agree on runtime-manifest v3. A real isolated probe
on this machine reported AMD64, 37 exact distributions, 829 Python-runtime
files, and five runtime resources and normalized identically on both sides.
This validates the launcher boundary, not any generated sequence. The v5
receipts remain rejected; a fresh complete v6 batch is still required. No
identity was promoted and no model weights or quality results were opened.

Sixth pre-resolution audit amendment: 2026-08-02. The exact 20-entry v6
command-manifest hash table is now regenerated from the authenticated launcher
and frozen by a producer-to-consumer regression. Identity input, candidate, and
frozen schemas advance together to v4 and bind a fourth execution artifact:
the exact immutable-Parquet materialization manifest. Bulk population reads
are restricted before network access to `url` for PG19 and `task_id` for
HumanEval+; prompts, solutions, and book text cannot be requested through that
surface. A live immutable-commit verification returned 13,684 PG19 training
IDs (`31050fa90be75b8c49a33c0fef9e2b1891ded6791e81a33451a9f05b8980c355`),
50 PG19 validation IDs
(`68da047a7274c57c1ee938beb2a604d082314b831ed21ed5590c1584aa4ec354`),
and 164 HumanEval+ IDs
(`913967d673127c28dc6858d4ded52e063b87496c1b4424bee696535243ebff9d`).
These are canonical projection hashes, not quality results. A clean source
commit, sealed calibration runtime, fresh v6 receipt batch, and promoted v4
identity are still required before model access.

Seventh pre-resolution audit amendment: 2026-08-02. An independent scientific
and execution audit found four defects before any model-weight or quality
access: the static/dynamic gate was step-matched rather than equal-resident;
"quality oracle" overstated a local allocator; Stage C did not explicitly own
the confirmation decision; and the named 2B checkpoint had no executable
quality contract. It also found that the source verifier safely accepts an
unchanged identity-only descendant of source commit H0, while the runner
duplicated that policy with an incompatible raw `HEAD == H0` check. These are
corrected below. Identity schemas must advance together to v5 to bind a causal
one-step diagonal empirical-Fisher comparator and its boundary commitments.
Every v4 candidate or identity is superseded. A new clean H0, sealed runtime,
fresh receipts, v5 identity, and identity-bound model staging are required. No
model weights, calibration scores, stability values, or quality results had
been opened.

Eighth pre-resolution audit amendment: 2026-08-02. The first runtime-preparation
attempt correctly rejected a previously compiled environment because an
installed distribution's RECORD inventory owned forbidden bytecode. A fresh
no-compile environment passed that check, but its first preparation attempt
stopped before publication when the staged interpreter legitimately reported
the runtime root as `.` in `sys.path` and the generic relative-path validator
indexed the sentinel as though it had path components. The runtime builder now
accepts exactly `.` only in the base-`sys.path` contract; ordinary repository,
runtime-file, interpreter, and import paths still reject it. The next
no-overwrite preparation completed with 38 distributions, 831 base-runtime
files, 20,201 package files, base paths `python311.zip`, `DLLs`, `Lib`, and `.`,
and manifest file SHA-256
`6cb19144bc373a38001f2349e8e3f9317a7809a41fd415457434bb2bc5cd74ee`.
An isolated import smoke confirmed Python 3.11.15, Torch 2.13.0+cu130, CUDA
13.0, BF16 support, Transformers 5.14.1, and PyArrow 25.0.0 on the RTX 5070.
These facts establish runtime readiness only. No model weights, calibration
scores, stability values, or quality results were opened.

Ninth pre-resolution audit amendment: 2026-08-02. The Stage-A evaluator
contract candidate was specified before the final H0 source capture and becomes
frozen only through the clean H0 and pre-access binding described above. Its
method order is
exactly `fp32_reference`, `rht_q468_uniform_q4`, `rht_q468_uniform_q8`,
`rht_q48_static_p14739`, `rht_q468_static_k27030`,
`rht_q468_dynamic_k27030`, `rht_q468_static_mse_k29334`,
`rht_q468_static_diag_empirical_fisher_h1_k29334`, and
`rht_q468_static_k29334`. The two uniform anchors are deterministic Q468
policies reconstructed from the authenticated candidate-score arrays at
`K=0` and `K=73728`. Both use the same RHT, per-row FP16 scales, two-bit code
stream, uint16 pool offsets, and packed Q4/Q6/Q8 pool implementation as the
mixed policies. Their natural resident sizes are `2,515,968` and `4,875,264`
bytes, respectively; they are descriptive lower- and upper-precision anchors,
not equal-byte comparators and never decide passage.

For every method and example, excess NLL is the ordinary mean over only the
identity-bound cache-exposed transitions. Family values first average the four
example means inside PG19, RULER, and HumanEval+, and the task macro is the
unweighted mean of those three family values. Top-1 agreement uses the same
example-then-family-then-three-family macro. Token micro is reported but never
gates. `cvar95_kl` means the FP32 mean of the largest
`max(1, ceil(0.05 * token_count))` finite per-token KL values, matching the
reviewed `fidelity_summary` implementation.

Each paired non-inferiority bootstrap independently resamples four examples
with replacement inside each family, averages inside family, then averages the
three family means equally. The generator is NumPy `PCG64`, initialized once
with seed 2,339; it draws one `10000 x 4` integer-index matrix for each family
in PG19, RULER, HumanEval+ order. The sample count is 10,000. The one-sided
95th-percentile upper bound is the nearest-rank order
statistic: sort the 10,000 replicate contrasts ascending and select zero-based
index `ceil(0.95 * 10000) - 1 = 9499`, with no interpolation. The primary is
checked separately against dynamic K27030, unweighted-MSE K29334, and diagonal
empirical-Fisher H1 K29334 under the prespecified three conjunctive margins.
Static K27030 remains diagnostic and cannot alter passage. "Beats Q48 on every
Stage-A workload" means strictly lower family-mean excess NLL in each of PG19,
RULER, and HumanEval+; equality in any family fails. No model weights, Stage-A
rows, calibration scores, or quality results were opened before this amendment.

Tenth pre-resolution audit amendment: 2026-08-02. Before the complete
calibration, the sealed runner must execute `--fisher-h1-smoke` against exactly
the first canonical record of the promoted calibration identity. It still
materializes and authenticates the complete 160-record identity before model
access, then loads the pinned model once and runs the selected record's full
token sequence through the ordinary causal capture path. Fisher calls replace
ordinary forwards at every identity-bound `B(T)` input position. Passage
requires exact token, anchor, Fisher-boundary, kernel-receipt, and adapter
Fisher-step counts; finite CPU-FP64 Q4/Q6/Q8 endpoint scores; frozen model
parameters with no gradients; successful cache detachment; model/source/runtime
reauthentication; and a clean close. The no-overwrite receipt records elapsed
time, peak CUDA allocated and reserved bytes, device identity, token count,
anchor count, and Fisher-boundary count. It publishes no score aggregate,
policy, binding, stability value, or quality result and uses the distinct
`FISHER_H1_SMOKE_COMPLETE` marker. A pass authorizes only attempting the full
calibration. Failure stops the run; there is no automatic retry or weakened
smoke input.

Eleventh pre-H0 scientific amendment candidate: 2026-08-14. A read-only
methodology audit found that an independent Stage-C HumanEval+ hash domain did
not by itself guarantee disjointness from Stage A/B, and that the Stage-C
bootstrap and relative-improvement gates were not yet executable-grade. The
rules below now exclude and bind every Stage-A/B HumanEval+ ID before Stage-C
selection; fail closed on any overlap; define the exact example, family, task,
contrast, bootstrap, bound, effect-size, top-1, and multiplicity equations; and
limit stability and fidelity language to what is actually tested. No protected
identity or row content, model weights, calibration score, stability value, or
quality result was opened, and no evaluator was run for this amendment. This
working copy remains unpreregistered until its exact bytes are committed and
bound as H0 before protected materialization.

Twelfth pre-H0 execution-boundary amendment candidate: 2026-08-14. A final
pre-execution audit found that the calibration capture's complete-RULER verifier
iterated, parsed, base64-decoded, and semantically replayed all 20 receipt
records before selecting the 16 calibration records. That path therefore could
open the four held-out Stage-A RULER receipt bodies before the one-run seal. The
then-current calibration capture was started and manually terminated after
approximately seven minutes. It published no identity input, candidate, or
frozen identity, and it loaded no model weights or quality results. The CLI has
no progress receipt, so the exact last call cannot be proven; because complete
RULER verification precedes every phase-specific dataset capture, the four
Stage-A RULER bodies are conservatively treated as possibly decoded by the
automated process. No receipt body or derived content was printed or inspected
by a person. Source commit `19ef835a8ec2341c36657b7e010ad1ae6135de9a`
and tag `experiment013-h0-19ef835` are preserved as the first bound H0 attempt,
but that H0 is now retired and superseded. It authorizes no further identity
capture or execution and produced no identity, H1, model, or quality artifact.

Calibration verification is now phase-scoped. It authenticates the canonical
complete generation-manifest bytes and the public identity, size, and SHA-256
commitments for every result, but it reads, base64-decodes, and semantically
replays only the 16 calibration receipt files. Parsing the complete manifest as
JSON necessarily parses the four Stage-A result envelopes, their embedded
command objects, and their raw-validation strings as JSON values. During
calibration those embedded protected values are deliberately uninterpreted: the
command objects are not traversed or authenticated semantically, the payload
strings are not base64-decoded, the separate Stage-A receipt files are not read,
and no Stage-A row is replayed. They are next decoded and semantically verified
only by the authenticated Stage-A identity-capture/materialization phase, which
necessarily precedes the evaluator's one-run seal; the offline evaluator then
reauthenticates and rematerializes them after reserving that one run. Thus the
repair restores cross-phase isolation from calibration, not absolute preseal
semantic blindness. Paired regressions supply deliberately invalid protected
embedded values: calibration must accept them as uninterpreted while reading
exactly its 16 receipt files, whereas Stage-A identity capture must reject them
and must never read the 16 calibration receipt files.

The conservative recovery is fixed before replacement generation and is not
result-adaptive. The original Stage-A seed 2,339 is incremented past the
already-frozen Stage-B seed grid 2,339 through 2,342; therefore 2,343, the first
seed outside that grid, is the sole replacement seed. The four old Stage-A
seed-2,339 cells are retired and are neither Stage-A nor Stage-B evidence. The
replacement inventory is exactly:

| Category/config | Replacement receipt | Canonical command-manifest SHA-256 |
| --- | --- | --- |
| retrieval / `niah_multiquery` | `retrieval__niah_multiquery__l4096__s2343.json` | `4f33fdcdf1902c17988ecce5cc344d5feffa3e99d6a948a98e4a4a0ccce51252` |
| multi-hop tracing / `vt` | `multi_hop_tracing__vt__l4096__s2343.json` | `9d2038bbc19723b27af364b870a14ed4f516272db83df13dc0e8b524b9a44bff` |
| aggregation / `fwe` | `aggregation__fwe__l4096__s2343.json` | `613477e4edc91064f120c728683858bd79c4f7c3c7c359f0010215b5adb01f48` |
| question answering / `qa_1` | `question_answering__qa_1__l4096__s2343.json` | `c0b5591282d51a94280aedafecf3c31400e9fc3423011bc4cedd6ce26e719663` |

A fresh complete 20-receipt batch and generation manifest must contain the 16
unchanged calibration identities plus exactly these four replacements. A batch
or manifest containing any retired Stage-A seed-2,339 receipt is rejected. The
capture and resolver procedure versions advance together from 5 to 6, while the
identity input, candidate, and frozen schema strings remain v5 because their
field sets did not change and no v5 identity was published. The RULER launcher
advances from v6 to v7, so all 20 canonical command-manifest hashes are
recomputed from the authenticated v7 launcher source. Generation-manifest
schema v2 remains unchanged because its structure did not change.

The next clean H0 must bind the seed-2,343 rule, exact replacement inventory,
complete v7 command-hash table, procedure versions, formatter, and regressions.
It must be committed and tagged before any seed-2,343 receipt is generated or
its raw body is accessed. Generated receipt-byte hashes do not exist at H0;
after generation they are bound by the canonical complete generation manifest
and then by the promoted identity chain. Seed 2,343 was chosen solely as the
first integer after the reserved 2,339-through-2,342 Stage-B grid, before
generation or content access. If a seed-2,343 receipt fails generation, length,
or semantic gates, stop: there is no alternate seed, config, or fallback. A
same-identity retry is allowed only after a documented infrastructure
interruption. This incident does not require new PG19 or HumanEval+ Stage-A
identities: the complete-RULER verifier ran before the phase-specific PG19 and
HumanEval+ reads.

Thirteenth pre-resolution execution-contract amendment: 2026-08-15. Under
source commit `447295e5f705a74a85ad74a74b68985914096357` and tag
`experiment013-h0-447295e`, the authenticated v7 RULER batch was produced and a
calibration identity was promoted at the identity-only descendant
`de4b8d8b514a331bcc8f4ab5b039f4c2b12473ef`. The first `stage-model`
authorization stopped before cache or output creation, model-metadata manifest
read, Hub import or download, model-payload access, adapter construction, model
loading, calibration, or quality measurement. The resolver correctly returned
recursively immutable sequence views, but calibration-runner v2 incorrectly
required mutable `list` values for the three Fisher-boundary position arrays.
This was a producer-to-consumer execution-contract defect, not an experimental
result. Both commits and frozen identity file SHA-256
`9ad6afe4a8513b8cc0cd467e75cb23bea72ee3be85c8aeebb8ed3b6e7772f260`
remain unchanged as superseded evidence and authorize no further model staging
or experiment execution.

Calibration-runner v3 accepts an authenticated non-text sequence for each
position array, still requires a nonempty sequence of exact nonnegative
integers, and immediately normalizes it to an ordinary list before the existing
evidence-equality, length, `H=1`, exact `B(T)`, and self-hash checks. The repair
does not change resolver immutability, frozen JSON, record selection, token
span, Fisher boundary, quantization policy, metric, or gate. A producer-to-
consumer regression uses the resolver's recursively frozen DTO, while text,
bytes, empty, Boolean, and negative-position variants fail closed. Runner
revision advances from v2 to v3. Identity schema v5, capture and resolver
procedure v6, RULER launcher v7, RULER generation-manifest v2, RULER runtime-
manifest v3, and source-manifest schema/profile v2 remain unchanged. The new
frozen-identity-contract and model-staging-authorization canonical stdout
documents both use schema version one. A new clean H0 and source manifest plus a
newly promoted H1 are required; the old commits must not be amended, moved, or
relabelled.

The replacement identity chain may reuse only the already fixed v7 RULER batch
whose generation-manifest file SHA-256 is exactly
`979f91848b6c0692160419c3e5e9ee555aa94d9e7add3092067f003ea0543e80`.
Regeneration is not required for this runner-only defect and would not restore
blindness after deterministic calibration materialization; exact replay
authentication is the relevant integrity check. Before reuse, exactly 20
receipt files plus one generation manifest and 20 raw sibling roots containing
100 raw files must be replay-authenticated without modification. The generator,
capture, resolver, RULER requirements, launcher revision, and their relevant
source blobs must be byte-identical to the batch-producing H0, including Git
blob OIDs `b981f693a248dbe870d27bb1d5d22a8fb09042c2` for the generator,
`43e64f3f4f72256de8eb58f3f4cd9068ef3fe305` for capture,
`dd579415f694d5900e1abdc0f46af358b2a8628b` for the resolver, and
`680c107636cc27be06652b2cfea18e0c0b82df0b` for the RULER requirements. The
replacement identity must match the superseded identity's content-manifest
commitment, records, datasets, selection, calibration split halves, tokenizer and model
contracts, and upstream revisions; only the H0/source-binding and consequent
promotion-hash cascade may differ. Any inventory, byte, hash, version, or
semantic-identity mismatch stops reuse and requires a separately preregistered
fresh complete batch after the replacement H0.

Fourteenth pre-resolution adapter-context amendment: 2026-08-15. Under source
commit `85625a5c4e4d7c6d1b015c0f3cccffea5c3d71c3`, tag
`experiment013-h0-85625a5`, and identity-only descendant
`84edf4299e8a5b3af970f74f03abc099d3696904`, the identity-bound `stage-model`
step published a local copy of the exact three-file pinned model. Its frozen
identity file SHA-256 is
`e401a3c18a002626da096ba6ba86aa5d297d16b5c8ab76711658ce730e5a5f77`.
The first sealed `--fisher-h1-smoke` attempt then authenticated the sealed
runtime, H0 source, the frozen identity bytes committed at H1, and all four
bound manifest byte strings. It parsed and matched the public model-file
metadata but failed deterministically
while initializing `AdapterConstructionContext`: calibration-runner v3 supplied
the inert absolute `git_executable` path that capture procedure v6 requires,
while the authenticated calibration API's exact runtime-context key set omitted
that key.

The failed smoke did not reach reviewed-adapter loading or construction,
calibration materialization, tokenizer, dataset, or RULER access, staged-model-
root traversal or file hashing, model configuration or weight deserialization,
CUDA or Fisher execution, or output-directory checking or publication. It
created no smoke output or staging sibling and produced no smoke report,
completion marker, score, policy, calibration binding, stability value, or
quality result. The separately completed `stage-model` action did access and
publish the model payload under H1; the failed smoke itself did not read that
published model root. This is a downstream execution-contract defect, not an
experimental result or infrastructure interruption. The H0, tag, H1, identity,
and staged model root remain unchanged as superseded incident evidence and
authorize no further official execution.

Calibration-runner v4 aligns the authenticated API with capture v6's existing
five-key runtime context. The API now accepts, validates, copies, and retains
the absolute inert Git path while preserving exact-key rejection and recursive
mapping immutability. Production `_official_main` and the regression share one
context-construction helper; the regression loads the exact authenticated API
and reviewed adapter and checks that construction creates none of its model,
cache, RULER, or bytecode sentinel paths. A separate no-data boundary
regression carries the same context through the real adapter into the
manifest-bound capture module and stops before artifact decoding, Hub,
tokenizer, or dataset access. No record selection, dataset revision, token
span, Fisher boundary, quantization policy, metric, gate, model contract, or
protected-stage rule changes.

Runner revision advances from v3 to v4. Identity schema v5; capture and
resolver procedure v6; adapter revision v2; RULER launcher v7, generation-
manifest v2, and runtime-manifest v3; calibration runtime-manifest v4; source-
manifest schema/profile v2; model-manifest v1; frozen-contract and model-
staging-authorization stdout schemas v1; run-report v2; and Fisher-boundary and
smoke-marker contracts v1 remain unchanged. A new clean H0, repository-source
manifest, promoted calibration identity, H1, and identity-bound model
publication are required. The exact authenticated RULER v7 batch, sealed
runtime, public model metadata manifest, Parquet manifest, and shared Hub cache
may be reused only after their existing byte, inventory, and semantic checks
pass unchanged. The old H1 model root may be treated only as preserved incident
evidence, not as the official model root for the next H1; a fresh no-overwrite
root must be published by a newly authorized `stage-model` execution.

Fifteenth pre-resolution model-staging-path amendment: 2026-08-15. Under
source commit `0f3ea5e86e5d2ec13d5c5836540ce105e41ad02b`, tag
`experiment013-h0-0f3ea5e`, and identity-only descendant
`dae5587adc8f9a2b16335dfdc501e7f0a3f5e6ab`, the required read-only
model-staging authorization succeeded. Its frozen identity file SHA-256 was
`65cd1ccd932db1aa4c8f2f06e4b7a88b67532734f611dc33bbba39fbfea1cdb7`.
The subsequent first `stage-model` invocation supplied repository-local Hub
cache root `.cache/exp013-identity`. Calibration-runner v4 fully
reauthenticated H1, H0 source, the frozen identity, and the public model-
metadata manifest, then deterministically rejected the normalized cache root
because an official Hub cache must be outside the repository.

The attempted output root
`C:\tmp\recurquant-exp013-model-h1-dae5587a` remained absent. The failed
command inspected only staging-path metadata after authorization. It did not
import a Hub client, invoke a downloader, traverse or read cached payload
files, create a staging directory, publish an output root, construct an
adapter, load model configuration or weights, materialize calibration data,
execute CUDA or Fisher computation, or produce a score, policy, smoke marker,
calibration binding, stability value, or quality result. The Git worktree
remained clean.

The immediate cause was an invalid command argument. The execution-control
defect was the absence of a non-consuming staging-path preflight and runner
v4's ordering of path validation after H1 authentication. This was neither an
experimental result nor an infrastructure interruption, so the existing
same-command infrastructure-retry exception does not apply and no retroactive
exception is introduced. The H0, tag, H1, identity, and failed command remain
unchanged as superseded incident evidence and authorize no further official
execution.

Calibration-runner v5 introduces one pure staging-path validator shared by
`verify-model-staging-paths` and `stage-model`. The read-only verifier accepts
only repository root, Hub cache root, and prospective model output root. It
performs no Git operation or identity, source-manifest, model-manifest, Hub,
cache-payload, adapter, model, or dataset access; imports no Hub downloader;
creates no directory or artifact; and writes only deterministic canonical JSON
to stdout. It requires an existing regular non-link repository root, an
existing regular non-link external Hub cache, an existing regular non-link
output parent, an absent non-root output destination, and pairwise disjoint
repository, cache, and output roots in both nesting directions. Every existing
path component must be free of links and reparse points. The Hub-cache root and
output parent may not themselves be filesystem roots. The output leaf is 1
through 128 characters, begins with an ASCII alphanumeric, contains only ASCII
alphanumerics, dot, underscore, or hyphen, and ends with an ASCII alphanumeric,
underscore, or hyphen. Reserved DOS names are also rejected.

`stage-model` invokes the same pure validator before Git-executable
authentication or any H1 authentication and requires
`--expected-model-staging-path-contract-sha256` to equal the digest produced by
the prior verifier. A missing, malformed, or unequal digest fails at that pure
boundary and does not consume H1. After successful authentication, it repeats
the validator and requires the same normalized roots and directory-component
identities before Hub import, payload access, or staging creation. Immediately
before atomic publication it repeats both staging-path and identity
authorization and rejects any root, component-identity, destination-existence,
or authorization drift. A semantic or authentication mismatch still retires
H1; a documented infrastructure interruption after authentication begins but
before payload access permits only the existing exact same-command retry.

Runner revision advances from v4 to v5. The new canonical stdout document uses
artifact kind `recurquant_experiment013_model_staging_paths_verification` and
schema version one. It publishes no raw local path. It contains SHA-256 digests
of the normalized absolute repository, Hub-cache, output-parent, and output-
root paths; SHA-256 digests of the ordered device/inode/mode identity chains for
the repository, cache, and output-parent components; explicit states
`existing_regular_non_link_directory` for those three existing roots and
`absent` for the output root; and `path_contract_sha256`. That contract digest
authenticates canonical newline-terminated JSON containing schema version one
plus exactly those path, component-identity, and state fields. A successful
`stage-model` result repeats the same digest as
`model_staging_path_contract_sha256`, coupling the standalone preflight to the
internally revalidated staging call. Identity schema v5; capture and resolver
procedure v6; adapter revision v2; RULER launcher v7, generation-manifest v2,
and runtime-manifest v3; calibration runtime-manifest v4; source-manifest
schema/profile v2; model-manifest v1; existing verification stdout schemas v1;
run-report v2; and Fisher-boundary and smoke-marker contracts v1 remain
unchanged. No record selection, dataset revision, calibration span, Fisher
boundary, quantization policy, metric, gate, model contract, or protected-stage
rule changes.

A fresh clean H0 and source manifest, newly promoted calibration identity,
identity-only H1, successful staging-path preflight, model-staging
authorization, and fresh no-overwrite identity-bound model publication are
required. The exact RULER v7 batch, sealed runtime, public model metadata
manifest, Parquet manifest, and external shared Hub cache remain reusable only
after their existing point-of-use checks pass unchanged. The replacement
identity must retain all 160 records and content manifest
`ee72483a8f8b4370c9e667e4287747e5bc358aeb0265a58167140f4e780a7b29`;
relative to the retired
`65cd1ccd932db1aa4c8f2f06e4b7a88b67532734f611dc33bbba39fbfea1cdb7`
identity, only these five repository-source and promotion-hash cascade JSON
pointers may differ:

```text
/canonical_evidence_sha256
/evidence/execution_bindings/repository_source_manifest_file_sha256
/evidence/promotion/candidate_canonical_evidence_sha256
/evidence/promotion/candidate_file_sha256
/evidence/source_manifest_sha256
```

Sixteenth pre-resolution RULER-receipt-directory amendment: 2026-08-15. Under
source commit `475659ac8a0a98aaf38814e89f0f95d31392ec8b`, tag
`experiment013-h0-475659a`, and identity-only descendant
`fd67384944dc92abac0422960ea53fc973b64736`, the runner-v5 staging-path
preflight, model-staging authorization, and identity-bound model publication
completed. The frozen identity file SHA-256 was
`40c434d038879608093fc8f74b66893062e4f52a0e1db9d33b40ac9fa411be90`,
and the published model root was
`C:\tmp\recurquant-exp013-model-h1-fd673849`.

The first sealed `--fisher-h1-smoke` invocation then supplied the external
RULER source checkout to runner v5's generically named `--ruler-root` option.
The reviewed adapter interprets that value only as the directory containing the
sealed RULER result receipts. The command therefore reached calibration
sequence materialization, authenticated source-head metadata and public
tokenizer material, and then rejected the shallow directory inventory: all 20
frozen receipt filenames plus `generation-manifest.json` were missing, while
the source checkout's Git, environment, documentation, source, auxiliary, and
raw-receipt entries were unexpected. The immediate cause was a wrong command
argument; the execution-control defect was an ambiguous CLI name with no pure
early receipt-directory precondition.

The inventory rejection occurred before any generation-manifest or RULER
receipt body was opened. It also preceded MBPP row access, PG19 projection or
row access, RULER record decoding or semantic replay, and HumanEval+ projection
or row access. Because complete sequence materialization precedes local model
authentication, the failed smoke did not traverse or hash the published model
root, read model configuration or weight bytes, deserialize parameters, enter
CUDA, execute a causal forward or Fisher step, or compute any calibration or
quality value. The declared smoke output directory remained absent; no report,
completion marker, score, policy, calibration binding, stability value, or
quality artifact was published. This is a pre-model and pre-dataset-row
execution-contract incident, not a Fisher result and not an infrastructure
interruption.

The child additionally reported that its sealed scratch directory was nonempty
while unwinding the primary inventory exception, and the host repeated that
postcondition failure before deleting the owned scratch tree. No scratch path
survived, but its transient inventory was not preserved and therefore must not
be guessed. This secondary diagnostic does not replace or weaken the proven
primary cause. The repaired launcher preserves a child exception as primary and
adds aggregated child-postcondition failures as notes; when `sealed_main` or
the sealed child returns nonzero, it preserves that return code and reports
secondary postcondition or cleanup failures separately. Only an otherwise
successful child promotes a postcondition failure to the primary error. The
host records each temporary root's device/inode/type identity at creation,
refuses cleanup if that identity changed or the owned tree contains a link,
reparse point, or entry other than a regular file or directory, removes only
the authenticated owned tree,
and detects survival. Regressions must cover primary-exception and nonzero-
return preservation, aggregation of multiple secondary failures, successful-
child postcondition failure, identity replacement, reparse refusal, partial
temporary-root creation, and cleanup of an ordinary nonempty owned tree.

Calibration-runner v6 removes `--ruler-root` from the official smoke and full-
calibration CLI without a compatibility alias. The sole public option is
`--ruler-receipt-dir`; its help and protocol meaning are exactly the sealed v7
receipt directory, never the RULER source checkout or raw-output directory. At
sealed-runner entry, before that runner reads or authenticates the runtime
manifest, a standard-library-only precondition requires an absolute existing
directory whose
ancestors and root contain no link or reparse point; an exact shallow,
case-insensitively unique inventory of `generation-manifest.json` plus the 20
frozen receipt filenames; and a regular non-link, non-reparse file for every
entry. It opens no file body or JSON value, performs no Git or artifact
authentication, imports no capture, tokenizer, dataset, adapter, Hub, model, or
CUDA code, and creates no artifact. `_official_main` repeats the same
precondition before reading the frozen identity or runtime, model, Parquet, or
source-manifest bytes. The existing phase-scoped semantic verifier remains
authoritative at point of use and still authenticates the permitted receipt
bodies and their complete manifest commitments.

Runner revision advances from v5 to v6. The external CLI rename does not rename
the reviewed adapter's internal `AdapterConstructionContext.ruler_root` field;
that field remains an implementation detail populated only from the normalized
`--ruler-receipt-dir` result. A regression must prove that the retired
`--ruler-root` spelling is unrecognized, the RULER source checkout fails at the
pure precondition, the exact 21-file receipt directory passes, and every
missing, extra, case-colliding, non-regular, link, reparse, relative, or
ancestor-link variant fails before manifest authentication, materialization,
model access, or scratch population. No record selection, dataset revision,
calibration span, Fisher boundary, quantization policy, metric, gate, model
contract, or protected-stage rule changes.

The incident H0, tag, H1, frozen identity, published model root, failed command,
and absent output remain unchanged as superseded evidence. They authorize no
same-command retry and no further official execution. In particular, the old
published model root may not be adopted, renamed, copied, or rebound as the
official model root of a replacement H1.

The exact RULER v7 batch may be reused only if its complete 21-file receipt
inventory and the 100 raw producer files replay-authenticate unchanged and its
generation-manifest file SHA-256 remains
`979f91848b6c0692160419c3e5e9ee555aa94d9e7add3092067f003ea0543e80`.
The intended CLI value is the absolute resolution of repository-relative
`artifacts/experiment013/ruler-receipts-v7-h0-447295e`; the separate pinned
RULER source checkout is used only for replay authentication and must never be
passed as `--ruler-receipt-dir`.
The sealed calibration runtime, public model-metadata manifest, immutable-
Parquet materialization manifest, and external shared Hub cache may likewise be
reused only after all existing byte, inventory, path, version, and semantic
checks pass at every required point of use. Their currently expected manifest
file SHA-256 values are, respectively,
`80ca233a29af4facbb334fd4fb51a4f6e9a3d6815465cb79b9f3db63ef668d6a`,
`586d9c7e520f3bbd99ecef30663bf07d283eb14622475c58891becd8e033b05c`,
and `ee5628e50e5d3516fd79077542d355fd915455ac0e53128d372f4177ad63d39c`.
Any mismatch stops reuse; it is not an authorization to regenerate, substitute,
or weaken a frozen dependency.

Before another smoke, the exact runner-v6, launcher cleanup repair, tests, and
this protocol must be committed in a fresh clean H0 and bound by a fresh
repository-source manifest. The exact promoted calibration identity must be
recaptured and reverified against that source binding, committed alone in a
fresh H1, and reauthorized. Model-staging paths and authorization must be
reverified, and the same authenticated three-file payload must be published by
a new no-overwrite `stage-model` execution under that H1 into a fresh model
root. Only then may the sealed Fisher H=1 smoke be attempted with the exact
receipt directory supplied through `--ruler-receipt-dir`. The replacement
identity must retain all 160 records and content manifest
`ee72483a8f8b4370c9e667e4287747e5bc358aeb0265a58167140f4e780a7b29`;
relative to the retired identity, only the five repository-source and
promotion-hash cascade pointers listed above may differ. Any other identity
difference stops the chain.

## Question

Can a calibration-frozen, static Q4/Q6/Q8 recurrent-state layout satisfy the
prespecified cache-exposed teacher-forced target-NLL and top-1 non-inferiority
gates against an online local-distortion allocator at the same complete
recurrent-resident byte budget while using one immutable code map that a packed
GPU kernel can execute efficiently?

The candidate is:

```text
rht_q468_static_k29334
```

The short name is **static RHT-Q468**. `K29334` is the exact sum of precision
steps over all 36,864 recurrent-state groups, where Q4, Q6, and Q8 contribute
zero, one, and two steps respectively. This experiment is an adoption study,
not a novelty, state-of-the-art, deployment, or breakthrough claim.

## Fixed model contracts

The primary checkpoint and tokenizer are:

```text
Qwen/Qwen3.5-0.8B-Base
revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
Transformers 5.14.1
```

The conditional scale resource probe is:

```text
Qwen/Qwen3.5-2B-Base
revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
Transformers 5.14.1
```

The 2B checkpoint is feasibility-only in Experiment 013. The only authorized
operation is an otherwise idle BF16/eager cold-start load with no prompt,
generation, cache-quality measurement, or offload. Record both peak allocated
and peak reserved device memory. A value above `7.5 GiB`, or inability to make
the measurement, is a resource stop.

A passing cold start authorizes only a separately frozen, model-specific
protocol defining geometry, complete byte budgets, calibration refit or transfer,
identities, workloads, thresholds, and implementation. It contributes no
quality, generalization, scaling, or confirmation evidence to Experiment 013.

The primary 0.8B run uses batch one, eager evaluation, no sampling, BF16 model
weights, and FP32 reference recurrent state. Model architecture, recurrent
layer indices, state geometry, tokenizer class, tokenizer files, and every
runtime package are identity-bound before weights are opened. A mismatch stops
the run. The 2B resource probe shares only the frozen BF16/eager load contract;
it does not enter recurrent-state evaluation under this protocol.

The metadata-only upstream identities are frozen without opening example
contents:

| Role | Source | Revision |
| --- | --- | --- |
| existing MBPP calibration | `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| PG19 | `emozilla/pg19` | `c021754c8e01c5b1cc83a1f549c1f97fbbb756b8` |
| RULER generator | `NVIDIA/RULER` | `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` |
| HumanEval+ dataset | `evalplus/humanevalplus` | `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` |
| EvalPlus source | `evalplus/evalplus` | `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` |

PG19 train, validation, and test parquet siblings were confirmed at the pinned
revision. That metadata check does not resolve, retain, format, or tokenize an
Experiment 013 row.

## Candidate format and exact bytes

Every recurrent group contains 128 FP32 values before quantization. The codec
applies the already specified deterministic right randomized Hadamard transform
within the value axis, followed by symmetric signed absmax Q4, Q6, or Q8 with
one FP16 scale per group. Nearest-even rounding is fixed. The sign schedule and
transform convention are inherited from Experiment 009; a source freeze must
bind their exact implementation and tests.

The static precision map is fixed after calibration and does not change by
prompt, token, cache write, batch element, or runtime activation. Codes use two
bits per group. Uint16 offsets address the packed group pool. The primary
resident allocation is:

| Component | Bytes |
| --- | ---: |
| packed Q4/Q6/Q8 payload | 3,297,984 |
| FP16 scales | 73,728 |
| two-bit precision codes | 9,216 |
| uint16 pool offsets | 73,728 |
| alignment padding | 8 |
| **resident recurrent state** | **3,454,664** |

The FP32 recurrent-state reference is `18,874,368` bytes. Shared metadata,
codec constants, transient decode workspaces, allocator peaks, model weights,
convolution state, and attention KV caches are reported separately. None may
be hidden inside the recurrent-state number. No candidate or comparator may
retain an undisclosed persistent FP32 state or dequantized mirror.

The exact-byte static Q4/Q8 comparator is `rht_q48_static_p14739`: 14,739 groups
use Q8 and the rest Q4. Its payload is `3,302,592` bytes, its one-bit code map
is `4,608` bytes, and its scales, uint16 offsets, and eight alignment bytes
make the same `3,454,664`-byte total.

Static `K27030` is a prespecified selection-step diagnostic. Its payload is
`3,224,256` bytes; scales, two-bit codes, and uint16 offsets add `73,728`,
`9,216`, and `73,728` bytes, respectively, for `3,380,928` resident bytes with
no alignment padding. It is never rounded up to the primary budget.

The online `rht_q468_dynamic_k27030` baseline owns the same `K27030` payload,
scales, and code bytes, plus `147,456` bytes of persistent FP32 query EMA and
eight alignment bytes, for exactly `3,454,664` recurrent-resident bytes. It has
the same complete resident budget as static `K29334`; static `K27030` does not.

## Calibration and static-policy freeze

Policy fitting uses only these calibration sources:

1. the existing frozen public-evaluation v0.2 MBPP calibration population of
   128 tasks;
2. 16 SHA-ranked eligible PG19 training books, with one deterministic
   2,304-token segment from each book; and
3. four sequences from each of NVIDIA RULER's four official task categories,
   using configured lengths 2,048 and 4,096 and generator seeds 12,339 and
   12,340 as frozen below.

Each workload family receives equal weight regardless of its number of tokens
or examples. Within a family, examples receive equal weight. The exact
configurations, canonical IDs, formatter identities, content hashes,
tokenizer-file hashes, and token spans must be resolved into an identity
amendment before fitting. The protocol does not guess unresolved row or token
identities.

Eligible PG19 IDs are ranked by lowercase SHA-256 of the UTF-8 domain-separated
canonical ID. Training and validation use distinct domains. HumanEval+ uses a
different domain, and Stage C uses a separate confirmation domain. Ties break
by canonical ID. The exact domains are:

```text
recurquant.experiment013.pg19.train.v1\0
recurquant.experiment013.pg19.validation.v1\0
recurquant.experiment013.pg19.test.v1\0
recurquant.experiment013.humaneval-plus.stage-a-b.v1\0
recurquant.experiment013.humaneval-plus.stage-c.v1\0
```

The canonical PG19 ID is the exact UTF-8 `url` field; the pinned PG19 schema
does not contain a `book_id` field. The population is the ordered `url`
projection from the exact Parquet objects at conversion commit
`b3624dc44b60cb01e74876e8869234d2660812cf`. Capture authenticates the pinned
source revision, conversion revision, ordered file paths, Git blob IDs, LFS
SHA-256 identities and sizes, and row-group footer counts before and after the
projection. It does not use a mutable Dataset Viewer endpoint. A training book
is eligible when the pinned tokenizer produces at least 2,304 tokens. Rank all
13,684 training URLs before opening text, then inspect them in that fixed order
only until 16 eligible books have been accepted. For an accepted book with `N`
tokens, define

```text
M = N - 2304
u = unsigned big-endian integer from the first 8 bytes of
    SHA256("recurquant.experiment013.pg19.segment.v1\0" || UTF8(url))
segment_start = u mod (M + 1)
segment_stop = segment_start + 2304
```

No tokenizer special tokens are added. The same URL identity and eligibility
rule applies to validation, except eligibility requires at least 4,224 tokens
for the frozen 4,096-token prefill and 128 continuation tokens. For an accepted
validation book, replace `2304` by `4224` in the equation above and use the
independent namespace
`recurquant.experiment013.pg19.validation-segment.v1\0`; the first 4,096
tokens of that slice are prefill and the last 128 are the continuation. HumanEval+ uses
the exact `task_id` field. RULER uses the complete domain-separated
configuration identity that the later generator amendment must freeze.

RULER category and exact configuration are separate identity fields. The
pinned `scripts/synthetic.yaml` contains these 13 configurations:

| Category | Exact configuration IDs |
| --- | --- |
| retrieval | `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery` |
| multi-hop tracing | `vt` |
| aggregation | `cwe`, `fwe` |
| question answering | `qa_1`, `qa_2` |

For calibration, order `(configured_length, seed)` as `(2048,12339)`,
`(2048,12340)`, `(4096,12339)`, `(4096,12340)`. Within each category, rank
its exact config IDs by lowercase SHA-256 of

```text
"recurquant.experiment013.ruler.calibration-config.v1\0" || UTF8(config_id)
```

and cycle through that ranked list across the four ordered pairs. The resolved
schedule is:

| Category | Exact configs in pair order |
| --- | --- |
| retrieval | `niah_multiquery`, `niah_multikey_2`, `niah_single_1`, `niah_multivalue` |
| multi-hop tracing | `vt`, `vt`, `vt`, `vt` |
| aggregation | `fwe`, `cwe`, `fwe`, `cwe` |
| question answering | `qa_1`, `qa_2`, `qa_1`, `qa_2` |

This is exactly 16 RULER calibration sequences. It is a compute-bounded,
category-balanced calibration sample, not the RULER evaluation grid.

RULER's configured length is not assumed to equal the actual token count: its
official generators reserve answer tokens and may emit a shorter tokenized
sequence. Identity records therefore bind `configured_length`, actual
`sequence_length`, prompt/scored half-open spans, and the generator's own
length receipt separately. Anchors use the actual processed token count only.

The pinned upstream `prepare.py` constructs its child command as a multiline
shell string. On Windows, an initial compatibility smoke test returned exit
zero but emitted only a truncated 155-token row with no generated context or
question. That row is rejected and is not evidence. Experiment 013 launcher v6
invokes the pinned task generator directly with a no-shell argument vector.
Every receipt must authenticate the RULER commit and source blobs, the exact
launcher source,
the isolated Python/package manifest, all tokenizer assets, all auxiliary
corpora, NLTK Punkt, and all Wonderwords noun, adjective, and verb lists. It
must also contain exactly one generated row, the frozen task markers, the
configuration's required output cardinality, and every required NIAH answer in
the prompt. A partial generation batch is not promotable.

Within each broad calibration family, and separately within each of RULER's
four official categories, SHA-rank canonical sequence IDs and alternate even
and odd ranks into split halves A and B. Recompute the complete equation and
both exact-K allocations independently on each half. This produces
deterministic halves without observing a quality result.

For every calibration sequence of `T` tokens, maintain the existing causal
normalized-query-energy EMA over every token, with decay `2^(-1/32)`, epsilon
`1e-6`, and a uniform `1/128` prior. Capture exactly the unique zero-based
post-token anchors

```text
p_j = floor((j + 1) T / 16) - 1,  j = 0,...,15.
```

If `T < 16`, capture all `T` positions. The 16-anchor rule is frozen to bound
calibration cost. An empty sequence, duplicate anchor after canonicalization,
non-finite state, energy, distortion, or aggregate fails closed. Every anchor
identity and its ordered manifest hash is recorded.

At anchor `p`, for physical row
`r = (frozen recurrent-layer order, head, key-row)`, transform the FP32 state
with right-RHT seed 2,339. For `b` in `{4, 6, 8}`, use symmetric per-row
group-size-128 quantize/dequantize and accumulate in CPU FP64:

```text
x[e,p,r,b] = EMA_query_energy[e,p,r]
             * mean_value((Q_b(RHT(S[e,p,r])) - RHT(S[e,p,r]))^2)
```

Mean anchors within each sequence. Mean sequences within MBPP and PG19. For
RULER, mean sequences within each of its four official categories, then mean
the four category means. The final score is

```text
D_b(r) = (D_MBPP,b(r) + D_PG19,b(r) + D_RULER,b(r)) / 3.
```

The three broad calibration families therefore have equal coefficients
irrespective of example counts. There is no additional normalization,
clipping, rescaling, quota, or task-loss input.

Flatten rows layer-major, then head-major, then key-row-major using the frozen
18-layer list. Feed `D4`, `D6`, and `D8` directly to the existing
`allocate_exact_multibit_codes_fast` allocator at `K=29334` and `K=27030`;
codes zero, one, and two mean Q4, Q6, and Q8. Its exact-equality tie rule is the
lexicographically greatest flattened code vector, so the lower flat row gets
higher precision first.

No task loss, Stage-A value, Stage-B value, or Stage-C value participates in
the candidate's query-energy code-map fit. The primary map uses `K29334`. The
diagnostic static and dynamic layouts both use `K27030`. Once an identity and
code map are committed, no seed, score, quota, tie rule, token span, group size,
or bit budget may be changed under Experiment 013.

## Prespecified sensitivity comparators

Two same-format `K29334` static maps separate the mixed-format question from the
candidate's query-energy selector. Both use the identical RHT, Q4/Q6/Q8 codec,
scales, codes, offsets, tie rule, and `3,454,664` resident-byte ledger.

The unweighted comparator is `rht_q468_static_mse_k29334`. It replaces the
query-energy factor by one and otherwise uses the same per-sequence and
family-balanced reduction.

The loss-sensitive comparator is
`rht_q468_static_diag_empirical_fisher_h1_k29334`. It is an adapted diagonal
empirical-Fisher baseline, not RateQuant itself and not an exact Fisher matrix
or Hessian. Let `x[0:T]` denote one frozen token sequence and let `S_b` be the
persistent FP32 recurrent state after consuming token `x_b`. Eligible stored
boundaries are `b = 0, ..., T-3`, selected exactly as

```text
B(T) = frozen_anchor_positions(T - 2).
```

At boundary `b`, the measured recurrent step consumes `x_(b+1)`. Its logits
`z_(b+1)` predict target `x_(b+2)`. The causal gradient is

```text
g_b = d CE(z_(b+1), x_(b+2)) / d S_b.
```

`z_b` is forbidden because it was produced before `S_b` was stored. For row
`r`, transform both state and gradient into the codec basis:

```text
Z_b(r) = RHT(S_b(r))
G_b(r) = RHT(g_b(r))
risk_q(b,r) = 0.5 * sum_value(G_b(r)^2 * (Q_q(Z_b(r)) - Z_b(r))^2)
```

for `q` in `{4, 6, 8}`. Compute the risk in deterministic FP64 reduction order.
Mean boundaries within sequence, then use the same MBPP/PG19/RULER and RULER
category balancing as the candidate. Feed the three endpoint risks directly to
the exact allocator at `K=29334` with the existing flattening and tie rule.
There is no query-energy multiplication, gradient normalization, clipping,
layer quota, protected-set loss, or post-result tuning.

Identity v5 domain-separates and binds, for every calibration sequence, the
horizon `H=1`, boundary positions, input positions, target positions, and their
token-ID hashes. The implementation must prove one successful recurrent kernel
call per layer, no model-parameter gradients, exact rollback on a failed Fisher
step, and a fully detached continuing FP32 cache. A real RTX 5070 peak-memory
and runtime smoke receipt is required before the complete calibration run.

## Policy-stability gates

Before any Stage-A quality result is opened, independently fit K29334 maps on
split halves A and B. All three gates are conjunctive:

1. Spearman rank correlation between the two flattened K29334 precision-code
   vectors is at least `0.70`, using average ranks for tied codes; a constant
   vector is undefined and fails closed;
2. Q8-set Jaccard similarity is at least `0.50`; and
3. every recurrent layer's absolute mean assigned-bitwidth shift is at most
   `0.25` bits, where codes zero, one, and two map to 4, 6, and 8 bits.

Failure stops the static candidate. The map may not be stabilized by changing
the data, metric, seed, threshold, or aggregation after the failure is known.

These split-half gates apply only to the query-energy candidate map. The
unweighted-MSE and diagonal empirical-Fisher maps are fixed comparator
instantiations fitted on the complete calibration identity; Experiment 013 does
not test their split-half stability. Therefore any Stage-A or Stage-C comparison
is only against those exact frozen maps. It cannot establish that query-energy
selection is generally more stable, robust, or effective than MSE or Fisher
sensitivity. Such a selector-principle claim requires a new pre-access amendment
with symmetric comparator-stability tests or a new experiment.

## Frozen evaluation identities

Identity resolution is staged. A resolver may create only a quarantined
candidate. A separate explicit promotion, checked by candidate SHA-256,
creates the identity that must be committed before model weights are loaded.
Stage-B and Stage-C content is protected and requires separate authorization;
ordinary resolver tests and dry runs must not read it.

Identity schema v5 also binds four exact pre-model evidence files under
`execution_bindings`:

```text
repository_source_manifest_file_sha256
calibration_runtime_manifest_file_sha256
model_file_manifest_file_sha256
parquet_materialization_manifest_file_sha256
```

The source manifest authenticates the implementation, tests, protocol, and
runner at point of use. The runtime manifest authenticates the Python and
installed package-code inventory used by calibration. The model manifest is
derived from immutable Hub repository/LFS metadata without downloading or
opening weight payloads. The Parquet manifest authenticates the source and
conversion commits plus the selected Parquet Git/LFS objects. Only after
promotion may the runner hash local model files and compare them with the
frozen model manifest. A missing, malformed, or byte-different dependency
stops before adapter data access or model loading.

The source manifest binds implementation commit H0. Committing the promoted
identity creates H1. H1 is authorized only when H0 is its Git ancestor, the
authenticated source verifier proves every frozen source path has identical H0
tree, H1/index, and worktree bytes, and the worktree is otherwise clean. Reports
and policy artifacts continue to record H0 as implementation provenance; H1 is
the identity authorization commit and may not be relabelled as source commit.

Before committing H1, the exact promoted identity bytes in their ignored,
no-overwrite precommit location must pass `verify-frozen-identity-contract`.
That read-only command authenticates H0 and its source manifest, loads the exact
H0 resolver, and consumes the complete record inventory through calibration-
runner v6's identity view. It accepts no H1, model manifest, Hub, cache, or
output argument. Its non-persisted canonical JSON stdout document uses artifact
kind `recurquant_experiment013_frozen_identity_contract_verification`, schema
version one, and binds the H0/source contract, portable Git identity, all four
execution bindings, complete identity/canonical/assignment hashes, public model
and tokenizer contracts, and record count. The bytes that passed are then copied
without modification as the sole H1 tree change; regeneration or hand editing
after that preflight is forbidden.

After H1 and before model-staging authorization, run the read-only
`verify-model-staging-paths` command twice against the exact intended
repository, external Hub cache, and absent output root. It accepts only those
three path arguments, uses the same pure validator as `stage-model`, performs no
Git or artifact authentication and no Hub, payload, adapter, model, or dataset
access, and creates no filesystem entry. Its non-persisted canonical JSON
stdout document uses artifact kind
`recurquant_experiment013_model_staging_paths_verification`, schema version one,
and binds status, runner revision, normalized absolute-path digests for the
repository, cache, output parent, and output root; component-identity-chain
digests for the three existing roots; their exact regular/non-link states; the
absent-output state; and the aggregate `path_contract_sha256`. The two stdout
byte strings must be identical. The exact digest must then be supplied to
`stage-model` as `--expected-model-staging-path-contract-sha256`; a missing,
malformed, or unequal value fails before Git or H1 authentication. A successful
`stage-model` result must echo that required digest as
`model_staging_path_contract_sha256`. A failure at this pre-authentication
boundary does not consume H1.

Only then may `verify-model-staging-authorization` invoke the same authorization
path used by `stage-model` and reauthenticate the H1, index, and worktree
identity bytes; H0 ancestry and unchanged source tree; the complete frozen
identity and execution bindings; and the exact public model-metadata manifest.
The command accepts no cache or output root, imports no Hub downloader,
downloads no file, and creates no directory or artifact. Its canonical JSON
stdout document, which the command does not persist, uses artifact kind
`recurquant_experiment013_model_staging_authorization`, schema version one, and
binds status, runner revision, frozen-identity hash, H1, H0, repository-source-
manifest hash, model-manifest hash, public model ID/revision, Hub-tree-manifest
hash, file count, and total bytes. Only successful path and authorization
documents permit `stage-model` to be attempted with those same roots. A
semantic or authentication mismatch retires that H1; it may not be hand-edited
or weakened. An argument-parse or initial pure path-precondition failure does
not consume H1 because authentication did not begin. A documented
infrastructure interruption after authentication begins but before model-
payload access permits only an exact same-command retry under that H1.

Model payload staging begins only after the frozen identity is tracked with
identical H1, index, and worktree bytes. The identity-bound stager downloads
only the exact sorted root files in the frozen model manifest at the exact
40-hex public Hub revision, using an external cache and no token. Before Git or
H1 authentication, immediately after authorization, and immediately before
publication, the stager repeats the pure path validator and requires identical
normalized roots and existing-component identities. The cache and output
parent may not be filesystem roots, and the absent output leaf must be the same
canonical 1-through-128-character basename accepted by the preflight: an ASCII
alphanumeric first character, only ASCII alphanumeric/dot/underscore/hyphen
interior characters, an ASCII alphanumeric/underscore/hyphen final character,
and no reserved DOS name. Returned cache paths are untrusted: every source must
resolve inside that cache, then be stream-copied into a fresh sibling staging
directory.
Ordinary files are checked by Git blob OID and size; LFS payloads are checked by
payload SHA-256 and size. The staged tree must have exact case-insensitive-
unique inventory and contain no links, reparse points, cache metadata, marker,
or extra file. Reauthenticate the identity, source, and manifest immediately
before an atomic no-replace directory rename, then independently authenticate
the published model root. Failure may clean only the exact owned staging-
directory identity; it never overwrites the output or deletes the shared Hub
cache.

Every official Fisher smoke and full calibration invocation uses the public
option `--ruler-receipt-dir` for the normalized exact 21-file v7 receipt
directory. The retired `--ruler-root` spelling is invalid and has no legacy
alias. After the outer sealed launcher has authenticated the source and runtime
needed to establish the child, but before the sealed runner reads its runtime
manifest or any frozen-identity, source-manifest, model-manifest, or Parquet-
manifest byte, runner v6 performs the shallow pure directory precondition
defined by the Sixteenth amendment and repeats it inside `_official_main`.
Passing that precondition does not authenticate any receipt body; phase-scoped
point-of-use verification remains mandatory.

Stage-A resolution additionally consumes one strictly decoded
`experiment-013-stage-a-calibration-binding-v3` artifact. The resolved Stage-A
identity binds these eight dependency files directly, not merely semantic IDs
copied from a caller:

```text
calibration_identity_file_sha256
calibration_score_artifact_file_sha256
split_half_stability_artifact_file_sha256
static_k27030_policy_file_sha256
static_k29334_policy_file_sha256
comparator_score_artifact_file_sha256
static_fisher_k29334_policy_file_sha256
static_mse_k29334_policy_file_sha256
```

The comparator-score dependency is one strict canonical artifact containing
exactly the unweighted-MSE and diagonal empirical-Fisher H=1 aggregates, their
selector-specific sequence manifests, and their exact K29334 allocations. It
is combined only to keep the dependency inventory at eight files; the two
profiles retain separate score hashes, position manifests, and policy
bindings. A policy file without its matching embedded comparator scores is not
verifiable and fails closed.

The static Q4/Q8 comparator is deterministically reconstructed inside the
authenticated Stage-A evaluator from the bound candidate score artifact at
the frozen `P=14739` promotion count. A separately published convenience copy
is not a ninth trusted dependency and may not be accepted without exact
reconstruction equality.

Changing any byte in any dependency requires a new binding artifact and a new
Stage-A identity candidate.

### Stage A: multi-workload falsification

Stage A contains exactly 12 examples:

- the first four SHA-ranked eligible PG19 validation books, each using 4,096
  prefill tokens followed by 128 continuation tokens, of which 127 predictions
  are exposed to the committed quantized cache;
- four RULER category representatives at configured length 4,096 and recovery
  seed 2,343: `niah_multiquery`, `vt`, `fwe`, and `qa_1`, using each
  identity-bound teacher-forced target derived from the official references;
  and
- the first four SHA-ranked HumanEval+ canonical IDs, using at most the first
  128 canonical-solution tokens after the identity-bound prompt.

Every continuation is evaluated through one-token forwards. The prefill's last
logit predicts continuation token zero before the candidate checkpoint is
committed, so that unaffected prompt-to-first-token prediction is excluded.
For a continuation of `m` tokens, the metric covers exactly the `m - 1`
cache-exposed transitions from continuation token `i` to token `i + 1`.
Identities bind the complete continuation span and this narrower half-open
cache-exposed metric span separately; `m < 2` fails closed.

RULER's `outputs` field has two different meanings. Retrieval, variable
tracing, and aggregation list multiple required answer atoms; their Stage-A
target is every atom in source order joined by the exact separator `", "`.
Question-answering rows list alternative acceptable references; their target
is the first pinned alternative in source order. The receipt binds the complete
official output array in both cases. This is a deterministic teacher-forced NLL
target for this adoption study, not a claim that comma-space joining is
RULER's official generation metric.

The exact canonical IDs, configurations, source revisions, formatter hashes,
content hashes, prompt-token hashes, target-token hashes, and half-open token
spans must be committed before weights are opened. No example may be replaced
because its result is inconvenient.

`rht_q468_dynamic_k27030` is the prespecified online local-distortion allocator
baseline. "Exact" describes only its discrete `K27030` allocation under the
frozen local objective and tie rule; it is not an oracle for sequence NLL,
downstream accuracy, generation, or globally optimal recurrent trajectories.

The primary equal-resident Stage-A contrast is static `K29334` versus dynamic
`K27030`; both own exactly `3,454,664` recurrent-resident bytes. For paired
example `e`, define

```text
d_e = excess_NLL(static_K29334, e) - excess_NLL(dynamic_K27030, e).
```

Using seed 2,339, perform 10,000 paired stratified bootstrap resamples by
resampling examples with replacement inside PG19, RULER, and HumanEval+, then
averaging the three family means equally. The one-sided 95% upper percentile
bound for `d` must be no more than `0.010` nats/token; no family point estimate
may exceed `0.015`; and static `K29334` top-1 agreement may trail dynamic
`K27030` by at most `0.005`. All conditions are conjunctive.

Static `K27030` versus dynamic `K27030` remains a selection-step-matched
diagnostic and cannot decide the equal-resident adoption claim. Static `K29334`
must also beat exact-byte `rht_q48_static_p14739` on every Stage-A workload. If
either prespecified multi-workload contrast fails, stop rather than changing the
baseline or budget after observing the result.

The candidate must separately be non-inferior to each of
`rht_q468_static_diag_empirical_fisher_h1_k29334` and
`rht_q468_static_mse_k29334` under the same `0.010` upper-bound, `0.015`
per-family, and `0.005` top-1 margins. These are selector comparisons at the
same `K29334` format and byte ledger. If either comparator wins, the frozen
query-energy candidate fails Experiment 013; the winning comparator may seed a
new experiment but may not replace the candidate post hoc.

### Stage B: development

Stage B remains closed until Stage A and every identity gate pass. It contains:

- the remaining 28 eligible PG19 validation books after Stage A;
- the original 48-cell RULER development grid at configured length 4,096: all
  52 combinations of the 13 exact configs and seeds 2,339 through 2,342 except
  the four retired seed-2,339 cells for `niah_multiquery`, `vt`, `fwe`, and
  `qa_1`; and
- the remaining 28 HumanEval+ tasks under the Stage-A/B ranking domain.

Stage-B identities and token spans are unresolved protected placeholders in
this protocol. They must be promoted and committed in a separate amendment
before Stage-B model access.

### Stage C: untouched confirmation

Stage C remains closed until the complete Stage-B decision is committed. It
contains:

- 32 SHA-ranked eligible PG19 test books;
- all 52 combinations of the 13 exact RULER configs at configured length 4,096
  and seeds 3,339 through 3,342; and
- 32 HumanEval+ canonical IDs selected from the exact eligible remainder by the
  disjoint procedure below.

Let `U_HE` be the exact 164-ID HumanEval+ canonical-ID projection bound by the
immutable projection manifest. Rank `U_HE` under
`recurquant.experiment013.humaneval-plus.stage-a-b.v1\0`, breaking a hash tie by
canonical ID. Let `H_A` be ranks 0 through 3, `H_B` be ranks 4 through 31, and
`H_AB = H_A union H_B`. Thus `H_AB` must contain exactly the 32 distinct IDs
used by Stage A/B. Define the Stage-C eligible remainder before looking up any
HumanEval+ row content:

```text
U_C = U_HE setminus H_AB
```

`U_C` must contain exactly 132 distinct IDs. Rank only `U_C`, never the complete
`U_HE`, by lowercase SHA-256 of the UTF-8 domain-separated canonical ID under
`recurquant.experiment013.humaneval-plus.stage-c.v1\0`, again breaking a hash
tie by canonical ID. Let `H_C` be ranks 0 through 31 of that remainder.

Before any Stage-C HumanEval+ row, prompt, canonical solution, token, or token
span is requested or materialized, the Stage-C identity candidate must bind:

1. the exact 164-ID projection-manifest hash;
2. the 32-entry `H_AB` exclusion manifest ordered by Stage-A/B rank, including
   each canonical ID, rank, and Stage-A/B selection hash, plus the canonical
   SHA-256 of that complete manifest;
3. the 132-entry `U_C` remainder manifest ordered by Stage-C remainder rank and
   its canonical SHA-256; and
4. the 32-entry `H_C` selection manifest ordered by Stage-C remainder rank,
   including canonical ID, rank, and Stage-C selection hash, plus its canonical
   SHA-256.

The resolver must recompute all four objects from the authenticated ID-only
projection and prove exact counts, distinctness, membership, ordering, hashes,
`H_AB union U_C = U_HE`, `H_AB intersection U_C = empty`, and
`H_AB intersection H_C = empty`. Any mismatch or overlap fails closed before
content access. There is no fallback row, replacement, or reranking.

Stage C alone decides confirmation. Stage-A and Stage-B observations may not be
pooled into a Stage-C point estimate, confidence interval, method choice,
comparator choice, effect threshold, or claim.

The Stage-C family order is PG19, RULER, HumanEval+, with exact example counts
`n_f = 32, 52, 32`. Within each family, examples use their authenticated
identity-file order. For method `j`, example `e`, and its exact `m_e` finite
cache-exposed transitions, define:

```text
x_(j,e) = (1 / m_e) * sum_t [NLL_(j,e,t) - NLL_(FP32,e,t)]
a_(j,e) = top1_agreement_count_(j,e) / m_e
X_(j,f) = (1 / n_f) * sum_(e in f) x_(j,e)
A_(j,f) = (1 / n_f) * sum_(e in f) a_(j,e)
X_j     = (1 / 3) * sum_f X_(j,f)
A_j     = (1 / 3) * sum_f A_(j,f)
```

`top1_agreement_count_(j,e)` is the exact number of transitions for which
method `j` and the matched FP32 reference have the same argmax token.
`a_(j,e)`, `A_(j,f)`, and `A_j` are exact rational values derived from integer
agreement and transition counts until the threshold comparison. Each method
must contain exactly the same identity-bound transitions and FP32 reference NLL
for an example. Missing, duplicate, reordered, non-finite, or reference-drifted
rows fail closed. Examples are equal-weighted within family and the three family
means are equal-weighted regardless of token or example counts. Token-micro NLL,
mean KL, tail KL, and maximum KL are diagnostic only and never gate.

All four primary contrasts use one shared deterministic bootstrap schedule.
Initialize exactly one NumPy `Generator(PCG64(2339))`. In PG19, RULER,
HumanEval+ order, call
`integers(0, n_f, size=(10000, n_f), dtype=np.int64, endpoint=False)` once per
family, producing matrices of shapes `10000 x 32`, `10000 x 52`, and
`10000 x 32`. Reuse those same three matrices, without reinitialization or any
additional random draw, for every contrast in this fixed order:

1. `rht_q468_static_k29334` versus
   `rht_q468_dynamic_k27030` non-inferiority;
2. superiority to `rht_q48_static_p14739`;
3. superiority to
   `rht_q468_static_diag_empirical_fisher_h1_k29334`; and
4. superiority to `rht_q468_static_mse_k29334`.

For any example-level contrast vector `c_e`, bootstrap replicate `b` first
averages the `n_f` indexed values inside each family, then averages the three
bootstrap family means equally. Sort the 10,000 replicate task-macro contrasts
ascending with a stable sort. The one-sided 98.75% upper percentile bound is
zero-based element `ceil(0.9875 * 10000) - 1 = 9874`; the one-sided 98.75%
lower percentile bound is zero-based element
`ceil(0.0125 * 10000) - 1 = 124`. Both are nearest-rank order statistics with
no interpolation.

For candidate `p = rht_q468_static_k29334` and equal-resident dynamic baseline
`d = rht_q468_dynamic_k27030`, define the non-inferiority contrast with positive
values meaning that the candidate is worse:

```text
d_e = x_(p,e) - x_(d,e)
D_f = X_(p,f) - X_(d,f)
D   = X_p - X_d
```

The upper bound is computed from bootstrap replicates of `d_e`. This contrast
passes only if that upper bound is at most `0.010` nats/token, every `D_f` is at
most `0.015`, and the exact task-macro top-1 trail `A_d - A_p` is at most
`0.005`. The bound threshold applies to the bootstrap bound; the family and
top-1 thresholds apply to their observed Stage-C point estimates.

For each superiority comparator `c` in the fixed Q48, Fisher, MSE order above,
define positive values to mean that the candidate is better:

```text
g_(c,e) = x_(c,e) - x_(p,e)
G_(c,f) = X_(c,f) - X_(p,f)
G_c     = X_c - X_p
R_c     = G_c / X_c
```

The lower bound is computed from bootstrap replicates of `g_(c,e)`. A
superiority contrast passes only if all of the following conjunctive conditions
hold: `X_c > 0`; observed point improvement `G_c >= 0.002` nats/token;
observed relative point improvement `R_c >= 0.10`; the 98.75% lower bound is
strictly greater than zero; every observed family point `G_(c,f) > 0`; and the
exact task-macro top-1 trail `A_c - A_p <= 0.005`. `X_c` is the sole denominator
of the 10% condition. If `X_c <= 0`, `R_c` is undefined and the contrast fails;
there is no absolute value, epsilon, clipping, sign reversal, or alternate
denominator. The `0.002` and `10%` thresholds apply to observed task-macro point
estimates, not bootstrap bounds. Equality passes the inclusive `0.002`, `0.10`,
and `0.005` checks but fails the strict lower-bound and every-family positivity
checks.

The dynamic non-inferiority contrast and three superiority contrasts are the
four primary hypotheses. Each uses one one-sided alpha of `0.0125`, implemented
by its 98.75% bound; Bonferroni therefore limits the family-wise alpha to
`4 * 0.0125 = 0.05`. All four hypotheses and every associated deterministic
effect-size, family-point, and top-1 gate must pass. These additional gates are
not separate confidence claims. No post-result comparator selection, alpha
reallocation, pooling, or strongest-comparator claim is permitted.

Passing Stage C supports only teacher-forced cache-exposed target-NLL and FP32
top-1-agreement statements for these exact frozen identities, spans, methods,
checkpoint, and budgets. Because KL is diagnostic rather than gated, passage
does not establish FP32 logit-distribution fidelity. It also does not establish
free generation, generated-code execution or correctness, downstream task
accuracy, selector-principle superiority, deployment, speed, novelty, or a
breakthrough.

The generated RULER IDs, auxiliary-source hashes, formatter hashes, actual
lengths, and token spans remain unresolved until a separate protected identity
amendment is frozen. Stage C may not be partially previewed.

## Methods and measurements

Every accepted quality run includes FP32 recurrent state, uniform RHT Q4 and
Q8 anchors, `rht_q48_static_p14739`, static K27030, online dynamic K27030,
`rht_q468_static_mse_k29334`,
`rht_q468_static_diag_empirical_fisher_h1_k29334`, and
`rht_q468_static_k29334`. A closest eligible published implementation is added
only through a pre-result identity amendment with its exact implementation and
byte accounting; an incompatible or unavailable implementation is documented
rather than imitated under its name.

[When Good Enough Is Optimal](https://arxiv.org/abs/2606.06034) already applies
low-bit integer arithmetic to the chunkwise matrix-inversion path of Gated
DeltaNet on Qwen3.5-family models. It targets multiplication-only inverse
approximation and kernel overhead, not mixed-precision storage of the
persistent recurrent state, but it means that low-bit Gated DeltaNet execution
itself is not a novelty claim. [SAW-INT4](https://arxiv.org/abs/2604.19157)
likewise demonstrates block-diagonal Hadamard rotation in a fused INT4 KV-cache
path under serving constraints. Its cache object differs from the fixed-size
Gated DeltaNet state, but it reinforces that rotation plus four-bit packing is
prior art and that deployment evidence must come from an integrated kernel.

Primary quality is task-macro aligned excess next-token NLL over only the
identity-bound cache-exposed transitions relative to the matched FP32
trajectory. Report task-macro and token-micro excess NLL, mean and
tail KL, top-1 agreement, local codec SSE, trajectory error, result by workload
family, resident bytes, transient bytes, peak HBM, and latency. Statistical
intervals are paired task bootstraps with 10,000 resamples and seed 2,339.

Stage-A resource fields are diagnostic rather than like-for-like deployment
measurements. Per-transition latency and CUDA allocator peaks cover the scored
one-token decode forward only; prefill latency and allocator peaks are reported
separately per method. Logical recurrent-resident bytes exclude model weights,
ordinary attention caches, allocator reservation, and temporary workspaces.
The cache-reported workspace value is a cumulative high-water sum since the
method began, including prefill, while CUDA reserved-byte peaks can retain
allocator history and method order. None of these fields may substitute for
the packed-native end-to-end deployment gate below.

Stage-A results are a falsification screen, not confirmation or selector
superiority evidence. Only the separately frozen Stage-C decision above can
support those claims. A non-positive comparator excess NLL makes a relative
superiority gate fail closed; it is not redefined.

## Packed-native deployment gate

The production path must consume the packed Q4/Q6/Q8 pools directly through a
fused Triton or CUDA implementation. A Python dequantization loop, persistent
FP32 mirror, or benchmark of an isolated helper cannot satisfy deployment.
Correctness must first match the frozen reference codec and recurrent update
within declared numerical tolerances across all three codes, offsets, boundary
groups, shapes, and deterministic fixtures.

Under a pinned hardware/software protocol and identical model loop:

- batch-1 p50 decode latency must be at most `1.05x` optimized FP32-cache;
- batch-1 p95 decode latency must be at most `1.10x` optimized FP32-cache;
- measured peak HBM must be lower than optimized FP32-cache; and
- either throughput at batch size at least eight improves by at least `10%`,
  or the maximum batch before OOM improves by at least `1.25x`.

Report warm-up, repeat count, synchronization, clock/power state, device,
driver, CUDA, kernel version, compiler flags, batch, prompt/decode lengths, and
all raw repeats. The existing uniform-kernel microbenchmark is an isolated
diagnostic only and cannot be used as end-to-end evidence.

### Opaque pre-staging boundary

The official Stage-A launcher uses two cold authenticated child processes.
Before the one-run seal, a credential-stripped network child copies and hashes
the exact pinned tokenizer, Parquet, RULER-generator, generation-manifest,
receipt, and model-metadata bytes into a content-addressed bundle without
decoding protected rows or receipt bodies. A new child then starts with
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`,
authenticates that bundle, binds its manifest SHA-256 into the seal and durable
evidence chain, and materializes Stage-A content only after reservation from
the local bundle. A Python socket audit guard makes attempted network access in
that offline child fatal.

"Opaque" describes the authenticated program's procedure, not encryption or
access control. The bundle necessarily contains raw Parquet and RULER bytes. A
human who controls the filesystem could inspect them before the seal, so this
is an honest-process one-run boundary rather than a claim of human blindness.
Stronger blindness would require independent custody, encryption, and a
post-seal key or data release outside this local evaluator.

### One-run evidence boundary

The Stage-A empty-diff seal commit, Git refs and reflogs, and the
identity-scoped lock in the repository's Git common directory make accidental
re-execution and ordinary local history changes detectable. They provide a
durable honest-process audit trail on the machine where the run is performed;
they are not a cryptographic proof that only one execution was ever possible.

The source and runtime manifests authenticate the exact canonical Git
executable bytes, file size, and normalized absolute-path digest used by the
runner. They do not recursively authenticate Git-for-Windows loaded DLLs,
helper executables, the Windows kernel, or the underlying operating system;
those remain part of the external trusted computing base. Public Hub,
GitHub, certificate, and TLS availability are also external, although every
accepted revision, manifest, and downloaded object is checked against its
frozen identity before protected execution. Reports must not describe this as
cryptographic attestation of the complete OS or toolchain.

The model-staging directory-component snapshots are honest-process race
hardening, not complete filesystem attestation. The runner snapshots every
component before and after resolution and repeats the complete path contract
before authentication, after authentication, and immediately before no-replace
publication; observed replacement or identity drift fails closed. It does not
retain kernel directory handles or perform every operation handle-relatively,
so a hostile local process or administrator could still race path-based I/O
between checks. The local OS, filesystem, and concurrently privileged processes
therefore remain in the trusted computing base. Stronger protection would
require held Windows directory handles and file IDs, or POSIX `openat`-style
no-follow operations.

A person with filesystem control can deliberately delete the lock and reflog,
rewrite or remove refs, or start from a fresh clone. The pre-run seal also
cannot authenticate a result that does not yet exist. Stronger public proof
requires an external append-only anchor for the seal before protected access
and a second external anchor or signature over the completed result bundle.
External anchoring is outside this evaluator. Accordingly, reports must call
the local controls one-run auditability or tamper evidence, never tamperproof
or cryptographically non-bypassable enforcement.

## Advancement and claim boundary

Every integrity, stability, quality, and deployment gate is conjunctive for an
adoption-ready claim. A quality pass without the deployment gate supports only
a quality/storage result. A kernel pass without the protected quality stages
supports only an implementation result.

The prior-art boundary is narrow. [RateQuant](https://arxiv.org/abs/2605.06675v2)
already fits calibration-based mixed-precision rate-distortion policies for KV
caches and makes loss-gradient sensitivity central. Its published KV
head/token implementation is not a direct Gated DeltaNet matrix-row baseline,
but the sensitivity principle transfers; the prespecified H1 comparator above
tests it without claiming to be RateQuant. [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/)
quantizes Mamba state caches, while [Quamba2](https://arxiv.org/abs/2503.22879v4)
provides quantized SSM deployment and kernels. [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791v1)
motivates the architecture family. Its official repository currently provides
training code but no tagged release or pretrained checkpoint for the reported
1.3B run. Therefore only a confirmed exact-byte static Q4/Q6/Q8 packed-native
Gated DeltaNet path plus end-to-end adoption benefit could be differentiated;
this protocol makes no novelty claim.

[MixKVQ](https://arxiv.org/abs/2512.19206) already combines query relevance
with intrinsic quantization difficulty for mixed-precision KV-cache allocation,
and [Block-GTQ](https://arxiv.org/abs/2606.24033) already uses label-free query/key
energy, marginal-gain bit allocation, and a packed serving path. Experiment
013's query-energy times measured row-MSE selector is therefore a prespecified
diagonal read-error approximation to test, not a new general allocation
principle. [LightMamba](https://arxiv.org/abs/2502.15260v2) already combines
Hadamard-assisted low-bit Mamba inference with hardware co-design and reports
that its elementwise Mamba hidden-state recurrence is not rotation-equivariant.
The potentially differentiating RecurQuant hypothesis is narrower: prove that
a Gated DeltaNet matrix state admits an exact right/value-axis orthogonal
basis, keep the persistent state in that basis, and consume its physical-row
Q4/Q6/Q8 representation directly in the recurrent update without a persistent
FP copy.
[Nemotron 3 Super](https://arxiv.org/abs/2604.12374) further makes recurrent
rounding drift and stochastic rounding mandatory baselines rather than optional
ablations.

Even a complete pass would establish only that the frozen static packed layout
was useful on the pinned 0.8B checkpoint, workloads, budgets, and hardware. It
would not establish that RHT, mixed precision, loss sensitivity, Q4/Q6/Q8,
static allocation, or packed kernels are new. It would not make RecurQuant a
new base model, prove generated-code correctness, eliminate contamination, or
justify "breakthrough," "state of the art," "lossless," or universal language.
It also would not prove a closed-loop StateLease controller; Experiment 013's
map is immutable after calibration.

A later breakthrough-level claim requires evidence beyond Experiment 013:
exact unquantized basis-equivalence tests; no-RHT and multi-seed RHT ablations;
round-to-nearest versus stochastic-rounding comparisons; uniform Q6 and static
INT8 baselines; long-horizon 4K, 32K, and 128K recurrence including free
generation; at least one larger and one independent Gated-DeltaNet-family
checkpoint; a packed-native kernel with no persistent FP mirror compared
against an optimized architecture-native baseline; fair batch-N byte and
throughput accounting; and independent reproduction on another GPU/software
stack. Until those pass, the public claim remains the exact frozen quality,
storage, and implementation result actually measured here.

Failure is a publishable result. Any change after a gate is observed creates a
new experiment number with new protected data.
