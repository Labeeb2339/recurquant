# Experiment 013: static RHT-Q468 packed-native adoption protocol

> **Status: unbound eleventh replacement-H0 amendment candidate. The tenth
> replacement H0 completed calibration-identity capture and promotion, H1,
> authenticated model staging, its sole Fisher H=1 smoke, full calibration, and
> the metadata-only Stage-A calibration authorization. Its first sealed Stage-A
> identity-capture attempt then failed closed before protected-provider
> construction or output publication because strict recursive binding
> verification reached a module-level Torch import forbidden by capture
> isolation. The tenth H0, tag, H1, identity, runtime, model root, cache,
> calibration, authorization, and populated or absent output namespaces are
> retired and non-authorizing, with no retry, adoption, or rebinding permitted.
> No Stage-A identity, Stage-A result, or quality result exists. The eleventh H0
> is not yet bound.**
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

Current eleventh replacement-H0 amendment candidate amended: 2026-08-25

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

Seventeenth pre-resolution sealed-runtime-containment, dependency, and
identity-capture-provenance amendment: 2026-08-15. Source commit
`3abaa502da47a0fa14f53e280868274f5ce17adf` passed its CI suite, but it was
never tagged, no `artifacts/experiment013/h0-3abaa502da47` H0 directory or
repository-source manifest was published, and no identity, H1, model staging,
dataset replay, model access, or quality action was performed under it. An
in-memory prospective source-manifest calculation is not an H0 artifact or an
authorization. Commit `3abaa502da47a0fa14f53e280868274f5ce17adf` is therefore
superseded as a pre-H0 candidate, not consumed or retired as an H0; there is no
tag to move or delete. A clean descendant containing this amendment and the
repairs below remains the fifth replacement-H0 candidate. Untagged draft
commits do not increment the replacement-H0 count.

A post-incident inventory found that the failed smoke described by the
Sixteenth amendment mutated the authenticated calibration-runtime root
`C:\tmp\recurquant-exp013-runtime-h0-19ef835-v4`. The sole unexpected entry
was
`base-runtime/~/.cache/huggingface/.agent_harnesses.json`, an ordinary file of
exactly 5,698 bytes with SHA-256
`df4bb03bf41e4ce850db5e7d20a9430e7bbdff173aa0941e39f0be5e8ade0906`;
its creation and last-write timestamps were both
`2026-08-14T22:11:10.9266419Z`, approximately eight seconds before the failure
output. Against calibration runtime-manifest v4 file
SHA-256
`80ca233a29af4facbb334fd4fb51a4f6e9a3d6815465cb79b9f3db63ef668d6a`,
the base-runtime inventory had exactly one extra file, no missing file, and no
changed expected file: 832 actual files rather than the authenticated 831.
The 20,201-file package inventory had no extra, missing, or changed entry.

The exact mechanism is a deterministic Hugging Face Hub 1.26.0 metadata-cache
side effect. That package defines `AGENT_HARNESSES_PATH` as
`os.path.join(HF_HOME, ".agent_harnesses.json")`; absent an explicit `HF_HOME`,
its default derives from `expanduser("~")/.cache/huggingface`. The retired
sealed environment omitted `HOME`, `USERPROFILE`, `HOMEPATH`, and `HF_HOME`.
Windows therefore preserved the literal `~`, while the child working directory
was the authenticated `base-runtime` directory. The first
`HfApi.model_info`/`list_repo_files` HTTP request-header construction invoked
`detect_agent()`, fetched `/api/agent-harnesses`, and created the relative cache
beneath `base-runtime/~`. This is an unpermitted public-metadata cache write,
not evidence of malicious tampering, model-payload access, Fisher computation,
or a quality result.

The old post-run authentication sequence did not observe the mutation at its
authoritative boundary. The child's primary RULER receipt-inventory exception
prevented child-final runtime reauthentication, and the host then raised on
nonempty scratch before reaching its own repeated runtime authentication. The
runner-v6 exception-aggregation repair preserves these failures prospectively,
but it cannot retroactively authenticate or rehabilitate the mutated root.
The entire v4 runtime root and manifest hash above are permanently retired from
official execution. Preserve them as incident evidence; deleting the extra
file, copying a clean-looking subset, renaming the root, or recapturing it in
place is forbidden and cannot authorize reuse.

A separate dependency audit found a second deterministic pre-model blocker.
The retired v4 runtime contains 38 distributions and omits `datasets`.
After the corrected RULER preflight, official calibration materialization would
reach `LiveCaptureSource.mbpp_train_rows`, fail while importing `datasets`, and
stop before local model-root authentication or model configuration and weight
access. `StagedCaptureSource` is Stage-A-only and explicitly forbids MBPP, so it
is not a calibration fallback. The next runtime must instead be prepared in a
fresh no-overwrite root with exactly 54 distributions: the prior 38-
distribution set with `fsspec` changed from `2026.7.0` to `2026.2.0`, plus this
exact 16-distribution closure:

```text
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
attrs==26.1.0
datasets==4.8.5
dill==0.4.1
frozenlist==1.8.0
multidict==6.7.1
multiprocess==0.70.19
pandas==3.0.5
propcache==0.5.2
python-dateutil==2.9.0.post0
six==1.17.0
tzdata==2026.3
xxhash==3.8.1
yarl==1.24.5
```

No dependency substitution or resolver-selected version drift is permitted.
This is deliberately a narrowed exercised-interface runtime, not a claim of
wheel-metadata dependency completeness. Torch 2.13.0+cu130 declares
`Requires-Dist: setuptools>=77.0.3`, while setuptools owns
`distutils-precedence.pth` and the sealed runtime forbids every `.pth` startup
hook. Setuptools therefore remains preparation-resolver-only and is not a 55th
staged distribution. Before runtime preparation, `uv pip check` must pass in
the source environment with that resolver-only setuptools present. In the
staged no-data probe, every official critical module must import from its
authenticated RECORD-owned path while `setuptools` and `pkg_resources` are
both absent and unimportable. Any official path that needs either module is a
hard stop requiring a new pre-access runtime amendment; it may not trigger an
ad hoc install or weakened startup-hook rule. In particular, Hugging Face Hub
remains 1.26.0, PyArrow remains 25.0.0, NumPy remains 2.4.6, Torch remains
2.13.0+cu130, and Transformers remains 5.14.1. The fresh root must be
independently captured and authenticated; its exact base and package
inventories and runtime-manifest file hash do not exist until that preparation
succeeds.

Calibration runtime-manifest schema advances from v4 to v5 and calibration-
runner revision advances from v6 to v7. Runtime-manifest v5 advances
`bootstrap_mode` from `stdlib-only-exact-runner-v1` to
`stdlib-only-exact-runner-and-capture-v2`, because the authenticated bootstrap
now admits exactly the runner or the sealed calibration-identity capture
entrypoint. It also adds exactly
`cache_confinement_mode=private-scratch-plus-explicit-dataset-root-v1` and
`child_cwd_mode=authenticated-launcher-owned-scratch-v1`; every other v4 launch
policy field remains unchanged. Each sealed child
must run with its current working directory equal to a newly created,
verified-empty, identity-recorded launcher-owned scratch directory, never a
base or package runtime root. The child inherits at most the existing five
Windows operating-system variables `SYSTEMROOT`, `WINDIR`, `COMSPEC`,
`PROCESSOR_ARCHITECTURE`, and `PROCESSOR_ARCHITEW6432`, when present, and binds
the following controlled environment exactly:

```text
LANG=C
LC_ALL=C
TEMP=<scratch>
TMP=<scratch>
TZ=UTC
HOME=<scratch>/private-home
USERPROFILE=<scratch>/private-home
XDG_CACHE_HOME=<scratch>/xdg-cache
HF_HOME=<scratch>/huggingface
HUGGINGFACE_HUB_CACHE=<scratch>/huggingface/hub
HF_HUB_CACHE=<scratch>/huggingface/hub
HUGGINGFACE_ASSETS_CACHE=<scratch>/huggingface/assets
HF_ASSETS_CACHE=<scratch>/huggingface/assets
HF_XET_CACHE=<scratch>/huggingface/xet
HF_MODULES_CACHE=<scratch>/huggingface/modules
HF_TOKEN_PATH=<scratch>/huggingface/token
TRANSFORMERS_CACHE=<scratch>/transformers
TORCH_HOME=<scratch>/torch
PYTORCH_KERNEL_CACHE_PATH=<scratch>/torch/kernels
TORCH_EXTENSIONS_DIR=<scratch>/torch/extensions
TORCHINDUCTOR_CACHE_DIR=<scratch>/torch/inductor
TRITON_CACHE_DIR=<scratch>/torch/triton
HF_DATASETS_CACHE=<cache-root>/datasets
HF_DATASETS_DOWNLOADED_DATASETS_PATH=<cache-root>/datasets/downloads
HF_DATASETS_EXTRACTED_DATASETS_PATH=<cache-root>/datasets/downloads/extracted
DISABLE_TELEMETRY=1
DO_NOT_TRACK=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_UPDATE_CHECK=1
HF_HUB_DISABLE_XET=1
```

The official `--cache-root` is the sole permitted persistent calibration-
dataset cache boundary. Before child creation it must be an absolute existing
regular directory, not a filesystem root, with no linked or reparse ancestor
or root, and disjoint in both nesting directions from every authenticated base
and package runtime root and, after their creation, the launcher-owned scratch
and bytecode roots. Alias, equality, or nesting in either direction fails. Its
normalized component and root identities are fixed and must be unchanged after
the child. All implicit home, Hub, assets, token, Xet, Transformers, Torch,
Triton, and compiler caches remain within scratch. Scratch and the bytecode
prefix must retain their recorded identities,
contain no link, reparse point, or non-regular entry at cleanup, and be absent
after identity-checked cleanup. A child exception or nonzero return remains
primary; postcondition and cleanup defects are aggregated as secondary. An
otherwise successful child fails closed on any residue or cleanup defect.

The Stage-A launcher must use the same child-cwd, private-home, cache,
telemetry, implicit-token, and Xet confinement for both cold children. Its
offline child additionally retains `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` plus the fatal socket
guard. Its permitted dataset cache is likewise rooted only below an explicitly
validated `--cache-root`. Stage A may not retain path-only temporary cleanup:
it must use the calibration launcher's fixed device/inode/type identity,
no-link/no-reparse traversal, exact-owned-root removal, primary-failure
preservation, aggregated secondary diagnostics, survival check, and partial-
creation behavior for every launcher-owned scratch and bytecode root.

At this amendment point, identity schema remained v5; capture and resolver
procedure remained v6; adapter revision remained v2; and RULER launcher v7,
generation-manifest v2, and RULER runtime-manifest v3 remained unchanged. The
new external calibration-identity-
capture-provenance receipt uses schema v1. Frozen-identity-contract and model-
staging-authorization stdout schemas advance from v1 to v2 solely to bind that
receipt's file SHA-256; model-staging-path stdout remains v1. This amendment
authorizes no byte change to the capture/resolver v6 or RULER v7 procedures.
Their exact source bytes may be retained only if the new source manifest
authenticates them unchanged. The existing RULER v7 receipt batch and its 100
raw producer files remain reusable only after complete replay authentication,
exact 21-file receipt inventory, and generation-manifest file SHA-256
`979f91848b6c0692160419c3e5e9ee555aa94d9e7add3092067f003ea0543e80`.
The public model-metadata manifest file SHA-256
`586d9c7e520f3bbd99ecef30663bf07d283eb14622475c58891becd8e033b05c`
and immutable-Parquet materialization-manifest file SHA-256
`ee5628e50e5d3516fd79077542d355fd915455ac0e53128d372f4177ad63d39c`
may likewise be reused only after every existing point-of-use byte, inventory,
path, version, and semantic check passes. The tokenizer contract and files are
unchanged. None of these reuse permissions extends to the retired v4 runtime.

The identity-capture runtime-provenance remediation audit remains open. The
diagnosis is concrete: the existing capture path authenticates an external
staged runtime manifest and context but never changes its interpreter or
`sys.path`; its later Hugging Face Hub, Transformers, `datasets`, `fsspec`, and
PyArrow imports resolve from the launching repository virtual environment.
That host environment contains `datasets==4.8.5` and `fsspec==2026.2.0`, while
the authenticated v4 runtime omitted `datasets` and bound
`fsspec==2026.7.0`. Every prior identity capture therefore lacks authenticated
live-import provenance. No claim that a prior identity was produced by the
sealed v4 runtime is permitted.

Runner v7 adds the sealed-only `capture-calibration-identity` command to close
the implementation surface before H0. Its exact runner-side option profile is:

```text
--repository-root
--source-commit
--repository-source-manifest
--expected-repository-source-manifest-sha256
--runtime-manifest
--expected-runtime-manifest-sha256
--model-file-manifest
--expected-model-file-manifest-sha256
--parquet-materialization-manifest
--expected-parquet-materialization-manifest-sha256
--cache-root
--ruler-receipt-dir
--output
--capture-provenance-receipt-output
```

Every option occurs exactly once, all path values are absolute, and capture
and ordinary-calibration profiles cannot be mixed. The unsealed runner rejects
this command. Its phase is hard-coded to calibration, and it exposes no phase,
model-root, adapter, weight, CUDA, package-root, interpreter, or Git override.
The outer authenticated launcher alone supplies the staged interpreter,
package/import roots, and Git executable.

The command must actually execute inside the freshly authenticated staged v5
interpreter and package/import environment. Merely reading or authenticating
the v5 manifest from a host virtual environment is insufficient. Before
tokenizer or dataset content access, it proves its `sys.executable`, base
`sys.path`, package roots, import paths, module origins, and exact 54-
distribution inventory equal the authenticated v5 runtime. It exact-loads the
unchanged H0 capture, resolver, source-verifier, and Parquet modules, performs
the pure 21-file RULER precondition, fixes phase to calibration, and repeats
source and runtime authentication after capture without any model-root, model-
loading, CUDA, Fisher, or protected-stage access. Identity input and receipt
paths are distinct, absent, no-overwrite destinations.

Only after the identity input is durably published may the command publish a
canonical receipt with artifact kind
`recurquant_experiment013_calibration_identity_capture_provenance`, schema
version one, capture version six, status
`captured_under_authenticated_runtime`, runner v7, H0, phase calibration, the
identity-input file SHA-256, all four execution bindings, and the exact H0
capture-source path and SHA-256. It also binds exact sorted origin records for
`datasets`, `fsspec`, `huggingface_hub`, `numpy`, `pyarrow`, `tokenizers`, and
`transformers`; each record contains module, distribution, version, package
root, relative path, file SHA-256, and size and must match both the v5 runtime
inventory and distribution RECORD ownership. Its exact
`excluded_runtime_modules` value is `["pkg_resources","setuptools"]`, and the
sealed importer must prove both names remain absent and unimportable.

The provenance receipt remains external to identity schema v5, so it creates
no seventh permitted identity-pointer change. Before H1,
`verify-frozen-identity-contract` requires
`--capture-provenance-receipt`,
`--expected-capture-provenance-receipt-sha256`, `--runtime-manifest`, and
`--expected-runtime-manifest-sha256`; it authenticates the receipt against the
identity-input hash, H0 source, four identity bindings, v5 runtime, capture
source, critical origins, and exclusions. After H1,
`verify-model-staging-authorization` requires the same four inputs and repeats
the same gate. `stage-model` also requires them and invokes the shared
authorization before download and again immediately before publication; any
receipt, runtime, source, identity, or origin drift blocks publication.

The capture and receipt implementation plus its regressions must pass before
the next H0 is tagged; implementation bytes alone are not provenance evidence.
Regressions must prove that only staged module origins are accepted; host-
virtual-environment, cwd, user-site, private-home, preloaded-module, and
`PYTHONPATH` shadows are rejected; missing `datasets`, wrong `fsspec`, or an
excluded-module import fails before network or output; capture and calibration
argument profiles cannot mix; wrong, missing, duplicate, relative, or
mismatched artifact/cache/RULER/output arguments fail at their earliest safe
boundary; source or runtime mutation prevents receipt publication; output or
receipt custody drift fails; and model root, adapter, model loading, weight,
CUDA, Fisher, Stage-B, and Stage-C surfaces remain structurally unreachable.

The replacement chain restarts from a clean descendant containing this
amendment, runner v7, runtime-v5 preparation and launch support, the exact
dependency closure, both launcher-containment repairs, the identity-capture
provenance repair, and their regressions. After clean CI, that descendant may
be tagged as the fifth replacement H0 and receive a fresh repository-source
manifest. Then, and only then, prepare the fresh no-overwrite 54-distribution
runtime and capture its v5 manifest; replay-authenticate the reusable RULER v7
batch; reauthenticate the unchanged model-metadata, Parquet, and tokenizer
contracts; execute sealed capture inside the authenticated staged v5
environment; and publish its new identity input and external provenance
receipt. Resolve and promote a new identity, then pass the schema-v2 frozen-
identity contract verifier with that exact receipt before committing only the
identity as H1. Repeat staging-path preflight and schema-v2 model-staging
authorization with the same receipt, then publish the same authenticated
three-file model payload through a new no-overwrite `stage-model` execution
into a fresh H1-bound model root. No prior receipt, H1, identity, or published
model root may be rebound or adopted. Only after that complete chain may a new
sealed Fisher H=1 smoke be attempted.

Relative to retired identity file SHA-256
`40c434d038879608093fc8f74b66893062e4f52a0e1db9d33b40ac9fa411be90`,
exactly these six JSON pointers, comprising the prior five source/promotion
cascade fields plus the new runtime-manifest binding, may differ:

```text
/canonical_evidence_sha256
/evidence/execution_bindings/calibration_runtime_manifest_file_sha256
/evidence/execution_bindings/repository_source_manifest_file_sha256
/evidence/promotion/candidate_canonical_evidence_sha256
/evidence/promotion/candidate_file_sha256
/evidence/source_manifest_sha256
```

Any seventh changed JSON pointer stops the chain. All 160 records, content-
manifest SHA-256
`ee72483a8f8b4370c9e667e4287747e5bc358aeb0265a58167140f4e780a7b29`,
split-assignment SHA-256
`a42cf4b332cc8cf58b27709d7d261fc03a356b27ec1c9ccd56914d99e60c1797`,
tokenizer file-manifest SHA-256
`e48bffe3aeaf5436b23f349a4517ebc8c8f965cd60b9566014191a7e7938f2ef`,
dataset revisions, selection, token IDs, spans, Fisher boundaries, model
contract and revision, model-file manifest, Parquet manifest, quantization
policies, metrics, gates, protected-stage rules, and every other scientific or
content field remain byte-for-byte unchanged. This amendment repairs
execution provenance and containment only. It is not a Fisher result, a
quality result, an adoption result, or a breakthrough claim.

Eighteenth pre-H0 Fisher-smoke prerequisite authentication amendment:
2026-08-23. A source-only mutation probe found that the full-calibration
prerequisite verifier authenticated the smoke report's outer canonical hash and
primary identity bindings but inspected the nested runtime and adapter records
only shallowly. After recomputing the unkeyed canonical hash, a report with an
empty GPU record, a non-float elapsed-time string, or an extra forged adapter
field could still pass that gate. No protected receipt body, model payload,
CUDA computation, calibration score, policy, stability value, or quality result
was accessed to find the defect.

The runner now requires exact runtime, GPU, adapter, and model-loading-
diagnostic field sets; canonical finite non-negative elapsed time; the frozen
Torch 2.13.0+cu130 and CUDA 13.0 identities; typed GPU identity and consistent
non-negative peak-memory counters; the exact reviewed adapter revision,
backend, dtype, shapes, recurrent layers, model contract, full-materialization
count, model-open status, empty loading diagnostics, the exact identity-bound
capture and token-sequence hashes, and the exact Fisher-step count. Rehashed
malformed receipts fail before canonical calibration materialization or model
loading. Regression coverage includes the previously accepted mutations and
additional type, counter, identity, and completion-state drifts.

This is semantic fail-closed validation, not proof that a local receipt is
unforgeable. External signing or append-only anchoring remains outside this
evaluator. A separate review also found that the identity-capture provenance
receipt can be published before launcher postconditions and cleanup are final;
that custody blocker remains open. Therefore no descendant containing only this
amendment may be tagged as H0. The fifth replacement candidate remains unbound
until capture completion is bound to successful launcher finalization and all
focused, full, and CI tests pass.

Nineteenth pre-H0 launcher-finalized capture-provenance amendment: 2026-08-23.
The capture-completion review found that runner v7 could publish a valid
provenance receipt before the outer launcher had completed child
postconditions, dataset-cache reauthentication, scratch and pycache cleanup,
and final host-side artifact authentication. A later launcher or cleanup
failure could therefore leave a receipt whose status overstated the completed
custody chain. No protected receipt body, model payload, CUDA computation,
calibration score, policy, stability value, or quality result was accessed to
find or repair the defect.

Runner v8 supersedes that publication order. The sealed child may durably
publish only the canonical identity-input bytes. It must prove that the
separate receipt destination remains absent, then emit exactly one canonical
schema-v2 provenance candidate on captured stdout; it never writes, renames,
links, or otherwise publishes the receipt destination. Schema v2 retains the
v6 capture identity and adds the exact publication contract
`sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1`. Its
status is
`captured_under_authenticated_runtime_and_launcher_finalized`. Schema v1,
runner v7, the former status, missing or extra fields, noncanonical bytes, and
any source, runtime, binding, origin, exclusion, or identity-input drift are
rejected.

The outer authenticated launcher captures child stdout without forwarding it.
It may publish those exact candidate bytes only after the child returns zero;
all child postconditions pass; the dataset cache, bound artifacts, source, and
runtime reauthenticate unchanged; both launcher-owned temporary roots are
successfully removed; the identity output still occupies its snapshotted
non-link parent and reads stably with the candidate-bound SHA-256; and the
receipt destination still occupies its original parent and remains absent. It
then uses atomic no-overwrite publication and emits only a canonical digest
summary. Child failure, postcondition failure, cleanup failure, candidate
failure, identity mutation, parent replacement, or a pre-existing receipt
leaves no finalized receipt. The receipt remains local custody evidence, not a
signature or append-only external attestation.

Focused runner and launcher regressions must exercise the successful order and
each fail-closed boundary above. Full clean-tree tests and CI must pass before
this descendant can contribute to a future H0. This amendment closes the
premature-publication blocker recorded by the Eighteenth amendment, but does
not authorize an H0: direct invocation review found that Fisher H=1 smoke and
full calibration can still be supplied a manually prepared exact model root
without themselves consuming the finalized provenance receipt. The next
pre-H0 amendment must mechanically close that downstream bypass. Until then,
no protected execution, replacement-H0 tag, identity recapture, promotion,
model staging, or quality claim is authorized.

Twentieth pre-H0 mandatory downstream capture-provenance amendment:
2026-08-23. A direct-invocation review confirmed the remaining bypass recorded
by the Nineteenth amendment: the finalized capture-provenance receipt was
required to verify the frozen identity and authorize model staging, but the
Fisher H=1 smoke and full-calibration entrypoints themselves accepted no such
input. A caller with an otherwise exact identity and manually prepared model
root could therefore reach those downstream paths without proving that the
identity input had survived the outer launcher's postconditions and owned-root
cleanup. No protected receipt body, dataset row, model payload, CUDA
computation, calibration score, policy, stability value, or quality result was
accessed to find or repair the defect.

Runner v9 requires both `--capture-provenance-receipt` and
`--expected-capture-provenance-receipt-sha256` for every official Fisher smoke
and full-calibration invocation. The direct `run_calibration` configuration
likewise requires the exact receipt bytes and explicit SHA-256; supplying only
an arbitrary digest is not an authorization. Before full identity decoding,
adapter validation or construction, calibration materialization, model-root
inspection, model loading, output creation, or CUDA work, the runner checks
canonical schema-v2 bytes; capture v6; runner v9; phase and source commit; the
launcher-finalized status and publication contract; the exact identity-input,
source, runtime, model, and Parquet bindings; the H0 capture-source hash; the
excluded-module policy; and all seven runtime-v5 critical-module origins
against package roots, distribution versions, RECORD ownership, file hashes,
and sizes. Malformed inputs fail through the public calibration error boundary.

Run-report schema advances from v2 to v3. Every smoke or full report binds the
authenticated receipt's file SHA-256 in `evidence.prerequisites`. A full run
accepts its prior Fisher smoke report only when that report is runner v9,
schema v3, and binds the same capture-provenance receipt SHA-256; a canonical
rehash around another receipt does not satisfy the prerequisite. This creates
one transitive custody chain from launcher-finalized identity capture through
smoke to full calibration. It is local authenticated evidence, not a signature
or external append-only attestation.

The outer launcher and its embedded standard-library bootstrap now require the
receipt pair for ordinary smoke and full profiles, require an absolute receipt
path at that boundary, authenticate the explicit digest and exact finalized
envelope before loading the runner, and bind the envelope's source commit to
the source-manifest H0. Host-side bound-artifact reauthentication repeats after
the child and therefore detects receipt mutation during execution. The sealed
`capture-calibration-identity` profile remains intentionally separate: it
accepts neither downstream receipt option nor a prior-smoke option, emits the
candidate receipt for host finalization, and does not attempt to consume its
own not-yet-published output.

Regression coverage must prove both ordinary modes fail before identity decode
or protected access on missing, mismatched, malformed, stale-schema, stale-
runner, former-status, wrong-source, or semantically drifted provenance; that
the host and embedded gates stop before subprocess or runner loading; that a
rehash cannot move a smoke report to another finalized receipt; and that the
capture and Stage-A profiles do not acquire downstream-only options. Focused,
full clean-tree, and CI tests must all pass before this descendant can
contribute to a future H0.

Runner revision is v9, capture procedure remains v6, capture-provenance schema
remains v2, frozen-identity schema remains v5, calibration-runtime schema
remains v5, and run-report schema is v3. No record selection, dataset revision,
token span, Fisher boundary, model contract, quantization policy, metric,
statistical gate, or claim boundary changes. This amendment closes the known
downstream provenance bypass only. It is not a Fisher result, calibration
result, quality result, deployment result, novelty result, or breakthrough
claim. No H0 tag, identity recapture, promotion, model staging, protected run,
or result publication is authorized by this working copy.

Twenty-first pre-H0 custody amendment candidate: 2026-08-23. The passing full
calibration directory now emits
`stage-a-calibration-core-binding.json`, a schema-v3 core binding that is part
of the runner-v9 report's artifact inventory but is deliberately ineligible
for Stage A. After that directory and the earlier Fisher H=1 smoke directory
are finalized, the metadata-only `authorize-stage-a-calibration` procedure
requires their exact closed inventories, explicit SHA-256 values for the full
report, smoke report, capture receipt, frozen calibration identity, repository
source manifest, calibration-runtime manifest, and model-file manifest, and
the explicit H0. It accepts no model root, protected input, or CUDA option.

The resulting schema-v1 authorization artifact embeds and reauthenticates the
full runner-v9 report, launcher-finalized capture receipt, Fisher smoke report
and marker, calibration completion marker, core binding, Q48 convenience
policy, and the exact source/runtime/model manifest bytes. The full report
must bind the exact hashes of all nine calibration outputs. Its repository
receipt uses the source manifest's canonical self-digest, which is distinct
from the frozen identity input's source-manifest SHA-256. Its identity,
repository, runtime, model-manifest, Parquet-manifest, capture-receipt, and
smoke-report commitments must agree with those embedded manifests and the
frozen identity.

Authorization rederives the exact six calibration counters from the frozen
records and Fisher boundaries; the frozen query-energy constants; the exact
model receipt; the runtime-distribution and complete-file counts; Torch,
CUDA, GPU and reviewed-adapter receipts; full/smoke identity parity; the smoke-
only stability receipt; and the full split-half stability record. The capture
receipt's source hash must equal the H0 source-manifest entry, its excluded
modules must be exact, and all seven sorted critical-module origins must match
runtime-tree bytes and distribution RECORD ownership. The Q48 convenience
policy must share the core policies' H0 and byte-equal deterministic
reconstruction from the bound candidate scores at exact `P=14739`; a
self-consistent alternative allocation is rejected. The procedure atomically
publishes that authorization, a schema-v4 Stage-A binding containing it, and a
distinct completion marker in a new directory; it never alters the finalized
calibration or smoke directories.

Only the schema-v4 binding is Stage-A eligible. Its resolved nine-field
identity binding includes the authorization artifact SHA-256 in addition to
the eight calibration dependency hashes. Identity capture and promotion
strictly deserialize that receipt. Stage-A capture verifies it before runtime,
source, tokenizer, or dataset providers are called, and both capture and the
resolver require the authorization execution bindings to equal the Stage-A
identity input's execution bindings. The non-promotion resolver CLI verifies
the binding before reading `--input`; the promotion CLI does the same before
reading candidate bytes. The Stage-A screen compares the
authorization H0 and execution bindings with the Stage-A source and identity
immediately after binding verification and before any model-root access.
Opaque input staging binds the complete schema-v4 binding file; launcher
preflight, reservation, seal, and final result evidence continue to bind that
complete file hash. A schema-v3 core binding, missing authorization, changed
marker, changed prerequisite, changed output byte, cross-chain execution
binding, or rewrapped dependency fails closed.

This amendment does not create a launcher-finalized provenance receipt for the
later live Stage-A identity-capture operation itself. That separate receipt
must bind the Stage-A capture child, source/runtime/schema-v4 binding,
postconditions, cache reauthentication, owned scratch/pycache cleanup, and
host-side no-overwrite publication before live Stage-A identity capture or
promotion is authorized. Therefore the next permitted operation after a green
metadata-only test audit is creation and verification of the post-calibration
authorization and schema-v4 binding; no H0 tag, live Stage-A identity capture,
promotion, model staging, Stage-A run, protected read, or result publication is
authorized by this working-copy amendment.

Twenty-second pre-H0 custody amendment candidate: 2026-08-23. Runner v10 adds
an exact, non-mixable `capture-stage-a-identity` profile alongside the existing
calibration-capture profile. It requires the schema-v4 Stage-A calibration
binding and its explicit SHA-256, strictly authenticates the binding's embedded
authorization, H0, calibration prerequisites, and four execution bindings
before constructing a live capture source, and calls the Stage-A capture API
with the exact authenticated binding bytes. The direct capture-script CLI is
not a custody boundary and therefore rejects Stage-A capture before reading a
binding, manifest, protected input, tokenizer, dataset, or model path. The
sealed runner emits one canonical receipt candidate only; it cannot publish
the launcher-finalized receipt that attests to conditions outside the child.
Before publishing even the identity input, the authenticated resolver applies
the same exact raw-input validator later used by candidate construction. It
requires the complete input field set, frozen dataset revisions and contracts,
tokenizer contract, four execution bindings and fixed Parquet hash, authorized
binding parity, normalized record hashes, exact 12-row cardinality, order,
family allocation, RULER schedule, token spans, and calibration binding. Empty,
partial, extra-field, forged-selection, or cross-chain input cannot acquire an
identity file or receipt candidate. This pre-finalization gate accepts no
receipt parameter; candidate construction still requires the separately
launcher-finalized receipt.

The outer launcher binds the child command, source and runtime inventories,
model and Parquet manifests, cache root, identity output, schema-v4 binding,
and absent receipt destination before execution. A zero child exit is
insufficient. The host must first verify scratch and pycache postconditions,
reauthenticate the cache and every bound artifact, clean only its owned roots,
repeat reauthentication after cleanup, stably reread the identity output,
strictly validate the single canonical candidate, and confirm that the receipt
destination still does not exist. It then performs atomic no-overwrite
publication and stably rereads and semantically revalidates the published
bytes. Any child failure, residue, link or parent substitution, artifact drift,
identity drift, malformed or extra stdout, cleanup failure, or publication
collision leaves no accepted receipt.

The finalized Stage-A capture-provenance receipt uses schema v1 and binds
capture v6, runner v10, phase, H0, identity-input hash, schema-v4 binding hash,
embedded authorization hash, the exact four execution bindings, authenticated
capture-source hash, critical module origins, excluded runtime modules, and
the host-finalization publication contract. Stage-A resolution and promotion
must receive both the exact canonical receipt and its explicit SHA-256 and
strictly verify all relationships before reading identity-input or candidate
bytes. Stage-A candidate and frozen promotion evidence advance to schema v6
only to bind this receipt; calibration identity remains schema v5. A receipt
from another H0, input, binding, authorization, source, runtime, model,
Parquet materialization, or publication route is ineligible even if its own
hash and JSON structure are internally consistent.

The official Stage-A execution chain advances to runner v4, attempt schema v3,
identity-attempt-lock schema v4, and execution-artifact schema v4. The host,
embedded bootstrap, and screen CLI all require the exact receipt path plus its
explicit SHA-256. They authenticate the schema-v6 frozen identity, schema-v4
binding, and flat finalized receipt before identity-bound artifact, protected
provider, or model-root access. The receipt SHA is committed by the
`Stage-A-Capture-Provenance` seal line and repeated in reservation, attempt,
identity lock, one-run evidence, execution dependencies, verifier,
publication, recovery, input-bundle authentication, and materialization.
Legacy schema-v5 Stage-A identity or a missing, altered, rehashed, or
cross-chain receipt fails before the one-run boundary. Calibration identity
and its historical schema-v5 consumers remain unchanged.

Security-critical Python loading on this path is exact-byte execution, not a
hash-then-import pathname check. The capture helper validates the canonical
source manifest before loading the calibration runner; the metadata-only
authorizer loads the resolver from that already explicit-hash-authenticated
manifest; the Stage-A host loads the calibration launcher from one stable
authenticated byte buffer; and the screen's exact-module loader does the same
for its resolver, capture, calibration-runner, source-verifier, and gate
modules. Each compiles and executes the held bytes after occupying a controlled
module name. Same-path swaps cannot execute a second buffer before a later
postcheck. The Stage-A host likewise parses the exact runtime-manifest bytes
first authenticated by the identity and exact-compares them again during
post-child reauthentication.

This code does not claim to solve every hostile-host filesystem race. The
operating system still starts the already authenticated Python and Git
executables by pathname; preventing a privileged or same-user writer from
replacing those files between the final identity check and process creation
requires an immutable/ACL-protected runtime or an OS-specific held-handle
launch contract. A future live H0 must stage these executables in a location
that the experiment account cannot mutate during the run, and record that host
protection. This residual host-administration requirement is not waived by the
Python-level exact-byte loader fixes.

This amendment changes custody mechanics only. It does not change the frozen
record set, token spans, model contract, Q4/Q6/Q8 layouts, Fisher boundary,
allocation, byte budget, metric, statistical gate, or claim boundary. Its
tests use synthetic metadata and monkeypatched providers only. It neither
creates nor authorizes an H0 tag, post-calibration authorization from a real
chain, live identity capture, promotion, model staging, calibration, Stage-A
execution, protected read, quality result, deployment claim, novelty claim, or
breakthrough claim.

Twenty-third pre-H0 executable-launch-custody amendment candidate: 2026-08-23.
The preceding exact-byte loaders still left a final path-launch interval for
the authenticated staged Python executable and canonical Git-for-Windows
executable. On the evaluated Windows host, calibration launcher v11 now opens
both exact regular non-reparse files with `CreateFileW`, `GENERIC_READ`, and
only `FILE_SHARE_READ`. The handles are non-inheritable and remain open across
a second complete runtime/source/binding authentication, child creation, every
sealed child, postconditions, owned-root cleanup, capture-receipt publication,
and the final host reauthentication. While each handle is held, probes that
request data/append-write or delete/rename access must fail with sharing
violation or an already stronger ACL denial. Attribute-only access is outside
the share-mode claim. File identity, size, attributes, and SHA-256 are rechecked
under custody, and `GetFileInformationByHandle` must bind each still-live
handle to the authenticated volume, file index, and size. A missing, dead, or
substituted handle, early release, byte or path-identity drift, or successful
conflicting open fails closed. Entry-verification failure releases every handle
before surfacing the failure.

Stage-A launcher v5 uses the same exact-byte-loaded calibration-launcher
primitive and holds one custody pair across both its network preparation child
and offline execution child. This protects the two executable path endpoints
named by the preceding amendment; it does not recursively attest Git DLLs,
Python DLLs, the Windows loader, the kernel, or privileged host administration.
Those components remain in the disclosed external trusted computing base.

Calibration runtime-manifest schema advances from v5 to v6 and adds exactly
`executable_custody_mode=platform-held-launch-handles-v1` to the sealed launch
policy. Calibration runner revision advances from v10 to v11 and Stage-A
runner revision advances from v4 to v5. Every downstream identity and result
already binds the exact runtime-manifest file, so the custody mode is recorded
without adding an outcome field or changing a scientific identity. The
per-process raw handles, their kernel identities, and acquisition outcome are
enforced in memory and are not serialized into downstream artifacts; the
artifacts attest the required custody policy, not a replayable OS handle. On
POSIX, the same source holds read descriptors only so the cross-platform unit
suite can exercise lifecycle and cleanup; such a run is explicitly ineligible
for Experiment 013 protocol evidence. The frozen live runtime remains Windows.

No prior runtime manifest is schema-v6 eligible. The next H0 must receive green
focused, full, and CI verification for these exact bytes before tagging. Only
after that H0 exists may a fresh no-overwrite runtime be prepared, the source
manifest be captured, or an identity be recaptured. This working copy does not
authorize protected data access, model staging/loading, Fisher smoke,
calibration, Stage-A execution, quality publication, deployment, novelty, or a
breakthrough claim.

Twenty-fourth replacement-H0 capture-isolation and RULER-custody amendment
candidate: 2026-08-23. Source commit
`fe88548ec58c4547456360d45fffdca3fba8ccf9` and tag
`experiment013-h0-fe88548` became the fifth replacement H0 after their focused,
full, and CI verification passed. A fresh repository-source manifest and a
fresh runtime-manifest-v6 environment were bound to that H0. Its first official
sealed calibration-identity capture then failed closed. Transformers 5.14.1
used an availability path that imported Torch and its descendants during the
tokenizer-only capture process; the existing postcondition detected those
forbidden model/CUDA modules and rejected the run. Independently, the standard
Hugging Face Hub cache layout created an internal snapshot-to-blob symbolic
link under the launcher-owned scratch tree, so strict scratch cleanup also
refused completion. These are capture-containment defects, not scientific
results or infrastructure interruptions.

The failed attempt published neither its calibration identity-input file nor
its schema-v2 capture-provenance receipt. It created no candidate or frozen
identity, no H1, no model root, no Fisher smoke or calibration artifact, and no
quality result. No model weight was downloaded, opened, hashed, or loaded. The
preserved incident scratch root is exactly
`C:\Users\Labeeb\AppData\Local\Temp\recurquant-exp013-sealed-scratch-xw7c42_d`.
Its observed Hub link was the dataset snapshot `README.md` pointing relatively
to the corresponding content-addressed blob; the tree is retained as incident
evidence and must not be adopted as an experimental cache or silently deleted
by an official retry. The H0 commit, tag, source manifest, runtime manifest,
failed command, absent outputs, and scratch root remain historical evidence,
but H0 `fe88548ec58c4547456360d45fffdca3fba8ccf9` is retired and non-authorizing.
It permits no retry, identity promotion, H1, model staging, protected access,
Fisher smoke, calibration, or Stage-A execution.

Calibration runner v12 repairs only the identity-capture boundary. Around the
entire capture call it hides top-level `torch` from the pinned Transformers
`importlib.util.find_spec` availability probe and installs an independent
meta-path blocker that fails any real import of Torch or another forbidden
model/CUDA module. The exact ordered meta-path is held in a mutation-rejecting
scope with that blocker once at index zero. Guarded `__import__` and
`_find_and_load` entrypoints reject an import before finder dispatch if the
topology or either entrypoint changed, including a transient or self-removing
preceding finder. Installed-distribution metadata is not hidden.
Activation requires that the hidden module is not already loaded. The scoped
`importlib.util.find_spec` replacement, guarded import entrypoints, exact
meta-path topology must remain identity-equal throughout capture. All guarded
import state must be restored exactly in `finally`, and every forbidden module
must remain absent; preloading, a direct import attempt, policy mutation,
restoration failure, or post-capture residue fails closed. This is not a general
environment override: it applies only to the tokenizer-and-dataset identity-
capture command. Model staging, Fisher smoke, calibration, and Stage A retain
their ordinary authenticated Torch contract.

The capture profile also assigns both Hub-cache environment variables to the
single exact subroot `<authenticated --cache-root>/hub`. The outer launcher and
stdlib-only bootstrap must derive and exact-compare that same absolute path.
The explicit cache root and its ancestors remain subject to the existing
non-redirected directory-identity checks. The link-bearing, content-addressed
Hub cache is routed below that authenticated external `hub` endpoint rather
than the owned scratch tree; its contents and link targets are not recursively
custody-checked. `HF_HOME`, token state, assets, Xet, modules, ordinary home,
and every other private cache remain under owned scratch and must still clean
to empty. Ordinary calibration does not receive this capture-only Hub routing.
The sealed bootstrap policy advances to
`stdlib-only-exact-runner-and-capture-v3` and cache confinement to
`private-scratch-plus-explicit-dataset-and-capture-hub-root-v2`.
Runtime-manifest schema v6 is
unchanged, but its exact launch-policy value changes, so the retired H0 runtime
cannot be rebound and a fresh no-overwrite runtime manifest is mandatory.

For reuse of the fixed RULER v7 batch, the sole authoritative bundle is exactly
21 shallow regular non-link, non-reparse files: canonical
`generation-manifest.json` plus the 20 named receipt files. The generation-
manifest file SHA-256 is frozen as
`979f91848b6c0692160419c3e5e9ee555aa94d9e7add3092067f003ea0543e80` and
must be enforced as an explicit expected value, not merely copied into a later
identity. The authenticated generation manifest binds every accepted receipt
filename, size, SHA-256, command, validation payload, producer source manifest,
runtime, launcher, and pinned RULER identity. The 20 raw sibling roots and their
100 producer files remain archived producer diagnostics only. They are not
scientific inputs, are not authorization evidence, must not be read by identity
capture, and need not replay-authenticate for this reuse decision.

This rule explicitly supersedes the Thirteenth and Sixteenth amendments wherever
they require replay or authentication of those 100 raw files. It also
supersedes the Thirteenth amendment's requirement that the current capture and
resolver Git blob OIDs equal the old batch-producing capture blob
`43e64f3f4f72256de8eb58f3f4cd9068ef3fe305` and resolver blob
`dd579415f694d5900e1abdc0f46af358b2a8628b`. The producer identities remain
authenticated from the exact frozen generation manifest. Current consumer
capture and resolver bytes instead must equal their entries in the replacement
H0 repository-source manifest; their security repairs do not require RULER
regeneration. Any contradiction with the earlier raw-replay or consumer-blob-
equality language is resolved in favor of this Twenty-fourth amendment.

Phase isolation remains exact. Calibration may parse the complete generation
manifest and authenticate its public 20-file commitment table, but it opens,
base64-decodes, and semantically validates only the 16 calibration receipt
files. The four separate Stage-A receipt bodies remain unopened; their embedded
protected values in the complete manifest remain uninterpreted strings and
objects during calibration. Stage-A capture later opens and semantically
validates exactly its four receipts and must not open the 16 calibration
receipts. A generic replay of all raw producer files or all 20 receipt bodies is
therefore forbidden rather than an additional integrity check.

Before calibration content access, the sealed chain must snapshot the exact
shallow 21-name inventory and regular non-link/non-reparse file types without
opening receipt bodies, then authenticate the generation-manifest size and
SHA-256. It may next open and hash only the active phase's receipt bodies:
exactly 16 for calibration, each matched by size and SHA-256 to its embedded
manifest commitment. The four Stage-A receipt names, sizes, and hashes remain
authenticated only as uninterpreted commitments in the frozen generation
manifest until Stage A; calibration must not open or hash those four files.
After capture returns, and before a capture-provenance receipt can authorize
identity verification or promotion, the chain must repeat and exact-compare the
shallow name/type inventory, generation-manifest size and SHA-256, and only the
same 16 active receipt sizes and SHA-256 values. Stage-A capture applies the
dual rule to its four receipt bodies while leaving the 16 calibration bodies
unopened. Any inventory drift or active-phase byte drift fails closed; an
identity input left by such a failure is non-authorizing and may not be
promoted.

Runner revision advances from v11 to v12. Capture procedure v6, resolver
procedure v6, identity-input/candidate/frozen schema v5, runtime-manifest schema
v6, capture-provenance schema v2, RULER launcher v7, RULER generation-manifest
schema v2, and Stage-A runner v5 remain unchanged. The repair changes no record
selection, tokenizer files or token spans, model contract, Q4/Q6/Q8 policy,
Fisher boundary, byte budget, metric, statistical gate, or claim boundary.

The chain must now restart from a clean descendant containing this amendment,
runner v12, both containment repairs, the authoritative-RULER check, the
pre/post bundle custody check, and their regressions. Only after focused, full,
and CI verification may that descendant be tagged as the sixth replacement H0.
It then requires a fresh H0-scoped repository-source-manifest path, a fresh
no-overwrite runtime root and manifest bound to the changed launch policy, and
fresh absent identity-input and capture-provenance output paths. The retired
H0 source/runtime/capture paths may not be renamed, copied, adopted, or rebound.
Before H1, the new promoted identity must pass the complete frozen-identity
contract and, relative to retired identity file SHA-256
`40c434d038879608093fc8f74b66893062e4f52a0e1db9d33b40ac9fa411be90`,
must differ at exactly these six JSON pointers and nowhere else:

```text
/canonical_evidence_sha256
/evidence/execution_bindings/calibration_runtime_manifest_file_sha256
/evidence/execution_bindings/repository_source_manifest_file_sha256
/evidence/promotion/candidate_canonical_evidence_sha256
/evidence/promotion/candidate_file_sha256
/evidence/source_manifest_sha256
```

All 160 records, content-manifest SHA-256
`ee72483a8f8b4370c9e667e4287747e5bc358aeb0265a58167140f4e780a7b29`,
split-assignment SHA-256
`a42cf4b332cc8cf58b27709d7d261fc03a356b27ec1c9ccd56914d99e60c1797`,
tokenizer file-manifest SHA-256
`e48bffe3aeaf5436b23f349a4517ebc8c8f965cd60b9566014191a7e7938f2ef`,
and every other scientific/content field must remain byte-identical. Only that
verified promoted identity may be committed as the sole H1 tree change. Until
the replacement H0, fresh source/runtime/capture chain, six-pointer diff, and
H1 are complete, no model staging or protected execution is authorized. This
amendment is containment and provenance work only; it is not a calibration,
quality, adoption, novelty, deployment, state-of-the-art, or breakthrough
result.

Twenty-fifth replacement-H0 authenticated-import-topology amendment candidate:
2026-08-23. Source commit
`f8dc9d1066bd9f6df666e74816d704ceeb1f38da` and tag
`experiment013-h0-f8dc9d1` became the sixth replacement H0 after CI, repository-
source-manifest verification, fresh runtime preparation and verification, and
fresh model-metadata-manifest capture and verification passed. The source,
runtime, and metadata-only model artifacts were therefore valid prerequisites,
not identity or scientific results. No model weight was downloaded or opened.

The sole official sealed calibration-identity capture under that H0 failed
closed on the pinned, authenticated runtime import path. During the
Datasets/Pandas import chain, authenticated `six.py` line 1003 executed
`sys.meta_path.append(_importer)`, where `_importer` was its
`_SixMetaPathImporter` instance. Runner v12's capture guard recorded and
rejected that attempted append before the topology changed. This is a
capture-containment compatibility defect:
the observed append came from authenticated dependency code, but the current
policy has no preregistered safe treatment for it. It is not a dataset,
tokenizer, calibration, quality, or model result.

The failed child published neither an identity-input file nor its schema-v2
launcher-finalized capture-provenance receipt. It created no candidate or
frozen identity, no H1, no model root, no Fisher smoke or calibration artifact,
and no Stage-A or quality result. No staged model root was created. The launcher
completed cleanup of its owned
scratch root; unlike the prior fifth-H0 incident, no failed owned-scratch tree
is preserved for reuse or inspection. The commit, tag, green prerequisite
artifacts, failed command, and absent outputs remain historical evidence only.
H0 `f8dc9d1066bd9f6df666e74816d704ceeb1f38da` and tag
`experiment013-h0-f8dc9d1` are retired and non-authorizing. They permit no
capture retry, identity promotion, H1, model staging, protected access, Fisher
smoke, calibration, or Stage-A execution, and their H0-scoped paths may not be
renamed, copied, adopted, or rebound into a replacement chain.

Before a seventh replacement H0 may exist, a clean descendant must record and
implement a narrow treatment for this authenticated `six.py` importer behavior
without generally permitting `sys.meta_path` mutation. Its regression must
exercise the pinned Datasets/Pandas import path that reaches `six.py` line 1003,
while the existing adversarial cases for preceding, transient, self-removing,
reordered, replaced, or unauthenticated finders continue to fail closed. The
Torch-availability isolation, actual forbidden-import blocker, exact restoration
requirements, external authenticated Hub-cache routing, and phase-scoped RULER
custody remain mandatory. Any resulting runner or launch-policy revision must
advance consistently across runner, launcher, resolver, runtime manifest, and
their verifiers and tests. The repair advances the calibration runner from v12
to v13. Capture procedure v6, resolver procedure v6, identity-input/candidate/
frozen schema v5, runtime-manifest schema v6, capture-provenance schema v2,
RULER launcher v7, RULER generation-manifest schema v2, and Stage-A runner v5
remain unchanged.

Only after focused, full, and CI verification may that clean descendant be
tagged as the seventh replacement H0. The chain must then use fresh H0-scoped,
no-overwrite repository-source, runtime, model-metadata, identity-input, and
capture-provenance paths. It must repeat every source, runtime, immutable model-
metadata, RULER-custody, and capture gate; success may not be inferred from the
sixth H0's green prerequisites. Before H1, the promoted identity must still
pass the complete frozen-identity contract and the Twenty-fourth amendment's
exact six-pointer-only diff from the retired identity. Until all of those gates
pass, no identity promotion, model staging, protected execution, or scientific
claim is authorized.

Twenty-sixth replacement-H0 canonical-stdout amendment candidate: 2026-08-23.
Source commit `10ce582659e0d273adc294b593ce078b39265011` and lightweight
tag `experiment013-h0-10ce582` became the seventh replacement H0 after the
focused and full suites, package gates, exact GitHub CI run `32612241548`, and
fresh prerequisite verification passed. Its repository-source-manifest file
SHA-256 is
`94ab62b546f1cbaa530e7be7034caa6b3869e87c483da869829610c1c5d37327`
with canonical self-hash
`9b36bd6d776b4d0b12d77acedb9637419da400d653717eadb49ff46da6ba162b`;
its fresh runtime-manifest file SHA-256 is
`0d710149bc04c0170dc2f99531247a458b9d155336433cbd3e93d94cb824cb26`;
and its metadata-only model-manifest file SHA-256 is
`586d9c7e520f3bbd99ecef30663bf07d283eb14622475c58891becd8e033b05c`.
Those are prerequisite artifacts, not identity or scientific results. No model
weight was downloaded or opened.

The sole official sealed calibration-identity capture under that H0 completed
its child-side record construction and atomically published a canonical
schema-v5 identity input: 288,838 bytes, 160 records, file SHA-256
`4b9aadbe7afcb1d33f77ea8244570aac9784a75edf21cb461470b37a8ce1f459`,
and `model_weights_loaded=false`. The runner then emitted its canonical
schema-v2 capture-provenance candidate with text-mode `print`. On Windows, the
stdout pipe translated the candidate's single terminal LF to CRLF. The outer
launcher retained its exact byte contract and failed closed with
`SealedLaunchError: capture provenance candidate is not canonical JSON` before
publishing the final provenance receipt. That rejection is correct and must not
be weakened by newline normalization or tolerance for trailing bytes.

The identity input is failed-attempt evidence only. Its finalized provenance
receipt is absent, so it cannot be verified, promoted, renamed, copied, adopted,
or rebound into another chain. It created no candidate or frozen identity, H1,
staged model root, Fisher smoke, calibration artifact, Stage-A artifact, quality
result, or scientific claim. New owned scratch and pycache were cleaned; only
the separately preserved fifth-H0 scratch remains. H0
`10ce582659e0d273adc294b593ce078b39265011` and tag
`experiment013-h0-10ce582` are retired and non-authorizing, with no capture
retry permitted under either.

Before an eighth replacement H0 may exist, a clean descendant must advance the
calibration runner from v13 to v14 and write the already-canonical provenance
payload through binary stdout without platform newline translation. The write
must be exact and flushed; an unavailable binary stream, non-integer or partial
write, flush failure, child failure, or cleanup failure must fail closed. Tests
must exercise a translating text stream, a real Windows pipe, unavailable and
failing binary streams, and the outer launcher's rejection of CRLF and every
extra byte. The launcher must continue comparing the exact canonical bytes and
must not normalize them.

This repair changes only runner revision and output transport. Bootstrap mode
`stdlib-only-exact-runner-and-capture-v3`, cache-confinement mode
`private-scratch-plus-explicit-dataset-and-capture-hub-root-v2`, capture and
resolver procedure v6, identity-input/candidate/frozen schema v5, runtime-
manifest schema v6, capture-provenance schema v2, RULER launcher v7, RULER
generation-manifest schema v2, publication contract v1, and Stage-A runner v5
remain unchanged. No record selection, tokenizer byte or span, model contract,
Q4/Q6/Q8 policy, Fisher boundary, byte budget, metric, statistical gate, or
claim boundary changes.

Only after focused, full, package, and exact-commit CI verification may that
clean descendant be tagged as the eighth replacement H0. It must use entirely
fresh, H0-scoped, no-overwrite repository-source, runtime, model-metadata,
identity-input, and provenance paths and repeat every prerequisite gate. The
seventh-H0 input and artifacts cannot authorize or substitute for any step.
Exactly one official sealed metadata-only capture is then permitted. Before H1,
the promoted identity must pass the complete frozen-identity contract and the
Twenty-fourth amendment's exact six-pointer-only diff from the retired identity.
Until then, model staging, protected access, Fisher smoke, calibration, Stage A,
and every breakthrough or scientific-result claim remain unauthorized.

Twenty-seventh replacement-H0 resolver-container and phase-Hub-routing
amendment candidate: 2026-08-24. Source commit
`50d3a1b08b2905b51fcab86b715be66289df5da0` and lightweight tag
`experiment013-h0-50d3a1b` became the eighth replacement H0 after the required
focused, full, package, exact-commit GitHub CI run `32615036536`, repository-
source, fresh runtime, and metadata-only model-manifest gates passed. Its
repository-source-manifest file SHA-256 is
`6d015f83ad01e93131d99b7389445c0f850bcfcbba8a6dc1406fa1041b6e4081`;
its runtime-manifest file SHA-256 is
`0d710149bc04c0170dc2f99531247a458b9d155336433cbd3e93d94cb824cb26`;
and its metadata-only model-manifest file SHA-256 is
`586d9c7e520f3bbd99ecef30663bf07d283eb14622475c58891becd8e033b05c`.
The sole sealed identity capture and launcher-finalized provenance flow
succeeded, the schema-v5 identity was independently resolved and promoted, and
the exact identity file SHA-256 is
`e782e47855979fdb0189b69898b31d13d736aa2c86bb7d1cf69d209da2426ec1`.
Committing only that identity produced H1
`d4573a34116d7319f1ff1f23c63bb109ededd4ac`.

Authenticated model staging under H1 passed both byte-identical path
preflights, its authorization gate, and the exact three-file no-link inventory.
The staged root
`C:\tmp\recurquant-exp013-model-h1-d4573a34` contains exactly
`config.json`, `model.safetensors-00001-of-00001.safetensors`, and
`model.safetensors.index.json`, totalling 1,746,996,407 bytes. These are
prerequisite and custody facts, not scientific results.

The one permitted Fisher H=1 smoke loaded the authenticated model and completed
the first sequence's causal/Fisher endpoint capture. It then failed before the
first reduced row could be accepted or published. The resolver deliberately
returned a recursively immutable record; the runner made only a shallow copy
and normalized its `fisher_boundary`, leaving the nested `token_span` as a
`MappingProxyType`. The production identity-record hasher passed that record to
`json.dumps`, which failed with `TypeError: Object of type mappingproxy is not
JSON serializable`. A reproducing resolver-to-reducer regression also exposed
a second masked container mismatch: immutable `anchor_positions` reached a
list-specific inequality check. Neither defect changes or calls into the
quantization method, scores, allocation, or statistical gates.

The declared smoke output
`C:\tmp\recurquant-exp013-fisher-h1-smoke-h1-d4573a34` is absent, and no hidden
publication staging sibling remains. No smoke report, completion marker,
aggregate, fitted policy, calibration binding, Stage-A artifact, or quality
result was published. The attempt therefore yields no positive or negative
scientific evidence about static RHT-Q468.

The child also caused Hugging Face Hub 1.26.0 to create its ordinary relative
snapshot-to-blob link for the MBPP dataset card under launcher-owned scratch.
The launcher correctly refused to traverse or delete that reparse point,
preserved child return code 1, and retained the incident tree at
`C:\Users\Labeeb\AppData\Local\Temp\recurquant-exp013-sealed-scratch-qy2au1k0`.
Its sole link is
`huggingface\hub\datasets--google-research-datasets--mbpp\snapshots\4bb6404fdc6cacfda99d4ac4205087b89d32030c\README.md`,
with relative target
`..\..\blobs\476c2286e2ef2713058cade8be49c2f9c1514055`. The tree is incident
evidence: it must not be mutated, deleted, adopted as a cache, or rebound into
a replacement chain.

H0 `50d3a1b08b2905b51fcab86b715be66289df5da0`, its tag, H1
`d4573a34116d7319f1ff1f23c63bb109ededd4ac`, the frozen identity, and the staged
model root are retired and non-authorizing. The failed smoke may not be retried
under that chain. Its model root may not be renamed, copied, adopted, or
rebound. The failed output name and every H0/H1-scoped publication path remain
historical and unavailable to a replacement attempt.

Calibration runner v15 repairs only these execution boundaries. Canonical
identity JSON now recursively materializes verified `Mapping` and non-text
`Sequence` containers while leaving unsupported leaf types to fail closed;
plain JSON and resolver-frozen records must produce byte-identical canonical
bytes and existing hashes. Anchor-position validation now accepts mutable or
immutable non-text sequence containers only after exact integer-type and
ordered-value checks. Resolver immutability, identity-record hash domains, and
schema v5 remain unchanged.

Cache confinement advances to
`private-scratch-plus-explicit-dataset-and-phase-hub-roots-v3`. Capture children
retain the authenticated `<cache-root>/hub` endpoint. Ordinary calibration and
Stage A route both Hub-cache variables to their already authenticated
`<cache-root>`, matching the explicit cache passed to their materializers.
`HF_HOME`, token, assets, Xet, modules, Transformers, Torch, compiler caches,
and private home remain in owned scratch. External Hub contents may retain
their standard content-addressed relative links; the authenticated cache root
and ancestor identities are still checked before and after each child. The
owned-scratch no-link/no-reparse postcondition and identity-checked cleanup are
not weakened. This paragraph supersedes the original scratch-Hub assignment
and the Twenty-fourth amendment's statement that ordinary calibration does not
receive external Hub routing.

Capture and resolver procedure v6, identity-input/candidate/frozen schema v5,
runtime-manifest schema v6, capture-provenance schema v2, bootstrap mode v3,
RULER launcher v7, RULER generation-manifest schema v2, publication contract
v1, and Stage-A runner v5 remain unchanged. No record selection, dataset row,
tokenizer byte or span, model revision, RHT transform, Q4/Q6/Q8 code budget,
Fisher H=1 boundary, comparator, metric, statistical gate, or claim boundary
changes.

Only after the new regressions plus focused, full, package, and exact-commit CI
verification may a clean descendant be tagged as the ninth replacement H0. It
must create fresh H0-scoped, no-overwrite repository-source, runtime,
model-metadata, identity-input, and capture-provenance paths and repeat every
prerequisite gate. The promoted replacement identity must pass the complete
contract and differ from identity file SHA-256
`e782e47855979fdb0189b69898b31d13d736aa2c86bb7d1cf69d209da2426ec1`
at exactly the six previously frozen source/runtime/promotion pointers and
nowhere else. All 160 scientific records and all tokenizer, content, split,
model, and Parquet commitments must remain byte-identical. Only that verified
identity may become the sole H1 tree change, after which model staging and one
fresh Fisher smoke require new no-overwrite roots. Until those gates complete,
no protected execution, quality conclusion, deployment, novelty,
state-of-the-art, or breakthrough claim is authorized.

Twenty-eighth replacement-H0 sealed-scratch custody and launcher-finalization
amendment candidate: 2026-08-25. Source commit
`6d0130d2b30f1b6bad24d926570fc91c586d7651` and lightweight tag
`experiment013-h0-6d0130d` became the ninth replacement H0 after local focused,
full, and package gates and exact-commit GitHub CI run `32748769745` passed all
five jobs. The sole identity capture and promotion produced schema-v5 identity
file SHA-256
`ec75fd101f7e5ab9bb8312a6adda0376590f4ebf7e760891db3c3bf51fcc581f`;
committing only that identity produced H1
`09d970e0423f5b37e34a53ab6c59b47fa0d558a6`. The launcher-finalized capture-
provenance receipt file SHA-256 was
`73930958b0b075b50b59f97f4cda56fff224ef3cbcd0e986dee8b75cf91a1dd5`.
The repository-source and runtime manifest file SHA-256 values were
`84992b9ee05369fc287491577ae212b5f360222d4ccc852ad580899ec8bea069`
and `0c15613a551f1857661ddf17d881465ed259a0ad9e48eb7ea6b440f6191dbe1f`.

Authenticated model staging under H1 passed and published exactly three
single-link regular files totalling 1,746,996,407 bytes at
`C:\tmp\recurquant-exp013-model-h1-09d970e0`. These are custody facts, not a
quality result.

The chain's one permitted Fisher H=1 smoke completed the first frozen sequence:
132 tokens and all 16 expected Fisher steps. The inner report recorded CUDA
13.0, PyTorch 2.13.0+cu130, the NVIDIA GeForce RTX 5070 Laptop GPU, and a loaded
model. The official outer invocation nevertheless returned 1. Both the child
and host mandatory postconditions found the launcher-owned scratch directory
non-empty, and the outer launcher preserved that failure. The declared output
contains exactly the old v15 canonical report and marker. Their file SHA-256
values are
`73b044d2895fcd7c3419dfdd86ae2e2b0e792a535992ce5822115d56de46ac2d`
and `c6eb5ebf54bb06f7914cfa8380d2b34496ca8a3dea12564853613c75bf4ec703`;
the report's canonical-evidence SHA-256 is
`ffb0b1bd99c2410255b6acf578ff6787aba189f96619de4d7954459fdcfedb62`.
Because the official invocation failed, those bytes are incident evidence only:
they are not a protocol-valid smoke receipt, do not authorize full calibration,
and are neither positive nor negative evidence about static RHT-Q468.

No current scratch tree survived the outer identity-checked cleanup, so this
amendment does not infer its former inventory from absent bytes. A separate
controlled metadata-only import diagnostic, outside the retired runtime and
model, reproduced the relevant behavior: importing the pinned Torch 2.13.0+cu130
and Transformers 5.14.1 versions created the empty path `torch\inductor` under
the configured `TORCHINDUCTOR_CACHE_DIR`. The v15 contract incorrectly required
literal scratch emptiness even though every cache endpoint was deliberately
confined there.

The incident also exposed a custody flaw independent of Torch. Runner v15
published its report and completion marker before the outer launcher finished
its scratch cleanup and final reauthentication, while the full-run gate
authenticated only those two files. A failed official invocation could
therefore leave mechanically reusable success-looking bytes. The old report
and marker remain immutable evidence of that flaw and may not be copied,
renamed, adopted, or rebound.

The current working tree implements the runner-v16 custody repair, but it is not
frozen or authorizing until committed in a clean H0 and verified by exact-commit
CI. It makes the child publish candidate report/payload files only. After child
return 0, every child and host postcondition, identity-checked owned-root cleanup,
and final source/runtime/cache/artifact reauthentication, the host publishes a
launcher-finalized v2 marker and then publishes canonical
`RUN_LAUNCH_FINALIZATION.json` last as the authoritative no-overwrite commit
sentinel. Smoke and full-calibration receipts bind the exact child inventory,
sizes and hashes, marker, H0, frozen identity, capture receipt, execution
bindings, launch policy, runner revision, and SHA-256 of the normalized absolute
output-directory path. A full receipt and full report also bind the exact prior-
smoke finalization-receipt hash. Every launcher, runner, and Stage-A authorizer
recomputes the actual directory-path digest; a byte-identical directory copy is
therefore inadmissible. Missing, old, noncanonical, moved-between-directories,
altered, or extra prerequisite files fail closed. Controlled stability failure
remains a report-only diagnostic and receives neither marker nor finalization
receipt.

Scratch confinement advances to bootstrap/cache policy v4. Scratch may contain
only exact-case top-level directories `huggingface`, `private-home`, `torch`,
`transformers`, and `xdg-cache`; every descendant must be a real directory or
single-link regular file. Symlinks, junctions, reparse points, hardlinks,
nonregular files, duplicate case-folded roots, unexpected roots, and scratch-
local `huggingface\hub` or `huggingface\datasets` endpoints remain fatal. The
child and host independently enforce the same rule. Pycache remains exactly
empty, and host cleanup still proves disappearance of both owned roots. An
owned root that disappears before identity-checked cleanup is a fatal custody
failure, not successful cleanup; this closes the move-between-postcondition-and-
cleanup interval.

The sealed run output must be disjoint from every authenticated runtime,
repository, model, dataset-cache, RULER-receipt, manifest, capture-receipt,
identity, and prior-smoke evidence path before launch, again after scratch
creation, and again after owned-root cleanup. The prior-smoke directory itself,
not only its three files, is protected. The post-calibration Stage-A authorizer
also rejects an output directory that contains, is contained by, or resolves
through a link into the repository or any full-calibration, smoke, identity,
capture, runtime, source-manifest, or model-manifest evidence path.
On Windows, every sealed capture, calibration-run, and Stage-A-authorization
output destination must use an ordinary local-drive absolute spelling; device/
extended and UNC spellings are rejected. Lexical
containment is not sufficient because `\\?\`, administrative-share UNC, and
mapped-drive spellings can name the same object with different anchors. Cache,
capture, run, and authorization boundaries therefore also compare directory-
component `(device, inode)` identities. The Stage-A authorization publisher
snapshots its existing non-link parent, revalidates that exact component chain
immediately before the staging-directory rename, and authenticates the owned
staging identity before any recursive failure cleanup. Parent or staging drift
fails without following the redirected path.

The runner advances v15 to v16, runtime-manifest schema v6 to v7, run-report
schema v3 to v4, completion-marker contract v1 to launcher-finalized v2, and
Stage-A calibration-authorization schema/revision v1 to v2. The new Stage-A
authorization depends on both smoke and full launch-finalization receipts;
the existing Stage-A binding transitively binds them and remains schema/revision
v4. Calibration identity schema v5, Stage-A identity schema v6, capture and
resolver procedure v6, capture-provenance schemas, model manifest, RULER
launcher and generation-manifest schemas, all 160 records, tokenizer/content/
split/Parquet commitments, model revision, Fisher boundary, byte budgets,
methods, metrics, statistical gates, and claim boundary do not change.

Existing ninth-chain artifacts are preserved. Every ninth-chain namespace,
whether populated or still absent, is retired and non-authorizing. There is no
retry, copying, adoption, renaming, or rebinding under that chain. Only after
the v16 regressions plus focused, full,
package, and exact-commit CI gates pass may a clean descendant become the tenth
replacement H0 with an entirely fresh H0-scoped artifact namespace. Its new
identity must keep all scientific/content fields byte-identical and differ from
the retired identity at exactly the six frozen source/runtime/promotion
pointers. Only its identity-only direct-child H1 may stage a fresh model and
attempt exactly one new Fisher smoke. Full calibration remains unauthorized
until that new official smoke returns 0 and its v2 marker plus launcher-last
finalization receipt authenticate successfully.

Twenty-ninth replacement-H0 Stage-A capture import-isolation amendment
candidate: 2026-08-25. Source commit
`a5188a0b3e7bc3ab9ab2a27a639cac26d93030bd` and lightweight tag
`experiment013-h0-a5188a0` became the tenth replacement H0. Its sole promoted
schema-v5 calibration identity had file SHA-256
`17b2aa18840a040883e74cb1f2ac17ad152aee3865e4d1537bbec59670099697`;
committing only those promoted bytes produced H1
`874e586bda98602cc712a543a071a0047df38659`. The repository-source,
runtime, model-manifest, and launcher-finalized calibration-capture receipt
file SHA-256 values were respectively
`d3d5640ae4a779357bb34bb7f1edc002a6803fa088c9aa7c0a7ffe36d59bb210`,
`95092677bcc4245c2ffe1d6dbbe83f39f411cb10d012efa61f1e3bc9378b4f36`,
`586d9c7e520f3bbd99ecef30663bf07d283eb14622475c58891becd8e033b05c`,
and
`e67e00c44e9956a6f1444d79c7bc545def82e4849aa748111e335d1aa1cee8bc`.

Authenticated model staging, the one permitted Fisher H=1 smoke, and the full
calibration all completed under that chain. The finalized smoke report and
launcher-last receipt had file SHA-256 values
`26a1b7f297133d3b8b6f854b7f3c0cce97d8332fb881a8054a9a3072cb34789b`
and
`2e6ad077612d81f1b8a6bd5c107cad953689cc3f5d0130555c76cbacb545cc4f`.
The finalized full-calibration report and launcher-last receipt had file
SHA-256 values
`196d382325141f46ec6415b601a5bc240a0d3243e062a55a119a0b6aeaa67b74`
and
`00af660cdc460b6246cbcb0b09661dc7487744304dfba8dbc25b1e87401d0d26`.
The post-calibration metadata-only authorizer then accepted the exact closed
12-file full-calibration directory, exact closed three-file smoke directory,
and their explicit prerequisite hashes. It atomically published exactly these
three files under the no-overwrite directory
`C:\tmp\recurquant-exp013-stage-a-calibration-authorization-h1-874e586b`:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `STAGE_A_CALIBRATION_AUTHORIZED` | 59 | `bda179b4e32cecc80b72132abd1e5613befa9b57fd9b539b7245b820d29ea227` |
| `stage-a-calibration-authorization.json` | 19,596,102 | `83ee580d7a9a63bb58a395174f700ab6cfe03df60c42c0ce879c60ae9d161e02` |
| `stage-a-calibration-binding.json` | 26,129,565 | `bc9b5c7db604826e2d7564bc8934871d5c19532c9f274304f67d086cb5e0c962` |

The schema-v2 authorization reported `authorized_for_stage_a`, bound H0
`a5188a0b3e7bc3ab9ab2a27a639cac26d93030bd`, embedded the exact 12
authorization dependencies, and authenticated all nine calibration outputs.
The schema-v4 binding embedded that authorization at its exact file hash and
exposed the required nine-field Stage-A calibration binding. These are
calibration and custody facts only; successful calibration and authorization do
not establish evaluation quality, non-inferiority, deployment readiness, or a
static-Q468 advantage.

The first official sealed `capture-stage-a-identity` attempt failed before a
live capture source, tokenizer, or protected dataset provider was constructed.
After authenticating the binding file hash, source/runtime/model/Parquet
manifests, H0, and the shallow RULER directory precondition, the sealed runner
installed the existing model/CUDA import isolation and exact-loaded the H0
resolver. Strict
`deserialize_stage_a_calibration_binding_artifact` verification decoded the
embedded authorization and rederived its complete dependency chain. That
rederivation unconditionally imported `recurquant.static_q468` and
`recurquant.static_q468_calibration`; both modules import `torch` at module
scope. The isolation guard correctly stopped that request with
`capture attempted forbidden model/CUDA import: torch`. The ordinary metadata
authorizer had succeeded because it performed the same recursive derivation
outside the sealed capture isolation. The mismatch is therefore an
authenticated source-graph defect at the Stage-A capture boundary, not an
invalid authorization artifact and not an infrastructure interruption.

The failed child published neither a Stage-A identity-input file nor a
launcher-finalized schema-v1 Stage-A capture-provenance receipt. It produced no
candidate, frozen Stage-A identity, identity-authorization commit, Stage-A
input bundle, one-run reservation, result, or completion marker. It did not
construct a model adapter, open or load model weights, open the four protected
Stage-A RULER receipt bodies, fetch a PG19 validation row, fetch a HumanEval+
record, tokenize protected content, or calculate a quality value. Thus the
attempt is evidence that the isolation boundary failed closed; it is neither
positive nor negative scientific evidence about static RHT-Q468.

This failure is deterministic containment and source-design evidence. The
same-command infrastructure-retry exception does not apply. H0
`a5188a0b3e7bc3ab9ab2a27a639cac26d93030bd`, tag
`experiment013-h0-a5188a0`, H1
`874e586bda98602cc712a543a071a0047df38659`, and every artifact and
namespace bound to them are retired and non-authorizing. This includes the
source/runtime/model manifests, calibration identity and capture receipt,
model root, cache, Fisher smoke, full-calibration directory, Stage-A
authorization directory and binding, failed command, and every declared
Stage-A capture, candidate, promotion, and frozen-identity path whether
populated or absent. They must be preserved as historical evidence and may not
be retried, copied, renamed, adopted, substituted, or rebound into a replacement
chain. A successful old authorization does not survive the retirement of its
H0 and execution bindings.

The repair must preserve strict recursive verification; accepting only the
outer binding envelope, trusting a previously reported status, weakening the
Torch blocker, preloading Torch, or exempting these imports is forbidden. A
new authenticated, Torch-free metadata module at
`src/recurquant/static_q468_artifact_contract.py`, imported only as
`recurquant.static_q468_artifact_contract`, must own the complete artifact
surface needed by Stage-A binding verification: frozen model/tokenizer IDs and
geometry, strict Q468 and Q48 policy deserialization, strict calibration-score
and split-half-stability deserialization, deterministic `P=14739` Q48
reconstruction, and canonical policy serialization. It may depend only on the
standard library and the already authenticated NumPy capture surface. It must
not import Torch or any module that transitively imports Torch, including the
current runtime-oriented `static_q468`, `static_q468_calibration`,
`multibit_policy`, `metrics`, quantization, cache, or adapter modules. The
runtime-oriented `static_q468` and `static_q468_calibration` modules remain the
independent Torch producer/reference implementation. The new module is a
deliberately independent NumPy N-version verifier for the same frozen artifact
contract, not a runtime adapter. Duplication is permitted only under the
source-bound differential-conformance gates below; neither implementation may
silently change canonical bytes, allocation or tie-breaking semantics,
accepted field/value domains, or decoded values.

The new metadata module and the contract regressions in
`tests/test_multibit_policy.py` and `tests/test_static_q468_calibration.py` must
be explicit frozen-source-manifest members. Before H0, those regressions must
compare both implementations on Q468 and Q48 canonical serialization,
reciprocal deserialization, code maps, counts, hashes, every feasible budget
for exhaustive small instances including ties, non-convex marginals and
endpoints, and the frozen `P=14739` reconstruction. Every valid and adversarial
policy, calibration-score, comparator, and split-half fixture must be processed
by both implementations, requiring identical accept/reject decisions and,
when accepted, equal canonical hashes, scalar values, and array bytes. Any
divergence or unilateral acceptance fails closed and forbids H0. A contract
change must update both implementations and the differential corpus in the
same clean source commit.

Every security-critical consumer must hash and exact-load its held bytes under
an authenticated module name before resolver use. In particular,
the metadata authorizer, sealed calibration/Stage-A capture runner, and Stage-A
screen must not fall back to an ambient package import. The capture bootstrap
must treat the module as a required pre-source-verification path, reject a
preloaded instance, install it under the authenticated `recurquant` namespace
before binding deserialization, and remove it during exact-module cleanup. The
resolver's authorization derivation must import only this pure metadata layer;
the existing import-isolation guard and postconditions remain mandatory. The
owned metadata authorizer must activate that same guard before exact-loading
the contract and resolver, retain it through recursive authorization and final
binding rederivation, and restore it before publishing any output.

A real-path regression is required before the repair can contribute to a new
H0. In a fresh isolated `-I -B` subprocess using the production runner, exact
source loader, production resolver, and a canonically generated complete
core-binding dependency chain, it must activate the real capture isolation and
call the real core-binding builder and deserializer rather than the prior
`SimpleNamespace` stub. It must finish recursive rederivation with identical
dependency hashes and byte-identical Q468 policies, with `torch`,
`recurquant.static_q468`, and `recurquant.static_q468_calibration` absent from
`sys.modules` and no forbidden import attempt recorded. Paired cases must
prove that a changed score, policy, dependency hash, or canonical byte fails
recursively. Existing adversarial import-topology, hidden-availability,
restoration, and forbidden-import regressions may not be weakened.

That always-run core regression does not replace end-to-end authorization.
Before H0 is accepted, the fresh requalification must also construct the
complete smoke, full-report, marker, launch-finalization, authorization, Q48,
and final-binding chain and traverse the actual sealed binding deserializer to
the pre-live-source boundary under the same isolation. A changed embedded
authorization, marker, report, Q48 policy, execution binding, or canonical
byte must fail before protected input or model access.

The repair advances the calibration runner from v16 to v17, the capture and
resolver procedures together from v6 to v7, and the sealed Stage-A runner in
`screen_static_q468_stage_a.py` from v5 to v6. No artifact field-set change is
intended: calibration identity remains schema v5;
Stage-A candidate and frozen identity remain schema v6; calibration and Stage-A
capture-provenance schemas remain v2 and v1; runtime-manifest schema remains
v7; run-report schema remains v4; Stage-A
calibration authorization remains schema/revision v2; and the only eligible
Stage-A binding remains schema/revision v4. If implementation requires any
field, canonical-byte, or semantic-contract change, its corresponding schema or
revision must advance in a further pre-H0 amendment rather than being silently
treated as this repair. No record selection, tokenizer byte or span, model
revision, Parquet or RULER commitment, Fisher boundary, quantization policy,
byte budget, metric, statistical gate, or claim boundary changes.

Only after the amendment, pure-metadata implementation, real-path regression,
all focused suites, the full clean-tree suite, package gates, and exact-commit
GitHub CI pass all five jobs may a clean descendant be tagged as the eleventh
replacement H0. That chain must use entirely fresh H0-scoped, no-overwrite
repository-source, runtime, model-metadata, calibration identity-input,
capture-provenance, candidate, promoted-identity, model, cache, smoke,
full-calibration, Stage-A authorization, and Stage-A capture paths. It must
repeat every source, runtime, immutable-model-metadata, RULER-custody, capture,
promotion, frozen-identity-contract, model-staging, smoke, full-calibration, and
post-calibration authorization gate. The replacement promoted calibration
identity must keep every scientific/content field byte-identical to identity
file SHA-256
`17b2aa18840a040883e74cb1f2ac17ad152aee3865e4d1537bbec59670099697`
and differ only at these seven frozen source/runtime/procedure/promotion
pointers:

```text
/canonical_evidence_sha256
/evidence/execution_bindings/calibration_runtime_manifest_file_sha256
/evidence/execution_bindings/repository_source_manifest_file_sha256
/evidence/promotion/candidate_canonical_evidence_sha256
/evidence/promotion/candidate_file_sha256
/evidence/resolver_version
/evidence/source_manifest_sha256
```

The added procedure pointer must change exactly from resolver version 6 to 7;
the other six pointers retain their established replacement semantics. Only
that identity may be the sole tree change in its direct-child H1.

The smoke, full calibration, authorization, and binding must be regenerated;
their old H0, runner, source/runtime, output-path, and prerequisite commitments
make reuse impossible even if the scientific inputs are unchanged. Existing
public immutable dataset and RULER materializations may be reused only through
their already frozen byte-level replay and provenance rules, never by adopting
an old H0-scoped artifact. Only after the new authorization and binding pass
may exactly one fresh sealed Stage-A identity capture be attempted. Until its
identity input and launcher-finalized receipt authenticate, no Stage-A
candidate, promotion, identity-authorization commit, model evaluation, quality
conclusion, deployment, adoption, novelty, state-of-the-art, or breakthrough
claim is authorized.

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
question. That row is rejected and is not evidence. Experiment 013 RULER
launcher v7 invokes the pinned task generator directly with a no-shell argument vector.
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
runner at point of use. Calibration runtime-manifest v7 authenticates the
Python and installed package-code inventory used by calibration plus the
private-scratch child-cwd and cache-confinement launch policy defined by the
Seventeenth amendment. The model manifest is derived from immutable Hub
repository/LFS metadata without downloading or opening weight payloads. The
Parquet manifest authenticates the source and conversion commits plus the
selected Parquet Git/LFS objects. Only after promotion may the runner hash
local model files and compare them with the frozen model manifest. A missing,
malformed, or byte-different dependency stops before adapter data access or
model loading.

Identity capture is an official calibration-runtime action, not a host-side
manifest inspection. The sole official entrypoint is the outer sealed launcher
with runner command `capture-calibration-identity` and the exact option profile
in the Seventeenth amendment; the ordinary unsealed runner rejects it. It must
use the authenticated staged interpreter bound by runtime-manifest v7, with the
exact authenticated base `sys.path`, package roots, import paths, and
54-distribution inventory. Those
facts are checked in the capture process before tokenizer or dataset content
access. A repository host virtual environment may coordinate preparation or
tests, but may not execute the official capture or satisfy this provenance
requirement merely by authenticating staged files. Success produces the no-
overwrite identity input in the child and, only after outer-launcher
finalization, its separate canonical schema-v2 capture-provenance receipt. The
receipt is custody evidence, not an identity field.

The source manifest binds implementation commit H0. Committing the promoted
identity creates H1. H1 is authorized only when H0 is its Git ancestor, the
authenticated source verifier proves every frozen source path has identical H0
tree, H1/index, and worktree bytes, and the worktree is otherwise clean. Reports
and policy artifacts continue to record H0 as implementation provenance; H1 is
the identity authorization commit and may not be relabelled as source commit.

Before committing H1, the exact promoted identity bytes in their ignored,
no-overwrite precommit location must pass `verify-frozen-identity-contract`.
That read-only command authenticates H0 and its source manifest, loads the exact
H0 resolver, and consumes the complete record inventory through the active
frozen calibration runner's identity view. It requires the exact receipt and runtime inputs
`--capture-provenance-receipt`,
`--expected-capture-provenance-receipt-sha256`, `--runtime-manifest`, and
`--expected-runtime-manifest-sha256`. It accepts no H1, model manifest, Hub,
cache, or output argument. Its non-persisted canonical JSON stdout document
uses artifact kind
`recurquant_experiment013_frozen_identity_contract_verification`, schema
version two, and binds the H0/source contract, portable Git identity, all four
execution bindings, complete identity/canonical/assignment hashes, public
model and tokenizer contracts, record count, and authenticated
`capture_provenance_receipt_file_sha256`. The receipt must bind the exact
identity input from which the promoted bytes were resolved and must pass every
runtime, source, capture-source, critical-origin, RECORD-ownership, and excluded-
module check above. The bytes that passed are then copied without modification
as the sole H1 tree change; regeneration or hand editing after that preflight
is forbidden.

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
identity and execution bindings; the exact public model-metadata manifest; and
the same capture-provenance receipt and runtime manifest accepted before H1.
It requires `--capture-provenance-receipt`,
`--expected-capture-provenance-receipt-sha256`, `--runtime-manifest`, and
`--expected-runtime-manifest-sha256`. The command accepts no cache or output
root, imports no Hub downloader, downloads no file, and creates no directory or
artifact. Its canonical JSON stdout document, which the command does not
persist, uses artifact kind
`recurquant_experiment013_model_staging_authorization`, schema version two, and
binds status, runner revision, frozen-identity hash, H1, H0, repository-source-
manifest hash, model-manifest hash, public model ID/revision, Hub-tree-manifest
hash, file count, total bytes, and
`capture_provenance_receipt_file_sha256`. Only successful path and
authorization documents permit `stage-model` to be attempted with those same
roots and receipt. A
semantic or authentication mismatch retires that H1; it may not be hand-edited
or weakened. An argument-parse or initial pure path-precondition failure does
not consume H1 because authentication did not begin. A documented
infrastructure interruption after authentication begins but before model-
payload access permits only an exact same-command retry under that H1.

Model payload staging begins only after the frozen identity is tracked with
identical H1, index, and worktree bytes. The identity-bound stager downloads
only the exact sorted root files in the frozen model manifest at the exact
40-hex public Hub revision, using an external cache and no token. It requires
the same receipt/runtime path and expected-SHA inputs as the standalone
authorization, invokes that shared receipt-aware authorization before download,
and repeats it immediately before publication. Before Git or H1 authentication,
immediately after authorization, and immediately before publication, the
stager repeats the pure path validator and requires identical normalized roots
and existing-component identities. The cache and output
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
manifest byte, runner v7 performs the shallow pure directory precondition
defined by the Sixteenth amendment, retained by the Seventeenth amendment, and
repeats it inside `_official_main`.
Passing that precondition does not authenticate any receipt body; phase-scoped
point-of-use verification remains mandatory.

Stage-A resolution consumes one strictly decoded
`experiment-013-stage-a-calibration-binding-v4` artifact. That artifact embeds
the post-calibration authorization receipt; the receipt embeds the schema-v3
core binding and its calibration dependencies. The resolved Stage-A identity
binds the authorization artifact plus these eight dependency files directly,
not merely semantic IDs copied from a caller:

```text
calibration_authorization_file_sha256
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
keeps the calibration dependency inventory at eight files; the two
profiles retain separate score hashes, position manifests, and policy
bindings. A policy file without its matching embedded comparator scores is not
verifiable and fails closed.

The static Q4/Q8 comparator is deterministically reconstructed inside the
authenticated Stage-A evaluator from the bound candidate score artifact at
the frozen `P=14739` promotion count. Its separately published convenience
copy is authenticated as a full-calibration output by the authorization
receipt, but it is not supplied to Stage-A method reconstruction and may not
replace reconstruction from the bound scores.

Changing any byte in the finalized calibration, smoke, or capture chain
requires a new authorization, schema-v4 binding, and Stage-A identity
candidate.

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

Both children use the authenticated staged interpreter and package roots bound
by runtime-manifest v7, run with a launcher-owned scratch directory as cwd,
confine private home, Hub,
assets, token, Xet, Transformers, Torch, Triton, and compiler caches to that
scratch directory, and place only dataset caches below the validated explicit
cache root. Their scratch and bytecode roots use the identity-bound cleanup and
failure-aggregation contract in the Seventeenth amendment. The offline flags
and socket guard strengthen the second child; they do not replace the common
containment contract.

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

The source and calibration runtime-manifest v7 authenticate the exact canonical Git
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
The Experiment 013 hypothesis is narrower: prove that
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
stack. Experiment 013 currently supports no method-result claim. If a future
protocol-valid run passes, claims remain limited to its pinned quality, storage,
and implementation measurements unless the additional evidence above is
completed.

A protocol-valid negative result will be reported. Execution-contract failures
remain incident evidence and do not count as method results. Any change after a
gate is observed creates a new experiment number with new protected data.
