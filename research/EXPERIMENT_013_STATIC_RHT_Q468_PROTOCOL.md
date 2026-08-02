# Experiment 013: static RHT-Q468 packed-native adoption protocol

> **Status: amended and frozen before Experiment 013 identity resolution, policy fitting,
> protected-set access, model-weight loading, or quality measurement.**
>
> Upstream revisions are frozen below. Canonical row identities, token spans,
> tokenizer-file hashes, and content hashes remain unresolved placeholders
> until a separate identity candidate is resolved and explicitly promoted. An
> identity candidate is not authorization to load model weights.

Protocol frozen: 2026-08-02

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

## Question

Can a calibration-frozen, static Q4/Q6/Q8 recurrent-state layout retain the
quality of an exact dynamic mixed-bit oracle while using one immutable code map
that a packed GPU kernel can execute efficiently?

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

The conditional scale check is:

```text
Qwen/Qwen3.5-2B-Base
revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
Transformers 5.14.1
```

The 2B check is authorized only if a cold-start language-model load measures no
more than `7.5 GiB` peak device memory on the RTX 5070 8GB. Failure or inability
to satisfy that gate is recorded as a resource stop; it cannot be replaced by
an unreported offload configuration.

Both checkpoints use batch one, eager evaluation, no sampling, BF16 model
weights, and FP32 reference recurrent state. Model architecture, recurrent
layer indices, state geometry, tokenizer class, tokenizer files, and every
runtime package are identity-bound before weights are opened. A mismatch stops
the run.

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

The `K27030` layout is a prespecified diagnostic budget used only to compare
static and dynamic selection without conflating that comparison with the
primary exact-byte contrast. Its physical allocation is reported exactly and
is never rounded up to the primary budget.

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
the code-map fit. The primary map uses `K29334`. The diagnostic static and
dynamic layouts both use `K27030`. Once an identity and code map are committed,
no seed, score, quota, tie rule, token span, group size, or bit budget may be
changed under Experiment 013.

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

## Frozen evaluation identities

Identity resolution is staged. A resolver may create only a quarantined
candidate. A separate explicit promotion, checked by candidate SHA-256,
creates the identity that must be committed before model weights are loaded.
Stage-B and Stage-C content is protected and requires separate authorization;
ordinary resolver tests and dry runs must not read it.

Identity schema v4 also binds four exact pre-model evidence files under
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

Stage-A resolution additionally consumes one strictly decoded
`experiment-013-stage-a-calibration-binding-v2` artifact. The resolved Stage-A
identity binds these five dependency files directly, not merely semantic IDs
copied from a caller:

```text
calibration_identity_file_sha256
calibration_score_artifact_file_sha256
split_half_stability_artifact_file_sha256
static_k27030_policy_file_sha256
static_k29334_policy_file_sha256
```

Changing any byte in any dependency requires a new binding artifact and a new
Stage-A identity candidate.

### Stage A: multi-workload falsification

Stage A contains exactly 12 examples:

- the first four SHA-ranked eligible PG19 validation books, each using 4,096
  prefill tokens followed by 128 continuation tokens, of which 127 predictions
  are exposed to the committed quantized cache;
- four RULER category representatives at configured length 4,096 and seed
  2,339: `niah_multiquery`, `vt`, `fwe`, and `qa_1`, using each
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

The exact dynamic Q468 allocator is the Stage-A quality oracle. Static K27030
must have an excess-NLL upper confidence bound no more than `0.01` nats/token
above dynamic K27030. No Stage-A workload may have a static-candidate
disadvantage above `0.015` nats/token. Static K29334 must beat exact-byte
`rht_q48_static_p14739` on every Stage-A workload. If the multi-workload advantage
does not reproduce, stop; do not proceed by reframing the oracle as optional.

### Stage B: development

Stage B remains closed until Stage A and every identity gate pass. It contains:

- the remaining 28 eligible PG19 validation books after Stage A;
- the remaining 48 configurations in the complete development grid of all 13
  exact RULER configs at configured length 4,096 and seeds 2,339 through
  2,342; and
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
- the next 32 HumanEval+ canonical IDs under the separate Stage-C confirmation
  hash domain.

The generated RULER IDs, auxiliary-source hashes, formatter hashes, actual
lengths, and token spans remain unresolved until a separate protected identity
amendment is frozen. Stage C may not be partially previewed.

## Methods and measurements

Every accepted quality run includes FP32 recurrent state, uniform RHT Q4 and
Q8 anchors, `rht_q48_static_p14739`, static K27030, exact dynamic K27030, and
`rht_q468_static_k29334`. A closest eligible published comparator is added only
through a pre-result identity amendment with its exact implementation and byte
accounting; an incompatible or unavailable comparator is documented rather
than imitated under its name.

Primary quality is task-macro aligned excess next-token NLL over only the
identity-bound cache-exposed transitions relative to the matched FP32
trajectory. Report task-macro and token-micro excess NLL, mean and
tail KL, top-1 agreement, local codec SSE, trajectory error, result by workload
family, resident bytes, transient bytes, peak HBM, and latency. Statistical
intervals are paired task bootstraps with 10,000 resamples and seed 2,339.

For the full evaluation, the candidate must improve on the strongest eligible
comparator by both at least `10%` and at least `0.002` nats/token, with the
paired 95% lower confidence bound above zero. The point improvement must be
positive in each of PG19, RULER, and HumanEval+. Candidate top-1 agreement may
trail by at most `0.005`. A non-positive comparator excess NLL makes the
relative gate fail closed; it is not redefined.

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

## Advancement and claim boundary

Every integrity, stability, quality, and deployment gate is conjunctive for an
adoption-ready claim. A quality pass without the deployment gate supports only
a quality/storage result. A kernel pass without the protected quality stages
supports only an implementation result.

The prior-art boundary is narrow. [RateQuant](https://arxiv.org/abs/2605.06675v2)
already fits calibration-based mixed-precision rate-distortion policies for KV
caches. [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) quantizes
Mamba state caches, while [Quamba2](https://arxiv.org/abs/2503.22879v4) provides
quantized SSM deployment and kernels. [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791v1)
motivates the architecture family. Its official repository currently provides
training code but no tagged release or pretrained checkpoint for the reported
1.3B run. Therefore only a confirmed exact-byte static Q4/Q6/Q8 packed-native
Gated DeltaNet path plus end-to-end adoption benefit could be differentiated;
this protocol makes no novelty claim.

Even a complete pass would establish only that the frozen static packed layout
was useful on the pinned checkpoints, workloads, budgets, and hardware. It
would not establish that RHT, mixed precision, loss sensitivity, Q4/Q6/Q8,
static allocation, or packed kernels are new. It would not make RecurQuant a
new base model, prove generated-code correctness, eliminate contamination, or
justify "breakthrough," "state of the art," "lossless," or universal language.

Failure is a publishable result. Any change after a gate is observed creates a
new experiment number with new protected data.
