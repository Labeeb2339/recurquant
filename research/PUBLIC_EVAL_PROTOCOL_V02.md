# Public evaluation protocol v0.2

Status: frozen protocol; no public-evaluation results have been run or reported

Date frozen: 2026-07-22

## Purpose

This experiment tests one narrow question: on a pinned Qwen3.5 checkpoint and
public MBPP reference programs, does the already frozen RecurQuant policy
preserve teacher-forced code-token predictions better than prespecified
baselines while physically storing the persistent Gated DeltaNet state in the
declared number of bytes?

This is not a code-generation benchmark. It does not sample programs, execute
untrusted code, or measure `pass@k`. It measures the effect of recurrent-state
quantization while every method receives the same gold prefix.

## Why v0.2 is a correction, not a victory lap

The v0.1 result used three short synthetic traces. It demonstrated that the
adapter worked and that a frozen layer-0 mixed-precision policy passed its small
diagnostic confirmation. It did not establish public-task generalization,
novelty, total inference-memory savings, or speed.

The prior-art boundary is also stricter than the initial repository wording:
[Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) already reports
4-bit Mamba-2 state-cache quantization, and
[Quamba2](https://arxiv.org/abs/2503.22879) and
[Nemotron 3 Super](https://docs.nvidia.com/nemotron/0.1.0/nemotron/super3/quantization.html)
provide additional recurrent-state quantization and rounding precedents.
RecurQuant must not claim to be the first recurrent-state or first 4-bit
recurrent-cache method. Stochastic rounding is therefore a required baseline,
not an optional favorable comparison.

The packed cache in v0.2 can establish resident tensor bytes. Because it still
dequantizes a state while a layer executes and has no fused quantized recurrent
kernel, resident-state bytes must be reported separately from transient memory,
whole-model peak memory, and latency.

## Frozen assets

### Model

- Model and tokenizer: `Qwen/Qwen3.5-0.8B-Base`.
- Revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Revision permalink:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68>.
- Expected architecture: 24 language layers, including exactly 18 Gated
  DeltaNet layers at model-layer indices
  `0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22`.
- Expected recurrent-state shape at batch one: `[1, 16, 128, 128]` per Gated
  DeltaNet layer.
- Model weights use BF16 on CUDA. The unquantized persistent-state reference is
  the model's FP32 recurrent state.
- `trust_remote_code=False`, eager attention, evaluation mode, batch size one,
  and no sampling.

Any model revision, layer-count, state-shape, or state-dtype mismatch invalidates
the run. It must not be silently adapted after seeing evaluation results.

The reference software environment is Python 3.11.15, PyTorch 2.11.0+cu128,
Transformers 5.14.1, and Datasets 4.8.5. Every artifact records exact package,
CUDA, driver, device, repository-commit, and command information. A result from
another environment is a replication, not a byte-for-byte repeat.

### Dataset

- Dataset: `google-research-datasets/mbpp`.
- Configuration: `full`.
- Revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Revision permalink:
  <https://huggingface.co/datasets/google-research-datasets/mbpp/tree/4bb6404fdc6cacfda99d4ac4205087b89d32030c>.
- Original split specification:
  <https://github.com/google-research/google-research/tree/master/mbpp>.

The official ID ranges are kept intact:

| Role | Source split | Task IDs | Count |
|---|---|---:|---:|
| Excluded few-shot examples | `prompt` | 1-10 | 10 |
| Untouched confirmation | `test` | 11-510 | 500 |
| Development | `validation` | 511-600 | 90 |
| Calibration pool | `train` | 601-974 | 374 |

Calibration uses exactly 128 tasks. For each training ID, compute lowercase
hexadecimal SHA256 over the UTF-8 bytes of `rq-v0.2|{task_id}` with no newline,
sort by `(digest, task_id)`, and take the first 128. In ascending task-ID order,
the frozen set is:

```text
602, 605, 606, 607, 614, 615, 616, 622, 627, 629, 630, 635, 636, 640, 641,
644, 646, 647, 648, 651, 653, 656, 657, 662, 666, 669, 671, 672, 674, 677,
679, 686, 687, 695, 698, 702, 704, 705, 707, 708, 709, 710, 712, 713, 720,
722, 725, 727, 728, 732, 741, 744, 751, 755, 756, 763, 764, 768, 771, 772,
783, 785, 789, 792, 793, 794, 795, 797, 800, 803, 804, 809, 819, 820, 821,
822, 823, 824, 827, 835, 839, 840, 846, 847, 848, 849, 851, 853, 854, 857,
858, 860, 862, 867, 868, 869, 870, 872, 874, 877, 878, 882, 884, 885, 886,
894, 895, 902, 903, 907, 908, 911, 918, 919, 920, 924, 929, 930, 936, 937,
944, 945, 946, 950, 955, 960, 962, 973
```

The remaining 246 training tasks are unused. The prompt split is not used for
few-shot examples. Development uses all 90 validation tasks, and confirmation
uses all 500 test tasks. There is no random task subsampling.

Before any development run, the evaluator must write and verify a manifest
containing each task ID, normalized input-field digest, prompt-token digest,
target-token digest, and token counts. No task may be removed because its result
is inconvenient. If any example exceeds the pinned model context after frozen
serialization, the run stops and the protocol is revised before inspecting
method comparisons.

## Frozen input construction

Normalize `\r\n` and bare `\r` to `\n` in `text`, each `test_list` entry, and
`code`; make no other content edits. Let `tests` be the three normalized test
strings joined by one newline. The zero-shot prompt is exactly:

```text
You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:

{tests}
[BEGIN]
```

Encode the prompt with `add_special_tokens=True`. Encode the normalized
reference `code` separately with `add_special_tokens=False`, then concatenate
the token-ID lists. Separate encoding intentionally makes the target boundary
unambiguous. Do not append `[DONE]` or EOS to the scored target.

Prefill the complete prompt. For code tokens `y_1, ..., y_T`, score `y_1` from
the final prefill logits, then feed each gold token to score the next token.
Candidate-generated tokens are never fed back. Quantize and pack the persistent
state once after prefill and after every single-token teacher-forced update.
Weights, convolution state, ordinary attention KV caches, token IDs, and model
arithmetic are otherwise shared with the reference.

## Frozen quantizer and byte accounting

All quantized methods use symmetric signed per-group absmax scaling, group size
128 over the flattened final two state dimensions, FP16 scales, no zero point,
and either the stated nearest or stochastic rounding. For `b` bits,
`qmax = 2^(b-1) - 1`; the absmax scale is stored in FP16 and clamped to the FP16
range with minimum positive value `2^-24`. Nearest rounding is PyTorch's
round-to-nearest-even behavior. INT4 stores two signed nibbles per byte; INT8
stores signed bytes. Padding, scales, and any required metadata count. Static
layer formats require no per-token precision tag.

For one `[1, 16, 128, 128]` state there are 262,144 elements and 2,048 groups:

| Format | Payload/layer | Scales/layer | Total/layer |
|---|---:|---:|---:|
| FP32 reference | 1,048,576 B | 0 B | 1,048,576 B |
| INT4 | 131,072 B | 4,096 B | 135,168 B |
| INT8 | 262,144 B | 4,096 B | 266,240 B |

The frozen whole-cache budgets are therefore:

| Policy layout | Resident recurrent-state bytes | Role |
|---|---:|---|
| 18 FP32 layers | 18,874,368 | Reference storage anchor |
| 18 INT4 layers | 2,433,024 | Lower-byte quality anchor |
| One INT8 layer plus 17 INT4 layers | 2,564,096 | Exact-byte comparison budget |
| 18 INT8 layers | 4,792,320 | Higher-byte quality anchor |

The primary mixed plan is a modeled 7.36x reduction in persistent-state bytes
relative to its FP32-state reference. The public artifact may call this a real
resident recurrent-state reduction only if the sum of the live packed payload
and scale tensor storages is exactly 2,564,096 bytes. It must also report the
largest transient materialized state and CUDA allocator peaks separately.

## Frozen methods

All exact-byte mixed methods store exactly one Gated DeltaNet layer at INT8 and
the other 17 at INT4, for 2,564,096 resident bytes.

1. `fp32_state`: unquantized persistent-state reference.
2. `uniform_int4_nearest`: lower-byte anchor.
3. `uniform_int8_nearest`: higher-byte anchor.
4. `read_risk_l0_nearest`: primary frozen candidate; model layer 0 is INT8.
5. `mse_selected_nearest`: baseline selected once on the 128 calibration tasks.
   At every post-prefill and teacher-forced state write from the FP32-state run,
   compute `||Q4(S)-S||_F^2 / max(||S||_F^2, 1e-24)` for each layer, average
   equally over writes and then tasks, and select the maximum; exact ties choose
   the lower layer ID.
6. `random_l18_nearest`, `random_l4_nearest`, and `random_l13_nearest`:
   prespecified random-layer controls. For seeds 1101, 2202, and 3303, compute
   `int(SHA256("rq-v0.2-random|{seed}"), 16) mod 18` over the UTF-8 string with
   no newline and use that position in the ordered Gated DeltaNet layer list
   above. The frozen results are model layers 18, 4, and 13 respectively.
7. `read_risk_l0_stochastic`: the same bit layout as the primary candidate,
   evaluated at stochastic-rounding base seeds 2339, 2340, and 2341. The seed
   schedule adds `layer_id * 1,000,000 + per_layer_cache_update_index`.

The primary layer remains 0 regardless of what MBPP calibration shows. Public
calibration may measure its read-risk rank as a diagnostic, but cannot replace
it. The MSE selector is the only baseline whose layer is chosen from the public
calibration partition. No confirmation-set layer sweep or oracle selection is
permitted.

QDQ versions of the same policies are implementation controls, not memory
baselines: they retain dequantized FP32 tensors. Packed and QDQ logits and
metrics must agree within the frozen numerical tolerance before a packed result
is accepted.

## Metrics

For task `i`, target code token `t`, method `m`, gold token `y_it`, and logits
`z_mit`, define

```text
nll_mit = -log_softmax(z_mit)[y_it]
excess_nll_mit = nll_mit - nll_fp32,it
```

First average over code tokens within each task, then average equally over
tasks. This macro task mean is primary; a long reference program must not count
as many independent tasks. The primary fidelity quantity is

```text
DeltaNLL_m = mean_i(mean_t(excess_nll_mit)).
```

Smaller is better and zero means no mean NLL change from the FP32-state run.
The prespecified 15% effect is evaluated against uniform INT4 when its
`DeltaNLL` is positive:

```text
relative_reduction =
    (DeltaNLL_uniform_int4 - DeltaNLL_read_risk_l0)
    / DeltaNLL_uniform_int4.
```

If uniform INT4 has non-positive `DeltaNLL`, this ratio is undefined and the
15% continuation gate fails rather than being redefined after inspection.

The equal-byte primary contrast averages the three random controls within each
task, then computes

```text
equal_byte_gain_i =
    mean_random_DeltaNLL_i - read_risk_l0_DeltaNLL_i.
```

Required secondary metrics are:

- token-weighted and task-macro candidate NLL and excess NLL;
- mean and worst-5% token `KL(fp32_state || candidate)`;
- top-1 next-token agreement with `fp32_state`;
- task-level median and quartiles of excess NLL;
- the same metrics by frozen reference-code-token-count quartile;
- resident payload, scale, metadata, padding, and transient bytes;
- prefill latency, teacher-forced decode latency per token, and peak CUDA
  allocated and reserved bytes.

Latency uses CUDA events with explicit synchronization, fixed GPU and power
settings recorded, one untimed warm-up pass, and five timed repeats over the
calibration set. Report the median repeat and dispersion. No sample may be
dropped as an outlier.

## Statistical procedure

The task is the resampling unit because tokens inside one program are not
independent. Use paired task-level percentile bootstrap intervals with 10,000
resamples and RNG seed 2339. All candidates use identical tasks and target
tokens. Report point estimates and two-sided 95% intervals, including failures
and negative effects.

There is one confirmatory statistical contrast: `read_risk_l0_nearest` versus
the within-task mean of the three exact-byte random controls on macro excess
NLL. It needs no multiplicity correction. MSE, stochastic-rounding, KL, top-1,
length-quartile, latency, and individual-random-layer comparisons are
prespecified secondary analyses. They receive effect sizes and intervals but no
standalone "significant" wording. If later reporting applies null-hypothesis
tests to that secondary family, Holm correction across the complete family is
required; favorable tests may not be selected after inspection.

## Development and confirmation sequence

### Stage 0: validity and calibration

1. Verify revisions, split IDs, architecture, state shapes, and token manifest.
2. Confirm that packed INT4 and INT8 round trips reproduce their QDQ values and
   that packed and QDQ model metrics agree to absolute tolerance `1e-6`.
3. Confirm the exact resident bytes in the table above.
4. Select and record the MSE baseline layer using only the 128 calibration
   tasks. Record the diagnostic public-data rank of layer 0 without changing
   the primary policy.
5. Freeze the evaluator commit, environment, command matrix, selected MSE
   layer, and artifact schema before development.

Any failed integration check stops the study. Fixing implementation defects is
allowed, but all affected calibration artifacts must be regenerated and the
fix disclosed before development.

### Stage 1: development

Run every frozen method on all 90 development tasks. Proceed to confirmation
only if all of the following hold:

- all values are finite and every method scores the identical token manifest;
- packed resident bytes exactly match the registered layouts;
- `read_risk_l0_nearest` reduces macro excess NLL by at least 15% relative to
  uniform INT4;
- the paired-bootstrap 95% interval for its equal-byte gain over the mean
  random control is entirely above zero;
- its mean token KL and worst-5% token KL are both lower than uniform INT4; and
- its top-1 agreement is not lower than uniform INT4.

These are continuation gates, not publishable confirmation. If a gate fails,
report the development failure and do not inspect the confirmation outcomes.
Changing a policy after development creates a new version and requires a new
untouched dataset.

Before confirmation, commit the frozen method configuration and a development
decision record. Merely changing formatting or correcting an evaluator bug does
not authorize retuning.

### Stage 2: untouched confirmation

If every development gate passes, run the already frozen method matrix once on
all 500 confirmation tasks. Do not run exploratory layer interventions, prompt
variants, threshold sweeps, or partial previews on these tasks first.

The v0.2 quality hypothesis passes only if:

- the point estimate gives at least 15% lower macro excess NLL than uniform
  INT4 and the paired-bootstrap interval favors `read_risk_l0_nearest`;
- the 95% interval for equal-byte gain over the mean random control is entirely
  above zero;
- mean and worst-5% token KL are lower than uniform INT4;
- top-1 agreement is not lower than uniform INT4; and
- every validity and byte-accounting condition still passes.

The performance claim gate is separate: packed median teacher-forced decode
latency must be no more than 1.10 times the FP32-state reference under the same
loop. Failing this gate does not erase a valid fidelity or resident-byte result,
but it prohibits a speed, efficient-runtime, or deployment-ready claim. Until a
fused kernel exists, latency regression is an expected possible result and must
be reported plainly.

A run may be replaced only for a documented infrastructure or evaluator failure
defined without reference to which method looked better. The invalid artifact,
reason, code change, and complete rerun must remain disclosed. A statistically
unfavorable result is not a failed run and cannot be rerun away.

## Claim boundary after confirmation

If the gates pass, defensible wording is limited to the evaluated scope, for
example:

> On `Qwen3.5-0.8B-Base` at the pinned revision, the frozen layer-0 INT8 plus
> 17-layer INT4 policy used 2,564,096 bytes of resident recurrent-state storage
> and preserved MBPP reference-code token likelihood better than the
> prespecified baselines under this teacher-forced protocol.

The exact measured effect and interval must replace any vague adjective.

This protocol cannot support claims that RecurQuant:

- is a new base model or that its authors created Qwen3.5;
- is the first recurrent-state, Mamba-state, or 4-bit cache quantizer;
- is a breakthrough, state of the art, lossless, or generally superior;
- improves generated-code correctness, `pass@k`, or executable behavior;
- reduces total model memory by 7.36x;
- speeds up inference unless the separate latency gate passes;
- generalizes beyond this checkpoint, architecture, language, or dataset; or
- is free of benchmark contamination.

MBPP is old and widely used. Whether its tasks or solutions appeared in the
Qwen pretraining corpus is unknown. Teacher-forced paired comparisons isolate
the effect of state quantization better than an absolute coding score, but they
do not remove possible pretraining contamination.

Failure is a valid result. If the frozen candidate misses a gate, publish the
negative evidence and treat any revised selector, including a future
state-Jacobian or empirical-Fisher method, as a separately preregistered
hypothesis rather than rewriting v0.2 after the fact.
