# Experiment 008 development identity

> **Status: exact `[16, 32)` identities pinned before model loading or quality
> evaluation.**

Identity resolved: 2026-07-23

The CORA-C2 protocol was already committed at `1a7c34a` before this window was
resolved. This step loaded the pinned public MBPP rows and local tokenizer only.
It did not load model weights, run a forward pass, compute logits, or observe a
quality metric.

## Frozen provenance

| Field | Value |
| --- | --- |
| Dataset | `google-research-datasets/mbpp`, config `full` |
| Dataset revision | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| Source split | `train` |
| Selection namespace | `rq-v0.2` |
| Ranked window | `[16, 32)` |
| Ordered task IDs | `666, 795, 944, 653, 857, 884, 878, 822, 687, 820, 920, 771, 869, 851, 728, 704` |
| Content-manifest SHA-256 | `21dcc6e1955918a9f6baae3d02e7ba2781600405f91fe42bbe18eac8ca6dde5e` |
| Prompt formatter | `recurquant.mbpp-prompt-code.v1` |
| Tokenizer | `Qwen2Tokenizer`, Transformers 5.14.1 |
| Tokenizer source | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Token-manifest SHA-256 | `5a8e7b56528e3ccecc95ff83b2e59749d81dab27d0233fefafc510622a973f87` |
| Total aligned scored tokens | 798 |
| Total full-code scored tokens | 814 |

The selector inputs remain the exact frozen pair:

```text
HRR canonical evidence SHA-256:
7970961fd88b522998189ad64f26b333aed9c88ff5f653de5449fd9e01d8cbc8

loss-sensitivity canonical evidence SHA-256:
bff4e33253990b8115e1f35e74516c4975c2fe4aac5066475afe968eb8a64609
```

## Ordered row and token identity

The rank column is zero-based in the complete frozen calibration ordering.
Row hashes come from canonical MBPP content. `Aligned scored` excludes the
prompt-to-first-code-token transition.

| Rank | Task ID | Row SHA-256 | Prompt tokens | Code tokens | Aligned scored |
| ---: | ---: | --- | ---: | ---: | ---: |
| 16 | 666 | `b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9` | 69 | 39 | 38 |
| 17 | 795 | `0da1480294ac295b6ceebf822dd2b512863c26b314e43f95ca6e5945e6ecfd82` | 290 | 35 | 34 |
| 18 | 944 | `9a7fbac2ccd13b8940f81ef75d20a4521cad8d39ccfe78714d85c49f712597d0` | 91 | 26 | 25 |
| 19 | 653 | `346aee9efc80955f531c1574db8b5c2eb25b48876c76a0aba1a35f15138800db` | 249 | 38 | 37 |
| 20 | 857 | `e9ac9acf8ca396274eef0b4479cfe2c0126ab6656222828b6198a497fa46f913` | 271 | 22 | 21 |
| 21 | 884 | `947efb4b2967163d5a5e7816e091b8b4a2a58c73e479c66ccc1b65fd7ea79cbb` | 112 | 68 | 67 |
| 22 | 878 | `88b2f4e5c166757c111f0baff74c20fa4d8f13a4fa13d8403c7cacc4f0558e9c` | 139 | 29 | 28 |
| 23 | 822 | `ba6eaa8bdd27506a5372e9e6beeb080d23bbbdc6996661d75ab26290d05fb3a4` | 71 | 132 | 131 |
| 24 | 687 | `350e5f813e96ccbba22a8db978e72840806c6e6fe4be28a06905c3eae5cb68ef` | 89 | 64 | 63 |
| 25 | 820 | `9ba789ae1cb3f3e21ec124ab7bd3447665019e54aa77fdb409c6d5a871994809` | 74 | 31 | 30 |
| 26 | 920 | `1539ae14cc92af940de077b6700a7a17ca2649aa26e165e15f6761cabde1363a` | 215 | 36 | 35 |
| 27 | 771 | `7470507a8c6c27b54bb1a56ef07e012d75c78bacbf8570aff8d0bb5f9919ac67` | 75 | 135 | 134 |
| 28 | 869 | `be0c42f832fb5afeb271845eee64ce97f919537ea266e41c881b798c9d15cabc` | 305 | 47 | 46 |
| 29 | 851 | `b1761ef5396f9ba11c5047b5cb0ff0955f3c8ad67f1c8449a4d7b394f8009076` | 95 | 36 | 35 |
| 30 | 728 | `1c0a375e35a7a918b8caaaf3314b75c36dd75e95f4ce01fa24ac80bb7c5af8cc` | 128 | 37 | 36 |
| 31 | 704 | `dfedafc5ad5c0985fcb5f3451f9434628f9d43077fe643ca1d5026106b5caccc` | 114 | 39 | 38 |

Any task order, row hash, prompt/code token count, manifest hash, tokenizer,
formatter, selector identity, or model revision mismatch must fail before model
execution.

Ranked window `[8, 16)` was not loaded or tokenized by this identity step.
