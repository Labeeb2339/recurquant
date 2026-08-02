# Experiment 010 Stage-A identity clarification

Date: 2026-07-31

Before the one permitted Stage-A quality run, I derived exact text and token-ID
fingerprints for the already-open MBPP task 666. This closes an identity gap in
the earlier Experiment 009 artifact, which fixed token counts but did not retain
the token-ID hashes.

## Frozen identity

- Dataset row SHA-256: `b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9`
- Prompt text SHA-256: `b6f0f93b9d15b96ac42bbabbdb349a09d2d24e57667d47cafe900c1ea91fd64b`
- Code text SHA-256: `d2701e79ccd968c9e5af78474af16256f3bbf39cdfedbec2199ac92e1a4f397e`
- Prompt token-ID SHA-256: `729215c4c99cdf96b13ad73f6ac7b537ddf9e882409b77e479d609aee046bffa`
- Code token-ID SHA-256: `a920370c4892513c8a5cdb9f88a33fd95d4c90201af39fdb7d517f3ad42a9d9a`
- Token counts: 69 prompt, 39 code, 38 aligned scored tokens

The token-ID hashes are SHA-256 over
`recurquant.evidence.canonical_json_bytes(list[int])`.

## Derivation boundary

The row was loaded through the guarded exact-ID calibration loader and accepted
only after its existing row hash matched. Text was produced by
`recurquant.public_data.format_mbpp_example`. Tokenization used
`Qwen/Qwen3.5-0.8B-Base` at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, `Qwen2Tokenizer`,
Transformers 5.14.1, prompt special tokens enabled, and code special tokens
disabled.

This was an identity-only clarification. It did not load a model configuration
or model weights, run a forward pass, inspect logits, calculate a quality
metric, select a new task, or access any row in the protected ranked
development window. It does not count as the Stage-A quality run and does not
support an improvement, novelty, or breakthrough claim.
