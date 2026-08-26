from __future__ import annotations

import itertools
import json
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from recurquant.rht import right_rht_decode, right_rht_encode
from recurquant.static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q48_PROMOTIONS,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    FROZEN_STATIC_Q468_UNIFORM_Q4_STEPS,
    FROZEN_STATIC_Q468_UNIFORM_Q8_STEPS,
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    STATIC_Q468_UNIFORM_Q4_METHOD,
    STATIC_Q468_UNIFORM_Q8_METHOD,
    StaticPackedRhtQ48State,
    StaticPackedRhtQ468State,
    StaticRhtQ468Geometry,
    allocate_exact_q48_mask,
    build_static_rht_q48_policy,
    build_static_rht_q468_policy,
    deserialize_static_rht_q48_policy,
    deserialize_static_rht_q468_policy,
    frozen_static_byte_accounting,
    load_static_rht_q48_policy,
    load_static_rht_q468_policy,
    pack_static_rht_q48,
    pack_static_rht_q468,
    save_static_rht_q48_policy,
    save_static_rht_q468_policy,
    serialize_static_rht_q48_policy,
    serialize_static_rht_q468_policy,
    static_q48_distortion_sha256,
    static_q468_distortion_sha256,
    verify_static_packed_rht_q48,
    verify_static_packed_rht_q468,
    verify_static_rht_q48_policy,
    verify_static_rht_q468_policy,
)

MANIFEST_SHA256 = "23" * 32
IDENTITY_SHA256 = "45" * 32
TOKENIZER_MANIFEST_SHA256 = "67" * 32
SOURCE_COMMIT = "89" * 20
BINDINGS = {
    "identity_artifact_sha256": IDENTITY_SHA256,
    "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
    "source_commit": SOURCE_COMMIT,
}
TINY_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=(0,),
    heads=1,
    key_rows=4,
    value_width=8,
    # K=3 data bytes are 39. The exact tiny layout owns eight reserved bytes.
    target_resident_bytes=47,
)
TINY_Q48_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=(0,),
    heads=1,
    key_rows=4,
    value_width=8,
    # P=2 data bytes are 41. The exact tiny layout owns eight reserved bytes.
    target_resident_bytes=49,
)


def _tiny_distortions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[8.0, 2.0, 5.0, 9.0]], dtype=torch.float64),
        torch.tensor([[7.0, 3.0, 1.0, 4.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 2.0, 6.0]], dtype=torch.float64),
    )


def _tiny_q48_distortions() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[5.0, 10.0, 4.0, 4.0]], dtype=torch.float64),
        torch.tensor([[0.0, 9.0, 1.0, 1.0]], dtype=torch.float64),
    )


def _brute_force_codes(
    distortions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    marginal_steps: int,
) -> torch.Tensor:
    rows = distortions[0].numel()
    stacked = torch.stack(distortions, dim=-1).reshape(rows, 3)
    exact = [tuple(Fraction.from_float(float(value)) for value in row) for row in stacked]
    candidates: list[tuple[Fraction, tuple[int, ...]]] = []
    for codes in itertools.product((0, 1, 2), repeat=rows):
        if sum(codes) != marginal_steps:
            continue
        objective = sum(
            (exact[row][code] for row, code in enumerate(codes)),
            start=Fraction(),
        )
        candidates.append((objective, codes))
    _, best = min(candidates, key=lambda item: (item[0], tuple(-code for code in item[1])))
    return torch.tensor(best, dtype=torch.uint8)


def _brute_force_q48_mask(
    distortions: tuple[torch.Tensor, torch.Tensor],
    promoted_rows: int,
) -> torch.Tensor:
    rows = distortions[0].numel()
    stacked = torch.stack(distortions, dim=-1).reshape(rows, 2)
    exact = [tuple(Fraction.from_float(float(value)) for value in row) for row in stacked]
    candidates: list[tuple[Fraction, tuple[int, ...]]] = []
    for mask in itertools.product((0, 1), repeat=rows):
        if sum(mask) != promoted_rows:
            continue
        objective = sum(
            (exact[row][selected] for row, selected in enumerate(mask)),
            start=Fraction(),
        )
        candidates.append((objective, mask))
    _, best = min(candidates, key=lambda item: (item[0], tuple(-bit for bit in item[1])))
    return torch.tensor(best, dtype=torch.bool)


def _decode_int4_pool_independently(payload: torch.Tensor) -> torch.Tensor:
    octets = payload.to(torch.int16)
    low = torch.bitwise_and(octets, 0x0F)
    high = torch.bitwise_right_shift(octets, 4)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    decoded = torch.empty(
        (payload.shape[0], payload.shape[1] * 2),
        dtype=torch.int16,
    )
    decoded[:, 0::2] = low
    decoded[:, 1::2] = high
    return decoded


def _decode_int6_pool_independently(payload: torch.Tensor) -> torch.Tensor:
    triples = payload.reshape(payload.shape[0], payload.shape[1] // 3, 3).to(torch.int16)
    byte0, byte1, byte2 = triples.unbind(dim=-1)
    unsigned = torch.stack(
        (
            torch.bitwise_and(byte0, 0x3F),
            torch.bitwise_right_shift(byte0, 6)
            | torch.bitwise_left_shift(torch.bitwise_and(byte1, 0x0F), 2),
            torch.bitwise_right_shift(byte1, 4)
            | torch.bitwise_left_shift(torch.bitwise_and(byte2, 0x03), 4),
            torch.bitwise_right_shift(byte2, 2),
        ),
        dim=-1,
    ).reshape(payload.shape[0], -1)
    return torch.where(unsigned >= 32, unsigned - 64, unsigned).to(torch.int16)


def _reference_quantize_rows(
    encoded: torch.Tensor,
    qmax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = encoded.abs().amax(dim=1, keepdim=True)
    ideal_scales = torch.where(
        absmax > 1.0e-8,
        absmax / qmax.to(torch.float32).unsqueeze(1),
        torch.ones_like(absmax),
    ).clamp(min=2.0**-24, max=torch.finfo(torch.float16).max)
    scales = ideal_scales.to(torch.float16).squeeze(1)
    normalized = encoded / scales.to(torch.float32).unsqueeze(1)
    limit = qmax.to(torch.float32).unsqueeze(1)
    codes = torch.minimum(torch.maximum(torch.round(normalized), -limit), limit)
    return codes.to(torch.int16), scales


def _tiny_policy(*, marginal_steps: int = 3):
    return build_static_rht_q468_policy(
        *_tiny_distortions(),
        geometry=TINY_GEOMETRY,
        marginal_steps=marginal_steps,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )


def _tiny_q48_policy(*, promoted_rows: int = 2):
    return build_static_rht_q48_policy(
        *_tiny_q48_distortions(),
        geometry=TINY_Q48_GEOMETRY,
        promoted_rows=promoted_rows,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )


def test_frozen_static_ledgers_distinguish_data_alignment_and_budget_eligibility() -> None:
    accounting = frozen_static_byte_accounting()

    assert accounting[STATIC_Q468_PRIMARY_METHOD] == {
        "alignment_bytes": 8,
        "budget_delta_bytes": 0,
        "codec": "q468",
        "data_bytes": 3_454_656,
        "exact_budget_eligible": True,
        "method_id": STATIC_Q468_PRIMARY_METHOD,
        "payload_bytes": 3_297_984,
        "pool_offset_bytes": 73_728,
        "precision_code_bytes": 9_216,
        "resident_bytes": 3_454_664,
        "scale_bytes": 73_728,
        "selected_units": FROZEN_STATIC_Q468_PRIMARY_STEPS,
        "target_resident_bytes": 3_454_664,
    }
    assert accounting[STATIC_Q468_ABLATION_METHOD] == {
        "alignment_bytes": 0,
        "budget_delta_bytes": 73_736,
        "codec": "q468",
        "data_bytes": 3_380_928,
        "exact_budget_eligible": False,
        "method_id": STATIC_Q468_ABLATION_METHOD,
        "payload_bytes": 3_224_256,
        "pool_offset_bytes": 73_728,
        "precision_code_bytes": 9_216,
        "resident_bytes": 3_380_928,
        "scale_bytes": 73_728,
        "selected_units": FROZEN_STATIC_Q468_ABLATION_STEPS,
        "target_resident_bytes": 3_454_664,
    }
    for method_id in (
        STATIC_Q468_MSE_METHOD,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    ):
        assert accounting[method_id] == {
            **accounting[STATIC_Q468_PRIMARY_METHOD],
            "method_id": method_id,
        }
    assert accounting[STATIC_Q468_UNIFORM_Q4_METHOD] == {
        "alignment_bytes": 0,
        "budget_delta_bytes": 938_696,
        "codec": "q468",
        "data_bytes": 2_515_968,
        "exact_budget_eligible": False,
        "method_id": STATIC_Q468_UNIFORM_Q4_METHOD,
        "payload_bytes": 2_359_296,
        "pool_offset_bytes": 73_728,
        "precision_code_bytes": 9_216,
        "resident_bytes": 2_515_968,
        "scale_bytes": 73_728,
        "selected_units": FROZEN_STATIC_Q468_UNIFORM_Q4_STEPS,
        "target_resident_bytes": 3_454_664,
    }
    assert accounting[STATIC_Q468_UNIFORM_Q8_METHOD] == {
        "alignment_bytes": 0,
        "budget_delta_bytes": -1_420_600,
        "codec": "q468",
        "data_bytes": 4_875_264,
        "exact_budget_eligible": False,
        "method_id": STATIC_Q468_UNIFORM_Q8_METHOD,
        "payload_bytes": 4_718_592,
        "pool_offset_bytes": 73_728,
        "precision_code_bytes": 9_216,
        "resident_bytes": 4_875_264,
        "scale_bytes": 73_728,
        "selected_units": FROZEN_STATIC_Q468_UNIFORM_Q8_STEPS,
        "target_resident_bytes": 3_454_664,
    }
    assert accounting[STATIC_Q48_COMPARATOR_METHOD] == {
        "alignment_bytes": 8,
        "budget_delta_bytes": 0,
        "codec": "q48",
        "data_bytes": 3_454_656,
        "exact_budget_eligible": True,
        "method_id": STATIC_Q48_COMPARATOR_METHOD,
        "payload_bytes": 3_302_592,
        "pool_offset_bytes": 73_728,
        "precision_code_bytes": 4_608,
        "resident_bytes": 3_454_664,
        "scale_bytes": 73_728,
        "selected_units": FROZEN_STATIC_Q48_PROMOTIONS,
        "target_resident_bytes": 3_454_664,
    }


@pytest.mark.parametrize("promoted_rows", range(5))
def test_static_q48_selector_matches_exhaustive_search_for_every_tiny_budget(
    promoted_rows: int,
) -> None:
    distortions = _tiny_q48_distortions()

    actual = allocate_exact_q48_mask(*distortions, promoted_rows=promoted_rows)
    expected = _brute_force_q48_mask(distortions, promoted_rows)
    policy = _tiny_q48_policy(promoted_rows=promoted_rows)

    assert torch.equal(actual.reshape(-1), expected)
    assert torch.equal(policy.high_precision_mask().reshape(-1), expected)
    assert int(actual.sum().item()) == promoted_rows


def test_static_q48_selector_uses_earlier_flattened_row_on_exact_tie() -> None:
    d4 = torch.tensor([[4.0, 4.0, 4.0]])
    d8 = torch.tensor([[1.0, 1.0, 1.0]])

    mask = allocate_exact_q48_mask(d4, d8, promoted_rows=2)

    assert torch.equal(mask, torch.tensor([[True, True, False]]))


@pytest.mark.parametrize("marginal_steps", range(9))
def test_static_policy_matches_exhaustive_allocation_for_every_tiny_budget(
    marginal_steps: int,
) -> None:
    distortions = _tiny_distortions()

    policy = build_static_rht_q468_policy(
        *distortions,
        geometry=TINY_GEOMETRY,
        marginal_steps=marginal_steps,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )

    assert torch.equal(
        policy.precision_codes().reshape(-1),
        _brute_force_codes(distortions, marginal_steps),
    )
    assert sum(policy.pool_counts) == TINY_GEOMETRY.total_rows
    assert int(policy.precision_codes().to(torch.int64).sum().item()) == marginal_steps


def test_policy_codes_offsets_hashes_and_serialization_are_deterministic(tmp_path) -> None:
    first = _tiny_policy()
    second = _tiny_policy()

    assert torch.equal(first.packed_precision_codes, second.packed_precision_codes)
    assert torch.equal(first.pool_offsets, second.pool_offsets)
    assert first.pool_offsets.dtype == torch.uint16
    assert first.pool_counts == second.pool_counts
    assert first.calibration_scores_sha256 == static_q468_distortion_sha256(
        *_tiny_distortions(),
        geometry=TINY_GEOMETRY,
    )
    assert first.model_id == "Qwen/Qwen3.5-0.8B-Base"
    assert first.model_revision == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert first.tokenizer_id == first.model_id
    assert first.tokenizer_revision == first.model_revision
    assert first.tokenizer_manifest_sha256 == TOKENIZER_MANIFEST_SHA256
    assert first.identity_artifact_sha256 == IDENTITY_SHA256
    assert first.source_commit == SOURCE_COMMIT
    assert len(first.code_map_sha256) == 64
    assert len(first.pool_offsets_sha256) == 64
    assert first.policy_sha256 == second.policy_sha256
    assert serialize_static_rht_q468_policy(first) == serialize_static_rht_q468_policy(second)

    path = tmp_path / "policy.json"
    save_static_rht_q468_policy(first, path)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_static_rht_q468_policy(second, path)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".policy.json.*.tmp")) == []
    loaded = load_static_rht_q468_policy(path)
    assert loaded.policy_sha256 == first.policy_sha256
    assert loaded.evidence_dict() == first.evidence_dict()
    assert torch.equal(loaded.precision_codes(), first.precision_codes())
    assert torch.equal(loaded.pool_offsets, first.pool_offsets)
    assert (
        verify_static_rht_q468_policy(
            loaded,
            expected_policy_sha256=first.policy_sha256,
        )["policy_sha256"]
        == first.policy_sha256
    )


def test_q48_policy_artifact_is_deterministic_strict_and_atomically_published(
    tmp_path,
) -> None:
    first = _tiny_q48_policy()
    second = _tiny_q48_policy()

    assert torch.equal(
        first.high_precision_mask().reshape(-1),
        torch.tensor([True, False, True, False]),
    )
    assert first.pool_counts == (2, 2)
    assert torch.equal(first.pool_offsets, torch.tensor([0, 0, 1, 1], dtype=torch.uint16))
    assert first.calibration_scores_sha256 == static_q48_distortion_sha256(
        *_tiny_q48_distortions(),
        geometry=TINY_Q48_GEOMETRY,
    )
    assert first.model_revision == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert first.tokenizer_manifest_sha256 == TOKENIZER_MANIFEST_SHA256
    assert first.identity_artifact_sha256 == IDENTITY_SHA256
    assert first.source_commit == SOURCE_COMMIT
    assert first.mask_sha256 == second.mask_sha256
    assert first.policy_sha256 == second.policy_sha256

    serialized = serialize_static_rht_q48_policy(first)
    assert serialized == serialize_static_rht_q48_policy(second)
    loaded = deserialize_static_rht_q48_policy(serialized)
    assert loaded.evidence_dict() == first.evidence_dict()
    assert (
        verify_static_rht_q48_policy(
            loaded,
            expected_policy_sha256=first.policy_sha256,
        )["policy_sha256"]
        == first.policy_sha256
    )

    path = tmp_path / "nested" / "q48-policy.json"
    save_static_rht_q48_policy(first, path)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_static_rht_q48_policy(second, path)
    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob(".q48-policy.json.*.tmp")) == []
    assert load_static_rht_q48_policy(path).policy_sha256 == first.policy_sha256


def test_q48_policy_tamper_invalid_offsets_and_identity_fail_closed() -> None:
    policy = _tiny_q48_policy()
    offsets = policy.pool_offsets.clone()
    offsets[-1] = 17
    with pytest.raises(ValueError, match="canonical per-pool prefix"):
        replace(policy, pool_offsets=offsets)
    with pytest.raises(ValueError, match="pool_counts"):
        replace(policy, pool_counts=(3, 2))
    with pytest.raises(ValueError, match="40-hex"):
        replace(policy, source_commit="main")
    invalid_mask = policy.packed_precision_mask.clone()
    invalid_mask[-1] = int(invalid_mask[-1].item()) | 0x80
    with pytest.raises(ValueError, match="unused precision-mask padding bits"):
        replace(policy, packed_precision_mask=invalid_mask)

    serialized = serialize_static_rht_q48_policy(policy)
    decoded = json.loads(serialized)
    decoded["content"]["tokenizer_revision"] = "a" * 40
    tampered = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="does not authenticate"):
        deserialize_static_rht_q48_policy(tampered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_revision", "incompatible-policy-v999", "policy revision"),
        ("selector_revision", "incompatible-selector-v999", "selector revision"),
        ("codec_revision", "incompatible-codec-v999", "codec revision"),
    ],
)
def test_q48_policy_rejects_unsupported_runtime_contracts(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_tiny_q48_policy(), **{field: value})

    with pytest.raises(ValueError, match="frozen geometry"):
        replace(_tiny_q48_policy(), method_id=STATIC_Q48_COMPARATOR_METHOD)

    with pytest.raises(ValueError, match="Q468 method cannot identify a Q48 policy"):
        replace(_tiny_q48_policy(), method_id=STATIC_Q468_PRIMARY_METHOD)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "model_id"),
        ("model_revision", "main", "40-hex"),
        ("tokenizer_id", " tokenizer", "stripped printable"),
        ("tokenizer_revision", "f" * 64, "40-hex"),
        ("tokenizer_manifest_sha256", "f" * 40, "64-character"),
        ("transformers_version", "5.14.1 dev", "semantic version"),
        ("identity_artifact_sha256", "z" * 64, "64-character"),
        ("source_commit", "ABCDEF" * 6 + "abcd", "40-hex"),
    ],
)
def test_policy_fails_closed_on_invalid_identity_bindings(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_tiny_policy(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_revision", "incompatible-policy-v999", "policy revision"),
        ("allocator_revision", "incompatible-allocator-v999", "allocator revision"),
        ("codec_revision", "incompatible-codec-v999", "codec revision"),
    ],
)
def test_q468_policy_rejects_unsupported_runtime_contracts(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_tiny_policy(), **{field: value})

    with pytest.raises(ValueError, match="frozen geometry"):
        replace(_tiny_policy(), method_id=STATIC_Q468_PRIMARY_METHOD)

    with pytest.raises(ValueError, match="Q48 method cannot identify a Q468 policy"):
        replace(_tiny_policy(), method_id=STATIC_Q48_COMPARATOR_METHOD)


@pytest.mark.parametrize(
    "method_id",
    [
        STATIC_Q468_MSE_METHOD,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        STATIC_Q468_UNIFORM_Q4_METHOD,
        STATIC_Q468_UNIFORM_Q8_METHOD,
    ],
)
def test_reserved_q468_comparator_methods_cannot_identify_q48_policy(
    method_id: str,
) -> None:
    with pytest.raises(ValueError, match="Q468 method cannot identify a Q48 policy"):
        replace(_tiny_q48_policy(), method_id=method_id)


def test_pool_offsets_are_canonical_prefix_indices_within_each_pool() -> None:
    policy = _tiny_policy()
    codes = policy.precision_codes().reshape(-1)

    for code, count in enumerate(policy.pool_counts):
        observed = policy.pool_offsets.to(torch.int64)[codes == code]
        assert torch.equal(observed, torch.arange(count, dtype=torch.int64))


def test_policy_rejects_invalid_offsets_counts_and_reserved_codes() -> None:
    policy = _tiny_policy()
    invalid_offsets = policy.pool_offsets.clone()
    invalid_offsets[-1] = 99
    with pytest.raises(ValueError, match="canonical per-pool prefix"):
        replace(policy, pool_offsets=invalid_offsets)

    with pytest.raises(ValueError, match="pool_counts"):
        replace(policy, pool_counts=(policy.pool_counts[0] + 1, *policy.pool_counts[1:]))

    invalid_codes = policy.packed_precision_codes.clone()
    invalid_codes[0] = int(invalid_codes[0].item()) | 0x03
    with pytest.raises(ValueError, match="reserved precision code 3"):
        replace(policy, packed_precision_codes=invalid_codes)


def test_policy_rejects_mismatched_calibration_hash_and_noncanonical_artifact() -> None:
    distortions = _tiny_distortions()
    with pytest.raises(ValueError, match="does not match supplied distortions"):
        build_static_rht_q468_policy(
            *distortions,
            geometry=TINY_GEOMETRY,
            marginal_steps=3,
            calibration_manifest_sha256=MANIFEST_SHA256,
            calibration_scores_sha256="ab" * 32,
            **BINDINGS,
        )

    serialized = serialize_static_rht_q468_policy(_tiny_policy())
    decoded = json.loads(serialized)
    decoded["content"]["model_revision"] = "a" * 40
    tampered = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="does not authenticate"):
        deserialize_static_rht_q468_policy(tampered)

    with pytest.raises(ValueError, match="not in canonical form"):
        deserialize_static_rht_q468_policy(serialized + b"\n")


def test_tiny_static_pack_owns_exact_bytes_without_persistent_fp32() -> None:
    generator = torch.Generator().manual_seed(2339)
    source = {0: torch.randn((1, 1, 4, 8), generator=generator)}
    policy = _tiny_policy()

    packed = pack_static_rht_q468(source, policy)

    assert packed.data_bytes == 39
    assert packed.resident_bytes == 47
    assert packed.ledger.exact_budget_eligible is True
    assert (
        packed.policy.packed_precision_codes.data_ptr() != policy.packed_precision_codes.data_ptr()
    )
    assert packed.policy.pool_offsets.data_ptr() != policy.pool_offsets.data_ptr()
    assert all(
        tensor.dtype not in (torch.float32, torch.float64)
        for _, tensor in packed.persistent_tensors()
    )
    assert [name for name, _ in packed.persistent_tensors()] == [
        "int4_payload",
        "int6_payload",
        "int8_payload",
        "scales",
        "packed_precision_codes",
        "pool_offsets",
        "padding",
    ]
    restored = packed.materialize()
    assert set(restored) == {0}
    assert restored[0].shape == source[0].shape
    assert restored[0].dtype == torch.float32
    assert torch.isfinite(restored[0]).all().item()
    assert restored[0].data_ptr() != source[0].data_ptr()

    evidence = verify_static_packed_rht_q468(packed)
    assert evidence["physical_data_bytes"] == 39
    assert evidence["physical_resident_bytes"] == 47
    assert "torch.float32" not in evidence["persistent_tensor_dtypes"].values()


def test_tiny_q48_pack_materializes_and_owns_exact_physical_bytes() -> None:
    generator = torch.Generator().manual_seed(2340)
    source = {0: torch.randn((1, 1, 4, 8), generator=generator)}
    policy = _tiny_q48_policy()

    packed = pack_static_rht_q48(source, policy)

    assert packed.data_bytes == 41
    assert packed.resident_bytes == 49
    assert packed.ledger.exact_budget_eligible is True
    assert packed.policy.packed_precision_mask.data_ptr() != policy.packed_precision_mask.data_ptr()
    assert packed.policy.pool_offsets.data_ptr() != policy.pool_offsets.data_ptr()
    assert [name for name, _ in packed.persistent_tensors()] == [
        "low_payload",
        "high_payload",
        "scales",
        "packed_precision_mask",
        "pool_offsets",
        "padding",
    ]
    assert all(
        tensor.dtype not in (torch.float32, torch.float64)
        for _, tensor in packed.persistent_tensors()
    )
    restored = packed.materialize()
    assert set(restored) == {0}
    assert restored[0].shape == source[0].shape
    assert restored[0].dtype == torch.float32
    assert torch.isfinite(restored[0]).all().item()
    assert restored[0].data_ptr() != source[0].data_ptr()

    evidence = verify_static_packed_rht_q48(packed)
    assert evidence["physical_data_bytes"] == 41
    assert evidence["physical_resident_bytes"] == 49
    assert "torch.float32" not in evidence["persistent_tensor_dtypes"].values()


def test_pool_offsets_independently_reconstruct_q468_and_q48_payloads() -> None:
    generator = torch.Generator().manual_seed(2341)
    source = {0: torch.randn((1, 1, 4, 8), generator=generator)}
    encoded = right_rht_encode(
        source[0],
        layer_index=0,
        expected_heads=1,
        output_dtype=torch.float32,
    ).reshape(TINY_GEOMETRY.total_rows, TINY_GEOMETRY.value_width)

    q468 = pack_static_rht_q468(source, _tiny_policy())
    q468_code_bytes = q468.policy.packed_precision_codes
    q468_codes = torch.tensor(
        [
            (int(q468_code_bytes[row // 4].item()) >> (2 * (row % 4))) & 0x03
            for row in range(TINY_GEOMETRY.total_rows)
        ],
        dtype=torch.uint8,
    )
    q468_qmax = torch.tensor((7, 31, 127), dtype=torch.int16)[q468_codes.to(torch.long)]
    expected_q468_codes, expected_q468_scales = _reference_quantize_rows(
        encoded,
        q468_qmax,
    )
    q468_pools = (
        _decode_int4_pool_independently(q468.int4_payload),
        _decode_int6_pool_independently(q468.int6_payload),
        q468.int8_payload.to(torch.int16),
    )
    observed_q468_codes = torch.empty_like(expected_q468_codes)
    q468_offsets = q468.policy.pool_offsets.to(torch.int64)
    for precision_code, pool in enumerate(q468_pools):
        selected = q468_codes == precision_code
        observed_q468_codes[selected] = pool.index_select(0, q468_offsets[selected])

    assert torch.equal(q468.scales, expected_q468_scales)
    assert torch.equal(observed_q468_codes, expected_q468_codes)

    q48 = pack_static_rht_q48(source, _tiny_q48_policy())
    q48_mask_bytes = q48.policy.packed_precision_mask
    q48_mask = torch.tensor(
        [
            bool((int(q48_mask_bytes[row // 8].item()) >> (row % 8)) & 0x01)
            for row in range(TINY_Q48_GEOMETRY.total_rows)
        ],
        dtype=torch.bool,
    )
    q48_qmax = torch.where(q48_mask, 127, 7).to(torch.int16)
    expected_q48_codes, expected_q48_scales = _reference_quantize_rows(encoded, q48_qmax)
    q48_pools = (
        _decode_int4_pool_independently(q48.low_payload),
        q48.high_payload.to(torch.int16),
    )
    observed_q48_codes = torch.empty_like(expected_q48_codes)
    q48_offsets = q48.policy.pool_offsets.to(torch.int64)
    for high_precision, pool in enumerate(q48_pools):
        selected = q48_mask == bool(high_precision)
        observed_q48_codes[selected] = pool.index_select(0, q48_offsets[selected])

    assert torch.equal(q48.scales, expected_q48_scales)
    assert torch.equal(observed_q48_codes, expected_q48_codes)

    for packed, integer_codes, scales in (
        (q468, observed_q468_codes, expected_q468_scales),
        (q48, observed_q48_codes, expected_q48_scales),
    ):
        manually_dequantized = integer_codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(
            1
        )
        manually_restored = right_rht_decode(
            manually_dequantized.reshape(1, 1, 4, 8),
            layer_index=0,
            expected_heads=1,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(
            packed.materialize()[0],
            manually_restored,
            rtol=0.0,
            atol=0.0,
        )


def test_q48_packed_state_rejects_alias_hidden_fp32_and_bad_payload() -> None:
    source = {0: torch.ones((1, 1, 4, 8), dtype=torch.float32)}
    packed = pack_static_rht_q48(source, _tiny_q48_policy())

    with pytest.raises(ValueError, match="may not use FP32"):
        replace(packed, scales=packed.scales.to(torch.float32))
    aliased_scales = packed.low_payload.view(torch.float16).reshape(-1)
    with pytest.raises(ValueError, match="aliases low_payload"):
        replace(packed, scales=aliased_scales)
    with pytest.raises(TypeError, match="low_payload must have shape"):
        replace(packed, low_payload=packed.low_payload[:-1].clone())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_static_q468_and_q48_reference_codecs_round_trip_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(2339)
    source = {0: torch.randn((1, 1, 4, 8), generator=generator, device="cuda")}

    q468 = pack_static_rht_q468(source, _tiny_policy())
    q48 = pack_static_rht_q48(source, _tiny_q48_policy())

    for packed, verifier in (
        (q468, verify_static_packed_rht_q468),
        (q48, verify_static_packed_rht_q48),
    ):
        restored = packed.materialize()[0]
        torch.cuda.synchronize()
        assert restored.device.type == "cuda"
        assert torch.isfinite(restored).all().item()
        assert verifier(packed)["physical_resident_bytes"] == packed.resident_bytes


def test_packed_state_fails_closed_on_hidden_fp32_and_bad_pool_shape() -> None:
    source = {0: torch.ones((1, 1, 4, 8), dtype=torch.float32)}
    packed = pack_static_rht_q468(source, _tiny_policy())

    with pytest.raises(ValueError, match="may not use FP32"):
        replace(packed, scales=packed.scales.to(torch.float32))

    with pytest.raises(ValueError, match="int4_payload must have shape"):
        replace(packed, int4_payload=packed.int4_payload[:-1].clone())


def test_real_geometry_primary_policy_has_exact_k_counts_and_uint16_offsets() -> None:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    row = torch.arange(geometry.total_rows, dtype=torch.float64)
    distortions = (
        (((17 * row + 13) % 1009) / 1009).reshape(geometry.layers, -1),
        (((29 * row + 7) % 1013) / 1013).reshape(geometry.layers, -1),
        (((43 * row + 3) % 1019) / 1019).reshape(geometry.layers, -1),
    )

    policy = build_static_rht_q468_policy(
        *distortions,
        geometry=geometry,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        method_id=STATIC_Q468_PRIMARY_METHOD,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )

    codes = policy.precision_codes().reshape(-1)
    assert codes.shape == (36_864,)
    assert int(codes.to(torch.int64).sum().item()) == FROZEN_STATIC_Q468_PRIMARY_STEPS
    assert sum(policy.pool_counts) == 36_864
    assert policy.packed_precision_codes.numel() == 9_216
    assert policy.pool_offsets.dtype == torch.uint16
    assert policy.pool_offsets.numel() == 36_864
    for code, count in enumerate(policy.pool_counts):
        observed = policy.pool_offsets.to(torch.int64)[codes == code]
        assert torch.equal(observed, torch.arange(count, dtype=torch.int64))
    assert policy.evidence_dict()["ledger"]["resident_bytes"] == 3_454_664
    with pytest.raises(ValueError, match="wrong exact-K budget"):
        replace(policy, marginal_steps=0)
    with pytest.raises(ValueError, match="frozen model identity"):
        replace(policy, model_id="Qwen/other-model")


@pytest.mark.parametrize(
    "method_id",
    [
        STATIC_Q468_MSE_METHOD,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    ],
)
def test_official_k29334_comparator_policy_round_trip_and_strict_budget(
    method_id: str,
) -> None:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    row = torch.arange(geometry.total_rows, dtype=torch.float64)
    policy = build_static_rht_q468_policy(
        (((17 * row + 13) % 1009) / 1009).reshape(geometry.layers, -1),
        (((29 * row + 7) % 1013) / 1013).reshape(geometry.layers, -1),
        (((43 * row + 3) % 1019) / 1019).reshape(geometry.layers, -1),
        geometry=geometry,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        method_id=method_id,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )

    serialized = serialize_static_rht_q468_policy(policy)
    restored = deserialize_static_rht_q468_policy(serialized)
    ledger = restored.evidence_dict()["ledger"]

    assert restored.method_id == method_id
    assert restored.marginal_steps == FROZEN_STATIC_Q468_PRIMARY_STEPS
    assert restored.policy_sha256 == policy.policy_sha256
    assert serialize_static_rht_q468_policy(restored) == serialized
    assert ledger["resident_bytes"] == 3_454_664
    assert ledger["target_resident_bytes"] == 3_454_664
    assert ledger["budget_delta_bytes"] == 0
    assert ledger["exact_budget_eligible"] is True

    with pytest.raises(ValueError, match="wrong exact-K budget"):
        replace(policy, marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS - 1)


@pytest.mark.parametrize(
    ("method_id", "marginal_steps", "expected_code", "expected_resident"),
    [
        (
            STATIC_Q468_UNIFORM_Q4_METHOD,
            FROZEN_STATIC_Q468_UNIFORM_Q4_STEPS,
            0,
            2_515_968,
        ),
        (
            STATIC_Q468_UNIFORM_Q8_METHOD,
            FROZEN_STATIC_Q468_UNIFORM_Q8_STEPS,
            2,
            4_875_264,
        ),
    ],
)
def test_official_uniform_rht_anchor_policy_uses_same_q468_physical_codec(
    method_id: str,
    marginal_steps: int,
    expected_code: int,
    expected_resident: int,
) -> None:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    row = torch.arange(geometry.total_rows, dtype=torch.float64).reshape(1, -1)
    policy = build_static_rht_q468_policy(
        row + 3.0,
        row + 2.0,
        row + 1.0,
        geometry=geometry,
        marginal_steps=marginal_steps,
        method_id=method_id,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )

    assert torch.all(policy.precision_codes() == expected_code).item()
    assert policy.evidence_dict()["ledger"]["resident_bytes"] == expected_resident
    assert policy.codec_revision == "rht-q468-pools-u16-offsets-v1"
    restored = deserialize_static_rht_q468_policy(serialize_static_rht_q468_policy(policy))
    assert restored.policy_sha256 == policy.policy_sha256

    wrong_steps = 1 if marginal_steps == 0 else marginal_steps - 1
    with pytest.raises(ValueError, match="wrong exact-K budget"):
        replace(policy, marginal_steps=wrong_steps)
    with pytest.raises(ValueError, match="frozen geometry"):
        replace(
            policy,
            geometry=replace(
                geometry,
                target_resident_bytes=geometry.target_resident_bytes + 8,
            ),
        )


def test_real_q468_policy_has_a_physical_exact_3454664_byte_state() -> None:
    """Construct every real Q4/Q6/Q8 pool without allocating dense model state."""

    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    row = torch.arange(geometry.total_rows, dtype=torch.float64)
    distortions = (
        (((17 * row + 13) % 1009) / 1009).reshape(geometry.layers, -1),
        (((29 * row + 7) % 1013) / 1013).reshape(geometry.layers, -1),
        (((43 * row + 3) % 1019) / 1019).reshape(geometry.layers, -1),
    )
    policy = build_static_rht_q468_policy(
        *distortions,
        geometry=geometry,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        method_id=STATIC_Q468_PRIMARY_METHOD,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )
    q4_count, q6_count, q8_count = policy.pool_counts
    packed = StaticPackedRhtQ468State(
        policy=policy,
        int4_payload=torch.zeros((q4_count, geometry.value_width * 4 // 8), dtype=torch.uint8),
        int6_payload=torch.zeros((q6_count, geometry.value_width * 6 // 8), dtype=torch.uint8),
        int8_payload=torch.zeros((q8_count, geometry.value_width), dtype=torch.int8),
        scales=torch.ones(geometry.total_rows, dtype=torch.float16),
        padding=torch.zeros(8, dtype=torch.uint8),
    )

    evidence = verify_static_packed_rht_q468(packed)
    assert q4_count + q6_count + q8_count == 36_864
    assert (
        packed.int4_payload.numel() + packed.int6_payload.numel() + packed.int8_payload.numel()
        == 3_297_984
    )
    assert policy.packed_precision_codes.numel() == 9_216
    assert policy.pool_offsets.numel() * policy.pool_offsets.element_size() == 73_728
    assert packed.scales.numel() * packed.scales.element_size() == 73_728
    assert packed.data_bytes == 3_454_656
    assert packed.resident_bytes == 3_454_664
    assert evidence["physical_resident_bytes"] == 3_454_664

    source_states = {
        layer_index: torch.zeros(
            (1, geometry.heads, geometry.key_rows, geometry.value_width),
            dtype=torch.float32,
        )
        for layer_index in geometry.layer_indices
    }
    with torch.no_grad():
        physically_packed = pack_static_rht_q468(source_states, policy)
    physical_evidence = verify_static_packed_rht_q468(physically_packed)
    assert physically_packed.policy.pool_counts == packed.policy.pool_counts
    assert physically_packed.resident_bytes == 3_454_664
    assert physical_evidence["physical_resident_bytes"] == 3_454_664


def test_real_q48_policy_has_a_physical_exact_3454664_byte_state() -> None:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    row = torch.arange(geometry.total_rows, dtype=torch.float64)
    d4 = (((19 * row + 11) % 1021) / 1021).reshape(geometry.layers, -1)
    d8 = (((31 * row + 5) % 1031) / 1031).reshape(geometry.layers, -1)
    policy = build_static_rht_q48_policy(
        d4,
        d8,
        geometry=geometry,
        promoted_rows=FROZEN_STATIC_Q48_PROMOTIONS,
        method_id=STATIC_Q48_COMPARATOR_METHOD,
        calibration_manifest_sha256=MANIFEST_SHA256,
        **BINDINGS,
    )
    low_count, high_count = policy.pool_counts

    packed = StaticPackedRhtQ48State(
        policy=policy,
        low_payload=torch.zeros(
            (low_count, geometry.value_width * 4 // 8),
            dtype=torch.uint8,
        ),
        high_payload=torch.zeros(
            (high_count, geometry.value_width),
            dtype=torch.int8,
        ),
        scales=torch.ones(geometry.total_rows, dtype=torch.float16),
        padding=torch.zeros(8, dtype=torch.uint8),
    )

    assert policy.packed_precision_mask.numel() == 4_608
    assert policy.pool_offsets.numel() * policy.pool_offsets.element_size() == 73_728
    assert policy.pool_counts == (36_864 - 14_739, 14_739)
    assert packed.low_payload.numel() + packed.high_payload.numel() == 3_302_592
    assert packed.scales.numel() * packed.scales.element_size() == 73_728
    assert packed.data_bytes == 3_454_656
    assert packed.resident_bytes == 3_454_664
    assert packed.ledger.exact_budget_eligible is True
    with pytest.raises(ValueError, match="wrong exact-P budget"):
        replace(policy, promoted_rows=0)
    with pytest.raises(ValueError, match="frozen model identity"):
        replace(policy, tokenizer_id="Qwen/other-tokenizer")
