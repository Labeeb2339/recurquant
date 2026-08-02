from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import recurquant.statelease_equal_byte_baselines as baselines
from recurquant.statelease_equal_byte_baselines import (
    EXPANDED_RHT_Q4_Q8,
    RHT_Q4_Q6_Q8,
    RHT_RESIDUAL_Q4,
    EqualByteCheckpoint,
    EqualByteLayout,
    EqualByteNoReplayCache,
    ExpandedRhtQ4Q8Layer,
    RhtQ4Q6Q8Layer,
    RhtResidualQ4Layer,
    frozen_equal_byte_accounting,
    pack_expanded_rht_q4_q8,
    pack_rht_q4_q6_q8,
    pack_rht_residual_q4,
    update_causal_query_ema,
)

TINY_LAYOUT = EqualByteLayout(
    layer_indices=(0, 3),
    heads=2,
    key_rows=4,
    value_width=8,
    expanded_q8_promotions=5,
    multibit_marginal_steps=4,
    residual_q4_rows=2,
    expanded_padding_bytes=2,
    multibit_padding_bytes=12,
    residual_padding_bytes=10,
    expected_resident_bytes=184,
)


def _states(*, seed: int = 2339) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        layer_index: torch.randn((1, 2, 4, 8), generator=generator)
        for layer_index in TINY_LAYOUT.layer_indices
    }


def _ema(*, seed: int = 2340) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (0.1 + torch.rand((2, 2, 4), generator=generator)).to(torch.float32)


def _queries(*, seed: int = 2341, tokens: int = 1) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        layer_index: torch.randn((1, tokens, 2, 4), generator=generator)
        for layer_index in TINY_LAYOUT.layer_indices
    }


PACKERS = {
    EXPANDED_RHT_Q4_Q8: pack_expanded_rht_q4_q8,
    RHT_Q4_Q6_Q8: pack_rht_q4_q6_q8,
    RHT_RESIDUAL_Q4: pack_rht_residual_q4,
}


def test_frozen_component_arithmetic_matches_protocol_exactly() -> None:
    accounting = frozen_equal_byte_accounting()

    assert accounting[EXPANDED_RHT_Q4_Q8] == {
        "payload_bytes": 3_228_864,
        "scale_bytes": 73_728,
        "precision_bytes": 4_608,
        "query_ema_bytes": 147_456,
        "padding_bytes": 8,
        "resident_bytes": 3_454_664,
    }
    assert accounting[RHT_Q4_Q6_Q8] == {
        "payload_bytes": 3_224_256,
        "scale_bytes": 73_728,
        "precision_bytes": 9_216,
        "query_ema_bytes": 147_456,
        "padding_bytes": 8,
        "resident_bytes": 3_454_664,
    }
    assert accounting[RHT_RESIDUAL_Q4] == {
        "payload_bytes": 3_202_496,
        "scale_bytes": 100_078,
        "precision_bytes": 4_608,
        "query_ema_bytes": 147_456,
        "padding_bytes": 26,
        "resident_bytes": 3_454_664,
    }
    assert baselines.FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.fp32_state_bytes == 18_874_368


def test_layout_rejects_a_comparator_that_does_not_spend_the_same_total() -> None:
    with pytest.raises(ValueError, match="mismatched"):
        replace(TINY_LAYOUT, expanded_padding_bytes=1)


@pytest.mark.parametrize("codec", tuple(PACKERS))
def test_each_codec_owns_exact_bytes_and_materializes_without_state_mirror(
    codec: str,
) -> None:
    states = _states()
    ema = _ema()

    checkpoint = PACKERS[codec](states, ema, layout=TINY_LAYOUT)

    assert checkpoint.codec == codec
    assert checkpoint.resident_bytes == TINY_LAYOUT.expected_resident_bytes
    assert checkpoint.evidence.resident_bytes == TINY_LAYOUT.expected_resident_bytes
    assert checkpoint.evidence.state_elements == TINY_LAYOUT.state_elements
    assert checkpoint.evidence.fp32_state_bytes == TINY_LAYOUT.fp32_state_bytes
    assert checkpoint.evidence.mean_squared_error >= 0
    assert checkpoint.evidence.relative_l2_error >= 0
    assert checkpoint.evidence.max_absolute_error >= 0
    assert checkpoint.query_energy_ema.data_ptr() != ema.data_ptr()
    assert torch.equal(checkpoint.query_energy_ema, ema)
    assert checkpoint.padding.dtype == torch.uint8
    assert not checkpoint.padding.any().item()

    persistent = checkpoint.persistent_tensors()
    assert [name for name, tensor in persistent if tensor.dtype == torch.float32] == [
        "query_energy_ema"
    ]
    assert all("replay" not in name and "buffer" not in name for name, _ in persistent)
    assert all(
        tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        for _, tensor in persistent
    )

    restored = checkpoint.materialize()
    assert set(restored) == set(TINY_LAYOUT.layer_indices)
    for layer_index, state in restored.items():
        assert tuple(state.shape) == (1, 2, 4, 8)
        assert state.dtype == torch.float32
        assert torch.isfinite(state).all().item()
        assert state.data_ptr() != states[layer_index].data_ptr()

    before = checkpoint.materialize()[0].clone()
    restored[0].fill_(123.0)
    assert torch.equal(checkpoint.materialize()[0], before)
    checkpoint.validate()


def test_physical_selected_unit_counts_match_each_exact_budget() -> None:
    states = _states()
    ema = _ema()

    expanded = pack_expanded_rht_q4_q8(states, ema, layout=TINY_LAYOUT)
    assert all(isinstance(layer, ExpandedRhtQ4Q8Layer) for layer in expanded.layers)
    assert (
        sum(
            layer.packed.high_precision_groups
            for layer in expanded.layers
            if isinstance(layer, ExpandedRhtQ4Q8Layer)
        )
        == TINY_LAYOUT.expanded_q8_promotions
    )

    multibit = pack_rht_q4_q6_q8(states, ema, layout=TINY_LAYOUT)
    assert all(isinstance(layer, RhtQ4Q6Q8Layer) for layer in multibit.layers)
    assert (
        sum(
            int(layer.packed.precision_codes().sum().item())
            for layer in multibit.layers
            if isinstance(layer, RhtQ4Q6Q8Layer)
        )
        == TINY_LAYOUT.multibit_marginal_steps
    )

    residual = pack_rht_residual_q4(states, ema, layout=TINY_LAYOUT)
    assert all(isinstance(layer, RhtResidualQ4Layer) for layer in residual.layers)
    assert (
        sum(
            int(
                baselines._unpack_bool_mask(
                    layer.lease_mask,
                    TINY_LAYOUT.rows_per_layer,
                )
                .sum()
                .item()
            )
            for layer in residual.layers
            if isinstance(layer, RhtResidualQ4Layer)
        )
        == TINY_LAYOUT.residual_q4_rows
    )


def test_global_ties_select_earlier_layer_head_row_indices() -> None:
    mask = baselines._stable_global_top_mask(torch.ones(16), 5)

    assert torch.equal(mask, torch.tensor([True] * 5 + [False] * 11))


@pytest.mark.parametrize("codec", tuple(PACKERS))
def test_selection_and_round_trip_are_deterministic(codec: str) -> None:
    first = PACKERS[codec](_states(), _ema(), layout=TINY_LAYOUT)
    second = PACKERS[codec](_states(), _ema(), layout=TINY_LAYOUT)

    assert first.evidence.selection_sha256 == second.evidence.selection_sha256
    assert first.evidence.evidence_dict() == second.evidence.evidence_dict()
    for layer_index in TINY_LAYOUT.layer_indices:
        assert torch.equal(
            first.materialize()[layer_index],
            second.materialize()[layer_index],
        )


def test_causal_query_ema_matches_frozen_one_token_recurrence() -> None:
    queries = _queries()

    actual = update_causal_query_ema(None, queries, layout=TINY_LAYOUT)

    prior = torch.full((2, 4), 0.25, dtype=torch.float32)
    for position, layer_index in enumerate(TINY_LAYOUT.layer_indices):
        query = queries[layer_index].to(torch.float32).squeeze(0).squeeze(0)
        energy = query.square() / (
            query.square().sum(dim=-1, keepdim=True) + baselines.QUERY_L2NORM_EPS
        )
        expected = baselines.QUERY_EMA_DECAY * prior + (1.0 - baselines.QUERY_EMA_DECAY) * energy
        assert torch.allclose(actual[position], expected, atol=1e-8, rtol=1e-6)

    next_actual = update_causal_query_ema(actual, _queries(seed=2342), layout=TINY_LAYOUT)
    assert not torch.equal(next_actual, actual)
    assert next_actual.data_ptr() != actual.data_ptr()


def test_cache_commit_is_atomic_when_candidate_packing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = EqualByteNoReplayCache(EXPANDED_RHT_Q4_Q8, layout=TINY_LAYOUT)
    first = cache.commit(_states(), _queries())
    previous_materialized = cache.materialize()
    previous_tensors = {name: tensor.clone() for name, tensor in first.persistent_tensors()}
    previous_evidence = cache.last_evidence

    def fail(
        states: dict[int, torch.Tensor],
        query_ema: torch.Tensor,
    ) -> EqualByteCheckpoint:
        del states, query_ema
        raise RuntimeError("injected complete-state packing failure")

    monkeypatch.setattr(cache, "_pack_candidate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cache.commit(_states(seed=999), _queries(seed=998))

    assert cache.checkpoint is first
    assert cache.update_count == 1
    assert cache.last_evidence is previous_evidence
    for name, tensor in first.persistent_tensors():
        assert torch.equal(tensor, previous_tensors[name])
    for layer_index in TINY_LAYOUT.layer_indices:
        assert torch.equal(cache.materialize()[layer_index], previous_materialized[layer_index])


def test_cache_reset_drops_the_only_persistent_checkpoint() -> None:
    cache = EqualByteNoReplayCache(RHT_RESIDUAL_Q4, layout=TINY_LAYOUT)
    cache.commit(_states(), _queries())

    cache.reset()

    assert cache.checkpoint is None
    assert cache.update_count == 0
    assert cache.last_evidence is None
    assert cache.resident_bytes() == 0
    with pytest.raises(RuntimeError, match="no committed checkpoint"):
        cache.materialize()


def test_packers_reject_malformed_state_and_query_ema() -> None:
    states = _states()
    states[0] = states[0].to(torch.bfloat16)
    with pytest.raises(TypeError, match="dtype torch.float32"):
        pack_expanded_rht_q4_q8(states, _ema(), layout=TINY_LAYOUT)

    states = _states()
    bad_ema = _ema()
    bad_ema[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pack_expanded_rht_q4_q8(states, bad_ema, layout=TINY_LAYOUT)

    bad_queries = _queries()
    bad_queries[3] = torch.empty((1, 0, 2, 4))
    with pytest.raises(ValueError, match="positive"):
        update_causal_query_ema(None, bad_queries, layout=TINY_LAYOUT)


def test_persistent_alias_validation_fails_closed() -> None:
    tensor = torch.zeros(4, dtype=torch.uint8)

    with pytest.raises(ValueError, match="aliases"):
        baselines._validate_owned_tensors((("first", tensor), ("second", tensor)))


def test_expanded_checkpoint_rejects_reserved_q4_code_after_corruption() -> None:
    checkpoint = pack_expanded_rht_q4_q8(_states(), _ema(), layout=TINY_LAYOUT)
    first = checkpoint.layers[0]
    assert isinstance(first, ExpandedRhtQ4Q8Layer)
    payload = first.packed.low_payload.clone()
    assert payload.numel()
    payload[0, 0] = 0x08
    malformed_packed = replace(first.packed, low_payload=payload)
    malformed_layer = replace(first, packed=malformed_packed)

    with pytest.raises(ValueError, match="reserved"):
        replace(checkpoint, layers=(malformed_layer, *checkpoint.layers[1:]))


def test_multibit_checkpoint_rejects_reserved_precision_code_after_corruption() -> None:
    checkpoint = pack_rht_q4_q6_q8(_states(), _ema(), layout=TINY_LAYOUT)
    first = checkpoint.layers[0]
    assert isinstance(first, RhtQ4Q6Q8Layer)
    codes = first.packed.packed_precision_codes.clone()
    codes[0] = torch.bitwise_or(codes[0], torch.tensor(3, dtype=torch.uint8))

    with pytest.raises(ValueError, match="reserved precision code 3"):
        replace(first.packed, packed_precision_codes=codes)


def test_residual_checkpoint_rejects_mask_count_drift() -> None:
    checkpoint = pack_rht_residual_q4(_states(), _ema(), layout=TINY_LAYOUT)
    first = checkpoint.layers[0]
    assert isinstance(first, RhtResidualQ4Layer)
    mask = baselines._unpack_bool_mask(
        first.lease_mask,
        TINY_LAYOUT.rows_per_layer,
    )
    mask[0] = ~mask[0]
    malformed_layer = replace(first, lease_mask=baselines._pack_bool_mask(mask))

    with pytest.raises(ValueError, match="metadata does not match"):
        replace(checkpoint, layers=(malformed_layer, *checkpoint.layers[1:]))


def test_checkpoint_transfer_preserves_physical_evidence() -> None:
    checkpoint = pack_expanded_rht_q4_q8(_states(), _ema(), layout=TINY_LAYOUT)

    moved = checkpoint.to("cpu")

    assert moved.evidence == checkpoint.evidence
    assert moved.resident_bytes == checkpoint.resident_bytes
    assert moved.evidence.selection_sha256 == checkpoint.evidence.selection_sha256
    for layer_index in TINY_LAYOUT.layer_indices:
        assert torch.equal(
            moved.materialize()[layer_index],
            checkpoint.materialize()[layer_index],
        )
