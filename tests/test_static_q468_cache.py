from __future__ import annotations

import copy
from collections.abc import Iterator

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.statelease_equal_byte_baselines import (
    FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    RHT_Q4_Q6_Q8,
    EqualByteLayout,
)
from recurquant.statelease_equal_byte_cache import (
    EqualByteLinearAttentionLayer,
    Qwen35EqualByteObserver,
)
from recurquant.static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q48_PROMOTIONS,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    StaticRhtQ48Policy,
    StaticRhtQ468Geometry,
    StaticRhtQ468Policy,
    build_static_rht_q48_policy,
    build_static_rht_q468_policy,
)
from recurquant.static_q468_cache import (
    DYNAMIC_Q468_ORACLE_METHOD,
    StaticRhtQwen35Cache,
    create_qwen35_dynamic_q468_oracle_cache,
    create_qwen35_static_rht_cache,
)
from tests.test_transformers_cache import tiny_config

MANIFEST_SHA256 = "12" * 32
IDENTITY_SHA256 = "34" * 32
TOKENIZER_MANIFEST_SHA256 = "56" * 32
SOURCE_COMMIT = "78" * 20
BINDINGS = {
    "calibration_manifest_sha256": MANIFEST_SHA256,
    "identity_artifact_sha256": IDENTITY_SHA256,
    "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
    "source_commit": SOURCE_COMMIT,
}

# This layout is valid for every inherited equal-byte layer implementation and
# structurally matches tests.test_transformers_cache.tiny_config().  The static
# policies intentionally remain below its 180-byte target.
TINY_LAYOUT = EqualByteLayout(
    layer_indices=(0,),
    heads=2,
    key_rows=8,
    value_width=8,
    expanded_q8_promotions=4,
    multibit_marginal_steps=8,
    residual_q4_rows=3,
    expanded_padding_bytes=2,
    multibit_padding_bytes=0,
    residual_padding_bytes=0,
    expected_resident_bytes=180,
)
TINY_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=(0,),
    heads=2,
    key_rows=8,
    value_width=8,
    target_resident_bytes=TINY_LAYOUT.expected_resident_bytes,
)


def _tiny_distortions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = TINY_GEOMETRY.total_rows
    order = torch.arange(rows, dtype=torch.float64).reshape(1, 2, 8)
    return 5.0 + order / 17.0, 2.0 + order.flip(-1) / 19.0, 0.5 + order / 23.0


def _tiny_q468_policy() -> StaticRhtQ468Policy:
    return build_static_rht_q468_policy(
        *_tiny_distortions(),
        geometry=TINY_GEOMETRY,
        marginal_steps=8,
        **BINDINGS,
    )


def _tiny_q48_policy() -> StaticRhtQ48Policy:
    d4, _, d8 = _tiny_distortions()
    return build_static_rht_q48_policy(
        d4,
        d8,
        geometry=TINY_GEOMETRY,
        promoted_rows=4,
        **BINDINGS,
    )


def _cache(policy: StaticRhtQ468Policy | StaticRhtQ48Policy) -> StaticRhtQwen35Cache:
    return StaticRhtQwen35Cache(
        tiny_config(),
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )


def _model() -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()


def _checkpoint_tensors(
    cache: StaticRhtQwen35Cache,
) -> tuple[tuple[str, torch.Tensor], ...]:
    assert cache.checkpoint is not None
    return tuple(
        (name, tensor.detach().clone()) for name, tensor in cache.checkpoint.persistent_tensors()
    )


def _assert_checkpoint_equal(left: StaticRhtQwen35Cache, right: StaticRhtQwen35Cache) -> None:
    actual = _checkpoint_tensors(left)
    expected = _checkpoint_tensors(right)
    assert [name for name, _tensor in actual] == [name for name, _tensor in expected]
    for (_, actual_tensor), (_, expected_tensor) in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _mutable_nonrecurrent_tensors(cache: StaticRhtQwen35Cache) -> Iterator[torch.Tensor]:
    for layer in cache.layers:
        attributes = getattr(layer, "__dict__", {})
        conv_states = attributes.get("conv_states", {})
        if isinstance(conv_states, dict):
            for value in conv_states.values():
                if isinstance(value, torch.Tensor):
                    yield value
        for name in ("keys", "values"):
            value = attributes.get(name)
            if isinstance(value, torch.Tensor):
                yield value


@pytest.mark.parametrize("policy_factory", [_tiny_q468_policy, _tiny_q48_policy])
def test_static_runtime_runs_tiny_qwen_prefill_and_decode_without_policy_duplicate(
    policy_factory,
) -> None:
    torch.manual_seed(1301)
    model = _model()
    policy = policy_factory()
    cache = StaticRhtQwen35Cache(
        model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )

    policy_attributes = {
        name: value for name, value in cache.__dict__.items() if "policy" in name.lower()
    }
    assert policy_attributes
    assert isinstance(cache._serialized_policy, bytes)
    assert all(not isinstance(value, torch.Tensor) for value in policy_attributes.values())
    assert all(
        not isinstance(value, (StaticRhtQ468Policy, StaticRhtQ48Policy))
        for value in policy_attributes.values()
    )
    assert cache.persistent_recurrent_tensors() == ()

    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    next_token = torch.randint(0, model.config.vocab_size, (1, 1))
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        prefill = model(prompt, past_key_values=cache, use_cache=True)
        decode = model(next_token, past_key_values=cache, use_cache=True)

    assert prefill.logits.shape == (1, 4, model.config.vocab_size)
    assert decode.logits.shape == (1, 1, model.config.vocab_size)
    assert cache.update_count == 2
    assert cache.successful_tokens == 5
    assert len(cache.update_evidence) == 2
    assert cache.last_evidence is cache.update_evidence[-1]
    assert cache.checkpoint is not None
    assert cache.checkpoint.policy.policy_sha256 == policy.policy_sha256
    assert cache.checkpoint.policy is not policy
    assert cache.persistent_raw_state_bytes() == 0
    persistent = cache.checkpoint.persistent_tensors()
    assert persistent
    assert all(tensor.dtype not in (torch.float32, torch.float64) for _, tensor in persistent)
    nonempty = [tensor for _, tensor in persistent if tensor.numel()]
    assert len({tensor.untyped_storage().data_ptr() for tensor in nonempty}) == len(nonempty)
    assert sum(tensor.untyped_storage().nbytes() for _, tensor in persistent) == (
        cache.checkpoint.resident_bytes
    )

    evidence = cache.last_evidence
    assert evidence is not None
    assert evidence.workspace_measurement_scope == "cache_retained_forward_tensors_only"
    assert evidence.cuda_allocator_peak_measured is False
    assert evidence.cuda_allocator_peak_bytes is None
    assert evidence.resident_tensor_storage_bytes == cache.checkpoint.resident_bytes
    summary = cache.storage_summary()
    assert summary["policy_tensor_storage_bytes_outside_checkpoint"] == 0
    assert summary["resident_tensor_storage_bytes"] == cache.checkpoint.resident_bytes
    assert summary["cuda_allocator_peak_measured"] is False
    assert summary["cuda_allocator_peak_bytes"] is None


def test_static_runtime_keeps_k27030_style_under_budget_without_padding() -> None:
    policy = _tiny_q468_policy()
    cache = _cache(policy)
    model = _model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 3))
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        model(prompt, past_key_values=cache, use_cache=True)

    assert cache.checkpoint is not None
    ledger = cache.checkpoint.ledger
    assert ledger.alignment_bytes == 0
    assert ledger.resident_bytes < ledger.target_resident_bytes
    assert ledger.budget_delta_bytes == ledger.target_resident_bytes - ledger.resident_bytes
    assert ledger.exact_budget_eligible is False
    assert cache.checkpoint.state.padding.numel() == 0
    summary = cache.storage_summary()
    assert summary["expected_resident_bytes"] == ledger.resident_bytes
    assert summary["target_resident_bytes"] == ledger.target_resident_bytes
    assert summary["budget_delta_bytes"] == ledger.budget_delta_bytes


def test_static_runtime_reauthenticates_mutable_policy_storage_before_decode() -> None:
    model = _model()
    policy = _tiny_q468_policy()
    cache = _cache(policy)
    prompt = torch.randint(0, model.config.vocab_size, (1, 3))
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        model(prompt, past_key_values=cache, use_cache=True)

    assert cache.checkpoint is not None
    codes = cache.checkpoint.policy.packed_precision_codes
    with torch.inference_mode():
        codes[0] = int(codes[0].item()) ^ 0x01

    with pytest.raises(ValueError):
        cache.begin_statelease_forward_transaction()
    assert cache._active_equal_byte_transaction is None


def test_lm_head_failure_rolls_back_static_checkpoint_and_retry_matches_clean() -> None:
    torch.manual_seed(1302)
    model = _model()
    clean_model = copy.deepcopy(model).eval()
    policy = _tiny_q468_policy()
    cache = StaticRhtQwen35Cache(
        model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )
    clean_cache = StaticRhtQwen35Cache(
        clean_model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 5))
    next_token = torch.randint(0, model.config.vocab_size, (1, 1))

    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            model(prompt, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean_model(prompt, past_key_values=clean_cache, use_cache=True)

    checkpoint_before = cache.checkpoint
    evidence_before = tuple(cache.update_evidence)
    count_before = cache.update_count
    nonrecurrent_before = tuple(
        tensor.detach().clone() for tensor in _mutable_nonrecurrent_tensors(cache)
    )

    def fail_after_lm_head(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs, output
        raise RuntimeError("injected static later-layer failure")

    handle = model.lm_head.register_forward_hook(fail_after_lm_head)
    try:
        with (
            torch.inference_mode(),
            Qwen35EqualByteObserver(model, caches=[cache]),
            pytest.raises(RuntimeError, match="injected static later-layer failure"),
        ):
            model(next_token, past_key_values=cache, use_cache=True)
    finally:
        handle.remove()

    assert cache.checkpoint is checkpoint_before
    assert tuple(cache.update_evidence) == evidence_before
    assert cache.update_count == count_before
    assert not cache._pending_observations
    assert not cache._previous_states
    assert not cache._final_states
    assert not cache._queries
    nonrecurrent_after = tuple(_mutable_nonrecurrent_tensors(cache))
    assert len(nonrecurrent_after) == len(nonrecurrent_before)
    for actual, expected in zip(nonrecurrent_after, nonrecurrent_before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            retry = model(next_token, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean = clean_model(next_token, past_key_values=clean_cache, use_cache=True)
    torch.testing.assert_close(retry.logits, clean.logits, rtol=0, atol=0)
    _assert_checkpoint_equal(cache, clean_cache)


def test_static_pack_failure_never_replaces_previous_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1303)
    model = _model()
    policy = _tiny_q48_policy()
    cache = StaticRhtQwen35Cache(
        model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    next_token = torch.randint(0, model.config.vocab_size, (1, 1))
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        model(prompt, past_key_values=cache, use_cache=True)

    checkpoint_before = cache.checkpoint
    evidence_before = tuple(cache.update_evidence)
    count_before = cache.update_count

    def fail_pack(states: dict[int, torch.Tensor]):
        del states
        raise RuntimeError("injected static global pack failure")

    monkeypatch.setattr(cache, "_pack_static_candidate", fail_pack)
    with (
        torch.inference_mode(),
        Qwen35EqualByteObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="injected static global pack failure"),
    ):
        model(next_token, past_key_values=cache, use_cache=True)

    assert cache.checkpoint is checkpoint_before
    assert tuple(cache.update_evidence) == evidence_before
    assert cache.update_count == count_before
    assert not cache._pending_observations
    assert not cache._previous_states
    assert not cache._final_states
    assert not cache._queries


def test_incomplete_static_transaction_and_direct_write_fail_closed() -> None:
    cache = _cache(_tiny_q468_policy())
    state = torch.randn(1, 2, 8, 8)
    with pytest.raises(RuntimeError, match="immutable-policy packer"):
        cache._pack_candidate({}, torch.empty(0))
    with pytest.raises(RuntimeError, match="active root-model"):
        cache.update_recurrent_state(state, 0)

    transaction = cache.begin_statelease_forward_transaction()
    with pytest.raises(RuntimeError, match="exactly one recurrent receipt"):
        cache.commit_statelease_forward_transaction(transaction)
    cache.rollback_statelease_forward_transaction(transaction)
    assert cache.checkpoint is None


def test_static_runtime_lifecycle_offload_prefetch_and_reset() -> None:
    model = _model()
    policy = _tiny_q48_policy()
    cache = StaticRhtQwen35Cache(
        model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 3))
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        model(prompt, past_key_values=cache, use_cache=True)

    cache.reorder_cache(torch.tensor([0], dtype=torch.long))
    with pytest.raises(ValueError, match="batch-one"):
        cache.reorder_cache(torch.tensor([0, 0], dtype=torch.long))
    with pytest.raises(RuntimeError, match="speculative past recording"):
        cache.activate_past_recording()
    with pytest.raises(RuntimeError, match="cannot crop"):
        cache.crop(-1)

    cache.offload_all()
    assert cache.checkpoint is not None
    assert all(tensor.device.type == "cpu" for _, tensor in cache.checkpoint.persistent_tensors())
    cache.prefetch_all()
    assert cache.checkpoint is not None
    assert cache.checkpoint.policy.policy_sha256 == policy.policy_sha256

    cache.reset()
    assert cache.checkpoint is None
    assert cache.update_count == 0
    assert not cache.update_evidence
    layer = cache.layers[0]
    assert isinstance(layer, EqualByteLinearAttentionLayer)
    assert not layer.has_previous_state[0]


def test_static_cache_rejects_policy_hash_geometry_and_public_method_drift() -> None:
    policy = _tiny_q468_policy()
    with pytest.raises(ValueError, match="policy SHA-256"):
        StaticRhtQwen35Cache(
            tiny_config(),
            policy=policy,
            expected_policy_sha256="ab" * 32,
            layout=TINY_LAYOUT,
        )

    mismatched_layout = EqualByteLayout(
        layer_indices=(0,),
        heads=1,
        key_rows=16,
        value_width=8,
        expanded_q8_promotions=4,
        multibit_marginal_steps=8,
        residual_q4_rows=3,
        expanded_padding_bytes=2,
        multibit_padding_bytes=0,
        residual_padding_bytes=0,
        expected_resident_bytes=180,
    )
    with pytest.raises(ValueError, match="geometry"):
        StaticRhtQwen35Cache(
            tiny_config(),
            policy=policy,
            expected_policy_sha256=policy.policy_sha256,
            layout=mismatched_layout,
        )

    with pytest.raises(ValueError, match="three frozen methods"):
        create_qwen35_static_rht_cache(
            tiny_config(),
            policy=policy,
            expected_policy_sha256=policy.policy_sha256,
        )


def _frozen_config():
    layer_types = [
        (
            "linear_attention"
            if index in FROZEN_QWEN35_STATIC_Q468_GEOMETRY.layer_indices
            else "full_attention"
        )
        for index in range(23)
    ]
    config = tiny_config(layer_types)
    config.linear_num_value_heads = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.heads
    config.linear_key_head_dim = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.key_rows
    config.linear_value_head_dim = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.value_width
    return config


@pytest.fixture(scope="module")
def frozen_policies() -> tuple[StaticRhtQ468Policy, StaticRhtQ468Policy, StaticRhtQ48Policy]:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    order = torch.arange(rows, dtype=torch.float64).reshape(1, rows)
    d4 = 5.0 + order / (rows + 1)
    d6 = 2.0 + order.flip(-1) / (rows + 3)
    d8 = 0.5 + order / (rows + 5)
    primary = build_static_rht_q468_policy(
        d4,
        d6,
        d8,
        geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        method_id=STATIC_Q468_PRIMARY_METHOD,
        **BINDINGS,
    )
    ablation = build_static_rht_q468_policy(
        d4,
        d6,
        d8,
        geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        marginal_steps=FROZEN_STATIC_Q468_ABLATION_STEPS,
        method_id=STATIC_Q468_ABLATION_METHOD,
        **BINDINGS,
    )
    q48 = build_static_rht_q48_policy(
        d4,
        d8,
        geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        promoted_rows=FROZEN_STATIC_Q48_PROMOTIONS,
        method_id=STATIC_Q48_COMPARATOR_METHOD,
        **BINDINGS,
    )
    return primary, ablation, q48


def test_public_factory_supports_all_reserved_methods_without_loading_weights(
    frozen_policies,
) -> None:
    primary, ablation, q48 = frozen_policies
    expected = {
        STATIC_Q468_PRIMARY_METHOD: (3_454_664, 0, True),
        STATIC_Q468_ABLATION_METHOD: (3_380_928, 73_736, False),
        STATIC_Q48_COMPARATOR_METHOD: (3_454_664, 0, True),
    }
    for policy in (primary, ablation, q48):
        cache = create_qwen35_static_rht_cache(
            _frozen_config(),
            policy=policy,
            expected_policy_sha256=policy.policy_sha256,
        )
        summary = cache.storage_summary()
        resident, delta, exact = expected[policy.method_id]
        assert cache.method_id == policy.method_id
        assert cache.checkpoint is None
        assert summary["expected_resident_bytes"] == resident
        assert summary["target_resident_bytes"] == 3_454_664
        assert summary["budget_delta_bytes"] == delta
        assert summary["exact_budget_eligible"] is exact
        assert summary["policy_tensor_storage_bytes_outside_checkpoint"] == 0
        assert all(
            not isinstance(value, (torch.Tensor, StaticRhtQ468Policy, StaticRhtQ48Policy))
            for name, value in cache.__dict__.items()
            if "policy" in name.lower()
        )


def test_named_dynamic_q468_oracle_is_existing_exact_k27030_path() -> None:
    cache = create_qwen35_dynamic_q468_oracle_cache(_frozen_config(), record_evidence=True)
    assert cache.codec == RHT_Q4_Q6_Q8
    assert cache.method_id == DYNAMIC_Q468_ORACLE_METHOD
    assert cache.layout is FROZEN_QWEN35_EQUAL_BYTE_LAYOUT
    assert cache.layout.multibit_marginal_steps == 27_030
    assert cache.storage_summary()["expected_resident_bytes"] == 3_454_664


def test_public_factories_validate_model_runtime_before_cache_construction(
    frozen_policies,
) -> None:
    primary = frozen_policies[0]
    training_model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    )
    with pytest.raises(ValueError, match="inference-only"):
        create_qwen35_static_rht_cache(
            training_model,
            policy=primary,
            expected_policy_sha256=primary.policy_sha256,
        )
    with pytest.raises(ValueError, match="inference-only"):
        create_qwen35_dynamic_q468_oracle_cache(training_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_static_runtime_tiny_qwen_cuda_smoke() -> None:
    torch.manual_seed(1304)
    model = _model().to("cuda")
    policy = _tiny_q468_policy()
    cache = StaticRhtQwen35Cache(
        model.config,
        policy=policy,
        expected_policy_sha256=policy.policy_sha256,
        layout=TINY_LAYOUT,
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 3), device="cuda")
    next_token = torch.randint(0, model.config.vocab_size, (1, 1), device="cuda")
    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        model(prompt, past_key_values=cache, use_cache=True)
        output = model(next_token, past_key_values=cache, use_cache=True)

    assert output.logits.device.type == "cuda"
    assert cache.checkpoint is not None
    assert all(tensor.device.type == "cuda" for _, tensor in cache.checkpoint.persistent_tensors())
    assert cache.persistent_raw_state_bytes() == 0
