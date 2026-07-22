from __future__ import annotations

import pytest
import torch

from recurquant.row_policy import RowLocation, select_rows_exact_budget


def test_qwen35_equal_byte_row_plan_accounts_for_mask_and_promotions() -> None:
    scores = {
        layer_index: torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128)
        + layer_index * 10_000
        for layer_index in range(18)
    }

    plan = select_rows_exact_budget(scores, target_resident_bytes=2_564_096)

    assert plan.total_groups == 36_864
    assert plan.mask_bytes == 4_608
    assert plan.promotion_increment_bytes == 64
    assert plan.promoted_group_count == 1_976
    assert plan.resident_bytes == 2_564_096
    assert plan.target_resident_bytes == 2_564_096
    assert plan.high_precision_rows[-1] == RowLocation(17, 15, 127)


def test_row_selection_breaks_score_ties_by_location() -> None:
    scores = {4: torch.ones((1, 4)), 1: torch.ones((1, 4))}
    minimum_with_mask = 8 * 66 + 1

    plan = select_rows_exact_budget(
        scores,
        target_resident_bytes=minimum_with_mask + 2 * 64,
    )

    assert plan.high_precision_rows == (
        RowLocation(1, 0, 0),
        RowLocation(1, 0, 1),
    )
    assert plan.groups_for_layer(1) == (0, 1)
    assert plan.groups_for_layer(4) == ()


def test_nonrepresentable_target_is_rejected() -> None:
    scores = {0: torch.ones((1, 2))}
    minimum_with_mask = 2 * 66 + 1

    with pytest.raises(ValueError, match="nearest valid targets"):
        select_rows_exact_budget(scores, target_resident_bytes=minimum_with_mask + 1)


def test_payload_rounding_must_leave_a_positive_promotion_increment() -> None:
    scores = {0: torch.ones((1, 1))}

    with pytest.raises(ValueError, match="positive promotion byte increment"):
        select_rows_exact_budget(
            scores,
            target_resident_bytes=4,
            low_bits=4,
            high_bits=8,
            group_size=1,
        )
