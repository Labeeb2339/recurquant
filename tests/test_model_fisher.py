from __future__ import annotations

import pytest
import torch

from recurquant.model_fisher import (
    row_block_model_fisher_risk,
    sample_model_pseudo_labels,
)


def _manual_endpoint_risk(
    gradients: torch.Tensor,
    errors: torch.Tensor,
) -> torch.Tensor:
    probes, batch_size, heads, key_dim, _ = gradients.shape
    result = torch.zeros((heads, key_dim), dtype=gradients.dtype)
    for probe in range(probes):
        for batch in range(batch_size):
            for head in range(heads):
                for row in range(key_dim):
                    projection = torch.dot(
                        gradients[probe, batch, head, row],
                        errors[batch, head, row],
                    )
                    result[head, row] += 0.5 * projection.square()
    return result / (probes * batch_size)


def test_row_risk_matches_manual_probe_and_batch_average() -> None:
    gradients = torch.tensor(
        [
            [
                [[[1.0, 2.0], [-1.0, 0.5]]],
                [[[0.0, 3.0], [2.0, -2.0]]],
            ],
            [
                [[[2.0, -1.0], [0.25, 4.0]]],
                [[[-3.0, 1.0], [1.5, 0.5]]],
            ],
        ],
        dtype=torch.float64,
    )
    int4_errors = torch.tensor(
        [
            [[[0.4, -0.2], [0.5, 0.25]]],
            [[[0.1, 0.3], [-0.2, 0.6]]],
        ],
        dtype=torch.float64,
    )
    int8_errors = torch.tensor(
        [
            [[[0.05, -0.1], [0.2, 0.1]]],
            [[[0.15, 0.05], [-0.1, 0.2]]],
        ],
        dtype=torch.float64,
    )

    result = row_block_model_fisher_risk(gradients, int4_errors, int8_errors)
    expected4 = _manual_endpoint_risk(gradients, int4_errors)
    expected8 = _manual_endpoint_risk(gradients, int8_errors)

    assert result.probes == 2
    assert result.batch_size == 2
    assert result.shape == torch.Size((1, 2))
    assert torch.allclose(result.int4_risk, expected4)
    assert torch.allclose(result.int8_risk, expected8)
    assert torch.allclose(result.risk_difference, expected4 - expected8)


def test_row_dot_is_squared_after_sum_and_negative_difference_is_preserved() -> None:
    gradients = torch.tensor([[[[[1.0, 1.0]]]]])
    # Within-row cancellation makes the INT4 projection exactly zero.  A
    # coordinate-diagonal approximation would incorrectly assign positive risk.
    int4_errors = torch.tensor([[[[1.0, -1.0]]]])
    int8_errors = torch.tensor([[[[1.0, 0.0]]]])

    result = row_block_model_fisher_risk(gradients, int4_errors, int8_errors)

    assert result.int4_risk.item() == 0.0
    assert result.int8_risk.item() == pytest.approx(0.5)
    assert result.risk_difference.item() == pytest.approx(-0.5)


def test_positive_difference_means_int8_has_lower_endpoint_risk() -> None:
    gradients = torch.tensor([[[[[1.0, 0.0]]]]])
    int4_errors = torch.tensor([[[[2.0, 0.0]]]])
    int8_errors = torch.tensor([[[[1.0, 0.0]]]])

    result = row_block_model_fisher_risk(gradients, int4_errors, int8_errors)

    assert result.int4_risk.item() == pytest.approx(2.0)
    assert result.int8_risk.item() == pytest.approx(0.5)
    assert result.risk_difference.item() == pytest.approx(1.5)


def test_scoring_is_deterministic_detached_and_does_not_mutate_inputs() -> None:
    generator = torch.Generator().manual_seed(2339)
    gradients = torch.randn((3, 2, 2, 4, 5), generator=generator, requires_grad=True)
    int4_errors = torch.randn((2, 2, 4, 5), generator=generator, requires_grad=True)
    int8_errors = torch.randn((2, 2, 4, 5), generator=generator, requires_grad=True)
    before = tuple(tensor.detach().clone() for tensor in (gradients, int4_errors, int8_errors))

    first = row_block_model_fisher_risk(gradients, int4_errors, int8_errors)
    second = row_block_model_fisher_risk(gradients, int4_errors, int8_errors)

    assert torch.equal(first.int4_risk, second.int4_risk)
    assert torch.equal(first.int8_risk, second.int8_risk)
    assert torch.equal(first.risk_difference, second.risk_difference)
    assert not first.int4_risk.requires_grad
    assert not first.int8_risk.requires_grad
    assert not first.risk_difference.requires_grad
    assert all(tensor.grad is None for tensor in (gradients, int4_errors, int8_errors))
    assert all(
        torch.equal(current.detach(), original)
        for current, original in zip(
            (gradients, int4_errors, int8_errors),
            before,
            strict=True,
        )
    )


def test_model_pseudo_label_sampling_is_explicit_and_reproducible() -> None:
    logits = torch.tensor([[0.2, -0.4, 1.1], [1.3, 0.7, -0.2]])
    first_generator = torch.Generator().manual_seed(72)
    second_generator = torch.Generator().manual_seed(72)

    first = sample_model_pseudo_labels(
        logits,
        probes=64,
        generator=first_generator,
    )
    second = sample_model_pseudo_labels(
        logits,
        probes=64,
        generator=second_generator,
    )

    assert first.shape == (64, 2)
    assert first.dtype == torch.int64
    assert torch.equal(first, second)
    assert first.min().item() >= 0
    assert first.max().item() < logits.shape[1]
    assert not first.requires_grad


def test_model_pseudo_label_sampling_uses_unmodified_softmax_distribution() -> None:
    logits = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]])
    labels = sample_model_pseudo_labels(
        logits,
        probes=16,
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.equal(labels[:, 0], torch.ones(16, dtype=torch.int64))
    assert torch.equal(labels[:, 1], torch.zeros(16, dtype=torch.int64))


def test_row_risk_rejects_invalid_shapes_dtypes_and_empty_dimensions() -> None:
    gradients = torch.zeros((2, 1, 1, 2, 3))
    errors = torch.zeros((1, 1, 2, 3))

    with pytest.raises(ValueError, match="pseudo_label_score_gradients"):
        row_block_model_fisher_risk(gradients[0], errors, errors)
    with pytest.raises(ValueError, match="int4_errors and int8_errors"):
        row_block_model_fisher_risk(gradients, errors[0], errors)
    with pytest.raises(ValueError, match="gradient dimensions"):
        row_block_model_fisher_risk(gradients[:, :, :, :, :2], errors, errors)
    with pytest.raises(ValueError, match="same shape"):
        row_block_model_fisher_risk(gradients, errors, errors[..., :2])
    with pytest.raises(ValueError, match="non-empty"):
        row_block_model_fisher_risk(gradients[:0], errors, errors)
    with pytest.raises(TypeError, match="floating-point"):
        row_block_model_fisher_risk(gradients.to(torch.int64), errors, errors)
    with pytest.raises(TypeError, match="same dtype"):
        row_block_model_fisher_risk(gradients, errors, errors.to(torch.float64))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_row_risk_rejects_nonfinite_inputs(bad_value: float) -> None:
    gradients = torch.ones((1, 1, 1, 1, 2))
    errors = torch.ones((1, 1, 1, 2))
    gradients[0, 0, 0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        row_block_model_fisher_risk(gradients, errors, errors)


def test_row_risk_rejects_nonfinite_computation() -> None:
    gradients = torch.full((1, 1, 1, 1, 2), 1e30)
    errors = torch.full((1, 1, 1, 2), 1e30)

    with pytest.raises(RuntimeError, match="became non-finite"):
        row_block_model_fisher_risk(gradients, errors, errors)


def test_pseudo_label_sampling_validation() -> None:
    generator = torch.Generator().manual_seed(1)
    logits = torch.ones((2, 3))

    with pytest.raises(ValueError, match=r"\[batch, vocabulary\]"):
        sample_model_pseudo_labels(logits[0], probes=2, generator=generator)
    with pytest.raises(ValueError, match="at least one"):
        sample_model_pseudo_labels(logits[:, :0], probes=2, generator=generator)
    with pytest.raises(ValueError, match="positive integer"):
        sample_model_pseudo_labels(logits, probes=0, generator=generator)
    with pytest.raises(ValueError, match="positive integer"):
        sample_model_pseudo_labels(logits, probes=True, generator=generator)
    with pytest.raises(TypeError, match="torch.Generator"):
        sample_model_pseudo_labels(logits, probes=2, generator=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        sample_model_pseudo_labels(logits.to(torch.int64), probes=2, generator=generator)
    nonfinite = logits.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        sample_model_pseudo_labels(nonfinite, probes=2, generator=generator)
