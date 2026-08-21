from __future__ import annotations

import pytest

from rolling_policy.metrics import equal_frequency_ece, roc_auc


def test_auc_matches_hand_checked_pair_ordering() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_equal_frequency_ece_matches_hand_checked_bins() -> None:
    assert equal_frequency_ece(
        [0, 1, 0, 1],
        [0.1, 0.35, 0.4, 0.8],
        bins=2,
    ) == pytest.approx(0.1875)


@pytest.mark.parametrize("labels", ([1, 1, 1], [0, 0, 0]))
def test_metrics_fail_without_both_outcome_classes(labels) -> None:
    with pytest.raises(ValueError, match="both outcome classes"):
        roc_auc(labels, [0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="both outcome classes"):
        equal_frequency_ece(labels, [0.2, 0.3, 0.4], bins=10)


def test_metrics_reject_nonfinite_or_out_of_range_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        roc_auc([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError, match="probabilities"):
        equal_frequency_ece([0, 1], [-0.1, 0.8], bins=2)
