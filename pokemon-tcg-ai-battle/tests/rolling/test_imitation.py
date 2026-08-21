from __future__ import annotations

import pytest

from rolling_policy.imitation import (
    ImitationDecision,
    balanced_option_weights,
    calibrate_semantic_threshold,
    confident_consensus_metrics,
    consensus_semantic_metrics,
    semantic_accuracy,
    semantic_set_accuracy,
    semantic_set_prediction,
)


def test_option_weights_balance_positive_and_negative_within_decision() -> None:
    weights = balanced_option_weights([1, 0, 0, 0], decision_weight=2.0)
    assert weights == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])
    assert sum(weights) == pytest.approx(2.0)


def test_semantic_accuracy_accepts_equivalent_selected_signature() -> None:
    decision = ImitationDecision(
        decision_id="episode:0:1",
        option_signatures=((1,), (1,), (2,)),
        selected_signatures=frozenset({(1,)}),
    )
    assert semantic_accuracy([decision], [0.8, 0.7, 0.1]) == 1.0


def test_consensus_metrics_require_same_unique_semantic_top() -> None:
    decisions = [
        ImitationDecision("a", ((1,), (2,)), frozenset({(2,)})),
        ImitationDecision("b", ((3,), (4,)), frozenset({(3,)})),
    ]
    metrics = consensus_semantic_metrics(
        decisions,
        [0.1, 0.9, 0.8, 0.2],
        [0.2, 0.8, 0.1, 0.9],
    )
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["covered_accuracy"] == pytest.approx(1.0)
    assert metrics["covered_correct"] == 1


def test_confident_consensus_selects_score_margin_without_using_labels() -> None:
    decisions = [
        ImitationDecision(str(index), ((1,), (2,)), frozenset({(1,)}))
        for index in range(4)
    ]
    first = [0.9, 0.1, 0.8, 0.2, 0.6, 0.4, 0.55, 0.45]
    second = [0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.54, 0.46]
    metrics = confident_consensus_metrics(
        decisions,
        first,
        second,
        target_coverage=0.5,
    )
    assert metrics["covered"] == 2
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["covered_accuracy"] == pytest.approx(1.0)
    assert metrics["minimum_margin"] == pytest.approx(0.5)


def test_semantic_set_prediction_selects_all_scores_above_threshold() -> None:
    decision = ImitationDecision(
        "multi",
        ((1,), (2,), (3,)),
        frozenset({(1,), (2,)}),
    )
    assert semantic_set_prediction(
        decision,
        [0.9, 0.8, 0.2],
        threshold=0.5,
        minimum=1,
        maximum=2,
    ) == frozenset({(1,), (2,)})


def test_semantic_set_prediction_clips_to_minimum_and_unique_options() -> None:
    decision = ImitationDecision(
        "duplicates",
        ((1,), (1,), (2,)),
        frozenset({(1,), (2,)}),
    )
    assert semantic_set_prediction(
        decision,
        [0.2, 0.7, 0.6],
        threshold=0.8,
        minimum=2,
        maximum=2,
    ) == frozenset({(1,), (2,)})


def test_semantic_set_prediction_supports_legal_empty_selection() -> None:
    decision = ImitationDecision(
        "optional",
        ((1,), (2,)),
        frozenset(),
    )
    assert semantic_set_prediction(
        decision,
        [0.2, 0.1],
        threshold=0.5,
        minimum=0,
        maximum=2,
    ) == frozenset()


def test_semantic_threshold_calibration_maximizes_exact_set_accuracy() -> None:
    decisions = [
        ImitationDecision("a", ((1,), (2,)), frozenset({(1,)})),
        ImitationDecision("b", ((3,), (4,)), frozenset({(3,), (4,)})),
    ]
    scores = [0.9, 0.4, 0.8, 0.7]
    bounds = [(0, 2), (0, 2)]
    threshold = calibrate_semantic_threshold(decisions, scores, bounds)
    assert threshold == pytest.approx(0.7)
    assert semantic_set_accuracy(
        decisions,
        scores,
        bounds,
        threshold=threshold,
    ) == 1.0
