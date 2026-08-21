from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImitationDecision:
    decision_id: str
    option_signatures: tuple[tuple[int, ...], ...]
    selected_signatures: frozenset[tuple[int, ...]]

    def __post_init__(self) -> None:
        if not self.decision_id or not self.option_signatures:
            raise ValueError("imitation decision must be non-empty")
        if not self.selected_signatures.issubset(set(self.option_signatures)):
            raise ValueError("selected signature is not a legal option")


def balanced_option_weights(
    labels: Sequence[int | bool],
    *,
    decision_weight: float,
) -> list[float]:
    clean = [int(label) for label in labels]
    positives = sum(clean)
    negatives = len(clean) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("option labels require positive and negative examples")
    if not math.isfinite(decision_weight) or decision_weight <= 0.0:
        raise ValueError("decision weight must be finite and positive")
    positive_weight = decision_weight / (2.0 * positives)
    negative_weight = decision_weight / (2.0 * negatives)
    return [
        positive_weight if label else negative_weight
        for label in clean
    ]


def _top_signature(
    decision: ImitationDecision,
    scores: Sequence[float],
) -> tuple[int, ...] | None:
    if len(scores) != len(decision.option_signatures):
        raise ValueError("decision score count mismatch")
    by_signature: dict[tuple[int, ...], float] = {}
    for signature, score in zip(decision.option_signatures, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("scores must be finite")
        by_signature[signature] = max(by_signature.get(signature, -math.inf), value)
    maximum = max(by_signature.values())
    winners = [
        signature
        for signature, value in by_signature.items()
        if value == maximum
    ]
    return winners[0] if len(winners) == 1 else None


def _top_signature_and_margin(
    decision: ImitationDecision,
    scores: Sequence[float],
) -> tuple[tuple[int, ...] | None, float]:
    if len(scores) != len(decision.option_signatures):
        raise ValueError("decision score count mismatch")
    by_signature: dict[tuple[int, ...], float] = {}
    for signature, score in zip(decision.option_signatures, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("scores must be finite")
        by_signature[signature] = max(by_signature.get(signature, -math.inf), value)
    ordered = sorted(by_signature.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1]:
        return None, 0.0
    return ordered[0][0], ordered[0][1] - ordered[1][1]


def semantic_predictions(
    decisions: Sequence[ImitationDecision],
    scores: Sequence[float],
) -> list[tuple[int, ...] | None]:
    predictions = []
    cursor = 0
    for decision in decisions:
        stop = cursor + len(decision.option_signatures)
        predictions.append(_top_signature(decision, scores[cursor:stop]))
        cursor = stop
    if cursor != len(scores):
        raise ValueError("unused option scores remain")
    return predictions


def semantic_accuracy(
    decisions: Sequence[ImitationDecision],
    scores: Sequence[float],
) -> float:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    predictions = semantic_predictions(decisions, scores)
    correct = sum(
        prediction in decision.selected_signatures
        for decision, prediction in zip(decisions, predictions, strict=True)
    )
    return correct / len(decisions)


def consensus_semantic_metrics(
    decisions: Sequence[ImitationDecision],
    v1_scores: Sequence[float],
    v2_scores: Sequence[float],
) -> dict[str, float | int]:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    first = semantic_predictions(decisions, v1_scores)
    second = semantic_predictions(decisions, v2_scores)
    covered = 0
    correct = 0
    for decision, left, right in zip(
        decisions,
        first,
        second,
        strict=True,
    ):
        if left is None or left != right:
            continue
        covered += 1
        correct += int(left in decision.selected_signatures)
    return {
        "decision_count": len(decisions),
        "covered": covered,
        "covered_correct": correct,
        "coverage": covered / len(decisions),
        "covered_accuracy": correct / covered if covered else 0.0,
    }


def confident_consensus_metrics(
    decisions: Sequence[ImitationDecision],
    v1_scores: Sequence[float],
    v2_scores: Sequence[float],
    *,
    target_coverage: float,
) -> dict[str, float | int]:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be within (0, 1]")
    cursor = 0
    agreed: list[tuple[float, bool]] = []
    for decision in decisions:
        stop = cursor + len(decision.option_signatures)
        first, first_margin = _top_signature_and_margin(
            decision,
            v1_scores[cursor:stop],
        )
        second, second_margin = _top_signature_and_margin(
            decision,
            v2_scores[cursor:stop],
        )
        cursor = stop
        if first is None or first != second:
            continue
        agreed.append(
            (
                min(first_margin, second_margin),
                first in decision.selected_signatures,
            )
        )
    if cursor != len(v1_scores) or cursor != len(v2_scores):
        raise ValueError("unused option scores remain")
    desired = math.ceil(target_coverage * len(decisions))
    if len(agreed) < desired:
        minimum_margin = 0.0
        selected = agreed
    else:
        minimum_margin = sorted(
            (margin for margin, _ in agreed),
            reverse=True,
        )[desired - 1]
        selected = [
            row for row in agreed if row[0] >= minimum_margin
        ]
    correct = sum(is_correct for _, is_correct in selected)
    return {
        "decision_count": len(decisions),
        "agreed": len(agreed),
        "covered": len(selected),
        "covered_correct": correct,
        "coverage": len(selected) / len(decisions),
        "covered_accuracy": correct / len(selected) if selected else 0.0,
        "target_coverage": target_coverage,
        "minimum_margin": minimum_margin,
    }


def _semantic_score_map(
    decision: ImitationDecision,
    scores: Sequence[float],
) -> dict[tuple[int, ...], float]:
    if len(scores) != len(decision.option_signatures):
        raise ValueError("decision score count mismatch")
    by_signature: dict[tuple[int, ...], float] = {}
    for signature, score in zip(
        decision.option_signatures,
        scores,
        strict=True,
    ):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("scores must be finite")
        by_signature[signature] = max(
            by_signature.get(signature, -math.inf),
            value,
        )
    return by_signature


def semantic_set_prediction(
    decision: ImitationDecision,
    scores: Sequence[float],
    *,
    threshold: float,
    minimum: int,
    maximum: int,
) -> frozenset[tuple[int, ...]]:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if minimum < 0 or maximum < minimum:
        raise ValueError("selection bounds are invalid")
    by_signature = _semantic_score_map(decision, scores)
    if minimum > len(by_signature):
        raise ValueError("selection minimum exceeds unique option count")
    legal_maximum = min(maximum, len(by_signature))
    ranked = sorted(
        by_signature,
        key=lambda signature: (-by_signature[signature], signature),
    )
    chosen = [
        signature
        for signature in ranked
        if by_signature[signature] >= threshold
    ]
    if len(chosen) < minimum:
        chosen = ranked[:minimum]
    elif len(chosen) > legal_maximum:
        chosen = chosen[:legal_maximum]
    return frozenset(chosen)


def semantic_set_accuracy(
    decisions: Sequence[ImitationDecision],
    scores: Sequence[float],
    bounds: Sequence[tuple[int, int]],
    *,
    threshold: float,
) -> float:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    if len(bounds) != len(decisions):
        raise ValueError("decision bound count mismatch")
    cursor = 0
    correct = 0
    for decision, (minimum, maximum) in zip(
        decisions,
        bounds,
        strict=True,
    ):
        stop = cursor + len(decision.option_signatures)
        prediction = semantic_set_prediction(
            decision,
            scores[cursor:stop],
            threshold=threshold,
            minimum=minimum,
            maximum=maximum,
        )
        correct += int(prediction == decision.selected_signatures)
        cursor = stop
    if cursor != len(scores):
        raise ValueError("unused option scores remain")
    return correct / len(decisions)


def calibrate_semantic_threshold(
    decisions: Sequence[ImitationDecision],
    scores: Sequence[float],
    bounds: Sequence[tuple[int, int]],
) -> float:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    candidates = {0.0, 0.5, 1.0}
    candidates.update(float(score) for score in scores)
    ranked = sorted(candidates)
    return max(
        ranked,
        key=lambda threshold: (
            semantic_set_accuracy(
                decisions,
                scores,
                bounds,
                threshold=threshold,
            ),
            threshold,
        ),
    )
