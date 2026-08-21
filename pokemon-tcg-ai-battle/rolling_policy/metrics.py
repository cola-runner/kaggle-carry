from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
) -> tuple[list[int], list[float]]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have equal non-zero length")
    clean_labels = [int(label) for label in labels]
    if any(label not in (0, 1) for label in clean_labels):
        raise ValueError("labels must be binary")
    if len(set(clean_labels)) != 2:
        raise ValueError("metrics require both outcome classes")
    clean_probabilities = [float(value) for value in probabilities]
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in clean_probabilities
    ):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return clean_labels, clean_probabilities


def roc_auc(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
) -> float:
    clean_labels, clean_probabilities = _validate(labels, probabilities)
    positives = [p for y, p in zip(clean_labels, clean_probabilities) if y == 1]
    negatives = [p for y, p in zip(clean_labels, clean_probabilities) if y == 0]
    ordered = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                ordered += 1.0
            elif positive == negative:
                ordered += 0.5
    return ordered / (len(positives) * len(negatives))


def equal_frequency_ece(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    details = calibration_bins(labels, probabilities, bins=bins)
    total = sum(int(group["count"]) for group in details)
    return sum(
        (int(group["count"]) / total)
        * abs(float(group["mean_probability"]) - float(group["positive_rate"]))
        for group in details
    )


def calibration_bins(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    clean_labels, clean_probabilities = _validate(labels, probabilities)
    if bins <= 0:
        raise ValueError("bins must be positive")
    ordered = sorted(
        enumerate(zip(clean_probabilities, clean_labels)),
        key=lambda item: (item[1][0], item[0]),
    )
    item_count = len(ordered)
    bin_count = min(bins, item_count)
    quotient, remainder = divmod(item_count, bin_count)
    cursor = 0
    details: list[dict[str, float | int]] = []
    for index in range(bin_count):
        size = quotient + int(index < remainder)
        group = ordered[cursor : cursor + size]
        cursor += size
        probabilities_in_bin = [item[1][0] for item in group]
        labels_in_bin = [item[1][1] for item in group]
        details.append(
            {
                "bin": index,
                "count": size,
                "minimum_probability": min(probabilities_in_bin),
                "maximum_probability": max(probabilities_in_bin),
                "mean_probability": sum(probabilities_in_bin) / size,
                "positive_rate": sum(labels_in_bin) / size,
            }
        )
    return details
