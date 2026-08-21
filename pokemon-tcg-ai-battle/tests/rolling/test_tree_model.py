from __future__ import annotations

import numpy as np

from rolling_policy.tree_model import (
    fit_exported_classifier,
    fit_exported_multiclass_classifier,
    predict_exported,
    predict_exported_multiclass,
    set_platt_calibration,
)


def test_exported_classifier_matches_native_probabilities() -> None:
    rows = [
        {"n:x": float(index % 7), "cat:seat=0": float(index % 2 == 0)}
        for index in range(240)
    ]
    labels = [int((index % 7) >= 3) for index in range(240)]
    weights = [1.0] * len(rows)
    model, metrics = fit_exported_classifier(
        rows,
        labels,
        weights,
        random_seed=1701,
        max_iter=30,
        max_leaf_nodes=7,
        min_samples_leaf=5,
    )
    exported = np.asarray(predict_exported(model, rows))
    native = np.asarray(metrics.pop("_native_probabilities"))
    assert np.max(np.abs(exported - native)) < 1e-6
    assert metrics["max_export_error"] < 1e-6


def test_exported_platt_calibration_changes_probability_not_tree_score() -> None:
    rows = [{"n:x": float(index)} for index in range(30)]
    labels = [int(index >= 15) for index in range(30)]
    model, _ = fit_exported_classifier(
        rows,
        labels,
        [1.0] * len(rows),
        random_seed=2909,
        max_iter=10,
        max_leaf_nodes=3,
        min_samples_leaf=3,
    )
    uncalibrated = predict_exported(model, rows)
    calibrated = set_platt_calibration(model, slope=0.5, intercept=0.2)
    assert predict_exported(calibrated, rows) != uncalibrated
    assert "calibration" not in model


def test_exported_multiclass_classifier_matches_native_probabilities() -> None:
    rows = [
        {
            "n:x": float(index % 9),
            "n:y": float((index // 3) % 5),
        }
        for index in range(360)
    ]
    classes = (7, 10, 14)
    labels = [classes[(index % 9) // 3] for index in range(360)]
    model, metrics = fit_exported_multiclass_classifier(
        rows,
        labels,
        [1.0] * len(rows),
        random_seed=4319,
        max_iter=30,
        max_leaf_nodes=7,
        min_samples_leaf=5,
    )
    exported = np.asarray(predict_exported_multiclass(model, rows))
    native = np.asarray(metrics.pop("_native_probabilities"))
    assert model["classes"] == list(classes)
    assert np.max(np.abs(exported - native)) < 1e-6
    assert metrics["max_export_error"] < 1e-6
