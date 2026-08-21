from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from .features import audit_feature_names


def _matrix(
    rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for name, value in row.items():
            feature_index = lookup.get(name)
            if feature_index is not None:
                matrix[row_index, feature_index] = float(value)
    return matrix


def _export(
    classifier: HistGradientBoostingClassifier,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    trees: list[list[list[float | int]]] = []
    for stage in classifier._predictors:
        trees.append(_export_nodes(stage[0].nodes))
    return {
        "version": 1,
        "model_type": "hist_gradient_boosting_binary_probability",
        "baseline": float(classifier._baseline_prediction.reshape(-1)[0]),
        "feature_names": list(feature_names),
        "trees": trees,
    }


def _export_nodes(nodes: Any) -> list[list[float | int]]:
    tree: list[list[float | int]] = []
    for node in nodes:
        if int(node["is_leaf"]):
            tree.append([float(node["value"])])
        else:
            tree.append(
                [
                    int(node["feature_idx"]),
                    float(node["num_threshold"]),
                    int(node["left"]),
                    int(node["right"]),
                    int(node["missing_go_to_left"]),
                ]
            )
    return tree


def _tree_value(tree: Sequence[Sequence[float | int]], values: np.ndarray) -> float:
    node_index = 0
    while len(tree[node_index]) > 1:
        feature, threshold, left, right, missing_left = tree[node_index]
        value = values[int(feature)]
        if np.isnan(value):
            node_index = int(left if missing_left else right)
        else:
            node_index = int(left if value <= threshold else right)
    return float(tree[node_index][0])


def _raw_scores(model: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    scores = np.full(matrix.shape[0], float(model["baseline"]), dtype=np.float64)
    for tree in model["trees"]:
        for row_index, values in enumerate(matrix):
            scores[row_index] += _tree_value(tree, values)
    return scores


def predict_exported(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, float]],
) -> list[float]:
    feature_names = list(model["feature_names"])
    matrix = _matrix(rows, feature_names)
    raw = _raw_scores(model, matrix)
    calibration = model.get("calibration")
    if isinstance(calibration, Mapping):
        slope = float(calibration["slope"])
        intercept = float(calibration["intercept"])
        raw = slope * raw + intercept
    return [_sigmoid(float(value)) for value in raw]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def predict_exported_raw(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, float]],
) -> list[float]:
    matrix = _matrix(rows, list(model["feature_names"]))
    return _raw_scores(model, matrix).tolist()


def set_platt_calibration(
    model: Mapping[str, Any],
    *,
    slope: float,
    intercept: float,
) -> dict[str, Any]:
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0.0:
        raise ValueError("Platt calibration requires finite positive slope")
    calibrated = deepcopy(dict(model))
    calibrated["calibration"] = {
        "type": "platt",
        "slope": float(slope),
        "intercept": float(intercept),
    }
    return calibrated


def fit_exported_classifier(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    weights: Sequence[float],
    *,
    random_seed: int,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    learning_rate: float = 0.08,
    l2_regularization: float = 1.0,
    return_native_probabilities: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows or len(rows) != len(labels) or len(rows) != len(weights):
        raise ValueError("rows, labels, and weights must have equal non-zero length")
    vectorizer = DictVectorizer(sparse=False, sort=True)
    matrix = vectorizer.fit_transform(rows).astype(np.float32, copy=False)
    feature_names = vectorizer.get_feature_names_out().tolist()
    audit_feature_names(feature_names)
    classifier = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_seed,
        early_stopping=False,
    )
    classifier.fit(
        matrix,
        np.asarray(labels, dtype=np.int8),
        sample_weight=np.asarray(weights, dtype=np.float64),
    )
    exported = _export(classifier, feature_names)
    parity_count = min(len(rows), 1_024)
    native = classifier.predict_proba(matrix[:parity_count])[:, 1]
    portable = np.asarray(predict_exported(exported, rows[:parity_count]))
    maximum_error = float(np.max(np.abs(portable - native)))
    metrics: dict[str, Any] = {
        "max_export_error": maximum_error,
    }
    if return_native_probabilities:
        metrics["_native_probabilities"] = native.tolist()
    return exported, metrics


def fit_exported_multiclass_classifier(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    weights: Sequence[float],
    *,
    random_seed: int,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    learning_rate: float = 0.08,
    l2_regularization: float = 1.0,
    return_native_probabilities: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows or len(rows) != len(labels) or len(rows) != len(weights):
        raise ValueError("rows, labels, and weights must have equal non-zero length")
    vectorizer = DictVectorizer(sparse=False, sort=True)
    matrix = vectorizer.fit_transform(rows).astype(np.float32, copy=False)
    feature_names = vectorizer.get_feature_names_out().tolist()
    audit_feature_names(feature_names)
    classifier = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_seed,
        early_stopping=False,
    )
    classifier.fit(
        matrix,
        np.asarray(labels),
        sample_weight=np.asarray(weights, dtype=np.float64),
    )
    classes = [int(value) for value in classifier.classes_]
    if len(classes) < 3:
        raise ValueError("multiclass export requires at least three classes")
    exported = {
        "version": 1,
        "model_type": "hist_gradient_boosting_multiclass_probability",
        "classes": classes,
        "baselines": [
            float(value)
            for value in classifier._baseline_prediction.reshape(-1)
        ],
        "feature_names": feature_names,
        "trees": [
            [_export_nodes(predictor.nodes) for predictor in stage]
            for stage in classifier._predictors
        ],
    }
    parity_count = min(len(rows), 1_024)
    native = classifier.predict_proba(matrix[:parity_count])
    portable = np.asarray(
        predict_exported_multiclass(exported, rows[:parity_count])
    )
    maximum_error = float(np.max(np.abs(portable - native)))
    metrics: dict[str, Any] = {"max_export_error": maximum_error}
    if return_native_probabilities:
        metrics["_native_probabilities"] = native.tolist()
    return exported, metrics


def predict_exported_multiclass(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, float]],
) -> list[list[float]]:
    classes = list(model["classes"])
    baselines = np.asarray(model["baselines"], dtype=np.float64)
    if len(classes) != len(baselines):
        raise ValueError("multiclass baseline count mismatch")
    matrix = _matrix(rows, list(model["feature_names"]))
    raw = np.tile(baselines, (len(rows), 1))
    for stage in model["trees"]:
        if len(stage) != len(classes):
            raise ValueError("multiclass tree count mismatch")
        for class_index, tree in enumerate(stage):
            for row_index, values in enumerate(matrix):
                raw[row_index, class_index] += _tree_value(tree, values)
    raw -= np.max(raw, axis=1, keepdims=True)
    probabilities = np.exp(raw)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities.tolist()
