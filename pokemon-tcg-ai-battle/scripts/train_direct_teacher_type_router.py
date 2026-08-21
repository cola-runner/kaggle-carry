from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.direct_teacher import (  # noqa: E402
    build_direct_examples,
    type_routed_prediction,
)
from rolling_policy.features import FEATURE_SCHEMA_VERSION  # noqa: E402
from rolling_policy.hashing import canonical_json_bytes, sha256_file  # noqa: E402
from rolling_policy.tree_model import (  # noqa: E402
    fit_exported_multiclass_classifier,
    predict_exported,
    predict_exported_multiclass,
)


ROUTER_SPECS = {
    "v1": {
        "seed": 4319,
        "keep_percent": 95,
        "class_balance_exponent": 0.0,
        "max_iter": 120,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "learning_rate": 0.06,
        "l2_regularization": 3.0,
    },
    "v2": {
        "seed": 7757,
        "keep_percent": 95,
        "class_balance_exponent": 0.35,
        "max_iter": 160,
        "max_leaf_nodes": 47,
        "min_samples_leaf": 30,
        "learning_rate": 0.05,
        "l2_regularization": 4.0,
    },
}


def _keep(seed: int, source_id: str, keep_percent: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < keep_percent


def _normalized(weights: list[float]) -> list[float]:
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


def _train_router(
    name: str,
    spec: dict[str, Any],
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[list[float]], dict[str, Any]]:
    seed = int(spec["seed"])
    keep_percent = int(spec["keep_percent"])
    exponent = float(spec["class_balance_exponent"])
    indices = [
        index
        for index, source_id in enumerate(train["decision_source_ids"])
        if _keep(seed, source_id, keep_percent)
    ]
    class_counts = Counter(train["decision_labels"][index] for index in indices)
    rows = [train["decision_features"][index] for index in indices]
    labels = [train["decision_labels"][index] for index in indices]
    weights = _normalized(
        [
            train["decision_weights"][index]
            / math.pow(class_counts[train["decision_labels"][index]], exponent)
            for index in indices
        ]
    )
    config = {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "seed",
            "keep_percent",
            "class_balance_exponent",
        }
    }
    model, export_report = fit_exported_multiclass_classifier(
        rows,
        labels,
        weights,
        random_seed=seed,
        return_native_probabilities=False,
        **config,
    )
    if export_report["max_export_error"] >= 1e-6:
        raise ValueError(f"{name} export parity failed")
    probabilities = predict_exported_multiclass(
        model,
        validation["decision_features"],
    )
    report = {
        "seed": seed,
        "keep_percent": keep_percent,
        "class_balance_exponent": exponent,
        "config": config,
        "training_decisions": len(indices),
        "training_source_count": len(
            {train["decision_source_ids"][index] for index in indices}
        ),
        "training_class_counts": {
            str(key): value for key, value in sorted(class_counts.items())
        },
        "classes": model["classes"],
        "feature_count": len(model["feature_names"]),
        "max_export_error": export_report["max_export_error"],
    }
    return model, probabilities, report


def _average_probabilities(
    first: list[list[float]],
    second: list[list[float]],
) -> list[list[float]]:
    if len(first) != len(second):
        raise ValueError("router row count mismatch")
    return [
        [
            (left + right) / 2.0
            for left, right in zip(first_row, second_row, strict=True)
        ]
        for first_row, second_row in zip(first, second, strict=True)
    ]


def _evaluate_router(
    validation: dict[str, list[Any]],
    option_scores: list[float],
    *,
    classes: list[int],
    probabilities: list[list[float]],
) -> dict[str, Any]:
    if len(probabilities) != len(validation["decisions"]):
        raise ValueError("router decision count mismatch")
    cursor = 0
    type_correct = 0
    exact_correct = 0
    selected_counts: Counter[int] = Counter()
    type_correct_counts: Counter[int] = Counter()
    exact_correct_counts: Counter[int] = Counter()
    for decision, truth, row in zip(
        validation["decisions"],
        validation["decision_labels"],
        probabilities,
        strict=True,
    ):
        stop = cursor + len(decision.option_signatures)
        legal_types = {signature[0] for signature in decision.option_signatures}
        predicted_type = max(
            (
                (float(probability), -int(option_type), int(option_type))
                for option_type, probability in zip(classes, row, strict=True)
                if int(option_type) in legal_types
            ),
            default=(0.0, 0, -1),
        )[2]
        prediction = type_routed_prediction(
            decision,
            option_scores[cursor:stop],
            classes=classes,
            type_probabilities=row,
        )
        type_ok = predicted_type == truth
        exact_ok = prediction == decision.selected_signatures
        selected_counts[truth] += 1
        type_correct_counts[truth] += type_ok
        exact_correct_counts[truth] += exact_ok
        type_correct += type_ok
        exact_correct += exact_ok
        cursor = stop
    if cursor != len(option_scores):
        raise ValueError("unused ranker scores remain")
    count = len(validation["decisions"])
    return {
        "decision_count": count,
        "type_correct": type_correct,
        "type_accuracy": type_correct / count,
        "exact_correct": exact_correct,
        "exact_semantic_accuracy": exact_correct / count,
        "by_selected_type": {
            str(option_type): {
                "decisions": selected_counts[option_type],
                "type_accuracy": (
                    type_correct_counts[option_type]
                    / selected_counts[option_type]
                ),
                "exact_semantic_accuracy": (
                    exact_correct_counts[option_type]
                    / selected_counts[option_type]
                ),
            }
            for option_type in sorted(selected_counts)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    ranker_dir = snapshot_dir / "direct_teacher_models"
    ranker_report_path = snapshot_dir / "direct_teacher_training_report.json"
    output_dir = snapshot_dir / "direct_teacher_type_models"
    report_path = snapshot_dir / "direct_teacher_type_training_report.json"
    if output_dir.exists() or report_path.exists():
        raise SystemExit("direct-teacher type-router outputs already exist")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    extraction = json.loads(
        (snapshot_dir / "extraction_report.json").read_text(encoding="utf-8")
    )
    ranker_report = json.loads(ranker_report_path.read_text(encoding="utf-8"))
    for name, expected in ranker_report["modes"]["main"][
        "model_sha256"
    ].items():
        actual = sha256_file(ranker_dir / f"main_{name}.json")
        if actual != expected:
            raise SystemExit(f"main ranker hash mismatch: {name}")

    data = build_direct_examples(
        snapshot_dir / "public" / "decisions.jsonl",
        snapshot_dir / "public" / "episodes.jsonl",
        "main",
    )
    train = data["train"]
    validation = data["validation"]
    ranker_models = [
        json.loads((ranker_dir / f"main_{name}.json").read_text(encoding="utf-8"))
        for name in ("v1", "v2")
    ]
    ranker_predictions = [
        predict_exported(model, validation["features"])
        for model in ranker_models
    ]
    ranker_scores = [
        (left + right) / 2.0
        for left, right in zip(
            ranker_predictions[0],
            ranker_predictions[1],
            strict=True,
        )
    ]

    with tempfile.TemporaryDirectory(
        prefix="direct-teacher-router-",
        dir=snapshot_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_models = temporary / "direct_teacher_type_models"
        temporary_models.mkdir()
        models: dict[str, dict[str, Any]] = {}
        probabilities: dict[str, list[list[float]]] = {}
        model_reports: dict[str, Any] = {}
        model_hashes: dict[str, str] = {}
        for name, spec in ROUTER_SPECS.items():
            print(f"training type router={name}", flush=True)
            model, rows, model_report = _train_router(
                name,
                spec,
                train,
                validation,
            )
            model["metadata"] = {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "mode": "main_type_router",
                "snapshot_id": snapshot["snapshot_id"],
            }
            path = temporary_models / f"main_type_{name}.json"
            with path.open("xb") as file:
                file.write(canonical_json_bytes(model) + b"\n")
            models[name] = model
            probabilities[name] = rows
            model_reports[name] = model_report
            model_hashes[name] = sha256_file(path)

        choices = {
            "v1": probabilities["v1"],
            "v2": probabilities["v2"],
            "ensemble": _average_probabilities(
                probabilities["v1"],
                probabilities["v2"],
            ),
        }
        choice_reports = {
            name: _evaluate_router(
                validation,
                ranker_scores,
                classes=models["v1"]["classes"],
                probabilities=rows,
            )
            for name, rows in choices.items()
        }
        selected = max(
            choice_reports,
            key=lambda name: (
                choice_reports[name]["exact_semantic_accuracy"],
                choice_reports[name]["type_accuracy"],
                name == "ensemble",
            ),
        )
        selected_report = choice_reports[selected]
        other_correct = sum(
            round(
                ranker_report["modes"][mode][
                    "validation_exact_set_accuracy"
                ]
                * ranker_report["modes"][mode]["row_counts"][
                    "validation_decisions"
                ]
            )
            for mode in ("bench", "search")
        )
        other_decisions = sum(
            ranker_report["modes"][mode]["row_counts"][
                "validation_decisions"
            ]
            for mode in ("bench", "search")
        )
        overall_correct = selected_report["exact_correct"] + other_correct
        overall_decisions = (
            selected_report["decision_count"] + other_decisions
        )
        overall_accuracy = overall_correct / overall_decisions
        passed = (
            selected_report["exact_semantic_accuracy"] >= 0.65
            and overall_accuracy >= 0.72
        )
        report = {
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": sha256_file(snapshot_path),
            "decision_dataset_sha256": extraction["output_sha256"]["decisions"],
            "ranker_training_report_sha256": sha256_file(ranker_report_path),
            "ranker_model_sha256": ranker_report["modes"]["main"][
                "model_sha256"
            ],
            "models": model_reports,
            "model_sha256": model_hashes,
            "router_choices": choice_reports,
            "selected_router": selected,
            "main_gate": 0.65,
            "overall_validation": {
                "correct": overall_correct,
                "decisions": overall_decisions,
                "exact_set_accuracy": overall_accuracy,
                "gate": 0.72,
            },
            "holdout_rows_skipped_before_label_access": data[
                "skipped_holdout_rows_before_label_access"
            ],
            "decision": "PASS" if passed else "REJECT_VALIDATION_GATE",
        }
        os.replace(temporary_models, output_dir)
        with report_path.open("xb") as file:
            file.write(canonical_json_bytes(report) + b"\n")

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
