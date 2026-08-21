from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.direct_teacher import (  # noqa: E402
    DIRECT_CONTEXTS,
    build_direct_examples,
)
from rolling_policy.features import FEATURE_SCHEMA_VERSION  # noqa: E402
from rolling_policy.hashing import canonical_json_bytes, sha256_file  # noqa: E402
from rolling_policy.imitation import (  # noqa: E402
    ImitationDecision,
    calibrate_semantic_threshold,
    semantic_set_accuracy,
)
from rolling_policy.tree_model import (  # noqa: E402
    fit_exported_classifier,
    predict_exported,
)


MINIMUM_VALIDATION_DECISIONS = 30
VALIDATION_GATES = {
    "main": 0.65,
    "bench": 0.55,
    "search": 0.55,
    "discard": 0.40,
}
MODEL_SPECS = {
    "v1": {
        "seed": 1701,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80,
        "learning_rate": 0.06,
        "l2_regularization": 3.0,
    },
    "v2": {
        "seed": 2909,
        "max_iter": 120,
        "max_leaf_nodes": 23,
        "min_samples_leaf": 70,
        "learning_rate": 0.055,
        "l2_regularization": 3.5,
    },
}


def _keep(seed: int, source_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < 92


def _normalized(weights: list[float]) -> list[float]:
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


def _first_option_scores(
    decisions: list[ImitationDecision],
) -> list[float]:
    scores: list[float] = []
    for decision in decisions:
        scores.extend(
            [1.0] + [0.0] * (len(decision.option_signatures) - 1)
        )
    return scores


def _train_model(
    name: str,
    spec: dict[str, Any],
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    seed = int(spec["seed"])
    indices = [
        index
        for index, source_id in enumerate(train["source_ids"])
        if _keep(seed, source_id)
    ]
    if not indices:
        raise ValueError(f"{name} source subsample is empty")
    train_x = [train["features"][index] for index in indices]
    train_y = [train["labels"][index] for index in indices]
    train_w = _normalized([train["weights"][index] for index in indices])
    config = {
        key: value
        for key, value in spec.items()
        if key != "seed"
    }
    model, export_report = fit_exported_classifier(
        train_x,
        train_y,
        train_w,
        random_seed=seed,
        return_native_probabilities=False,
        **config,
    )
    if export_report["max_export_error"] >= 1e-6:
        raise ValueError(f"{name} export parity failed")
    scores = predict_exported(model, validation["features"])
    report = {
        "seed": seed,
        "config": config,
        "training_option_rows": len(train_x),
        "training_source_count": len(
            {train["source_ids"][index] for index in indices}
        ),
        "feature_count": len(model["feature_names"]),
        "max_export_error": export_report["max_export_error"],
    }
    return model, scores, report


def _ensemble_scores(first: list[float], second: list[float]) -> list[float]:
    if len(first) != len(second):
        raise ValueError("model score count mismatch")
    return [
        (left + right) / 2.0
        for left, right in zip(first, second, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    output_dir = snapshot_dir / "direct_teacher_models"
    report_path = snapshot_dir / "direct_teacher_training_report.json"
    if output_dir.exists() or report_path.exists():
        raise SystemExit("direct-teacher training outputs already exist")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    extraction = json.loads(
        (snapshot_dir / "extraction_report.json").read_text(encoding="utf-8")
    )
    decisions_path = snapshot_dir / "public" / "decisions.jsonl"
    episodes_path = snapshot_dir / "public" / "episodes.jsonl"
    if sha256_file(decisions_path) != extraction["output_sha256"]["decisions"]:
        raise SystemExit("decision dataset hash mismatch")
    if sha256_file(episodes_path) != extraction["output_sha256"]["episodes"]:
        raise SystemExit("episode dataset hash mismatch")

    mode_reports: dict[str, Any] = {}
    total_correct = 0
    total_decisions = 0
    all_required_modes_passed = True
    with tempfile.TemporaryDirectory(
        prefix="direct-teacher-training-",
        dir=snapshot_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_models = temporary / "direct_teacher_models"
        temporary_models.mkdir()

        for mode in DIRECT_CONTEXTS:
            print(f"loading direct-teacher mode={mode}", flush=True)
            data = build_direct_examples(decisions_path, episodes_path, mode)
            train = data["train"]
            validation = data["validation"]
            validation_count = len(validation["decisions"])
            counts = {
                "train_decisions": len(train["decisions"]),
                "train_options": len(train["features"]),
                "validation_decisions": validation_count,
                "validation_options": len(validation["features"]),
                "holdout_rows_skipped_before_label_access": data[
                    "skipped_holdout_rows_before_label_access"
                ],
            }
            if validation_count < MINIMUM_VALIDATION_DECISIONS:
                mode_reports[mode] = {
                    "context": DIRECT_CONTEXTS[mode],
                    "status": "SKIP_INSUFFICIENT_VALIDATION",
                    "minimum_validation_decisions": (
                        MINIMUM_VALIDATION_DECISIONS
                    ),
                    "row_counts": counts,
                }
                if mode != "discard":
                    all_required_modes_passed = False
                print(
                    f"skipping mode={mode}: validation={validation_count} < "
                    f"{MINIMUM_VALIDATION_DECISIONS}",
                    flush=True,
                )
                del data, train, validation
                gc.collect()
                continue

            model_reports: dict[str, Any] = {}
            validation_scores: dict[str, list[float]] = {}
            model_hashes: dict[str, str] = {}
            for name, spec in MODEL_SPECS.items():
                print(f"training mode={mode} model={name}", flush=True)
                model, scores, model_report = _train_model(
                    f"{mode}_{name}",
                    spec,
                    train,
                    validation,
                )
                model["metadata"] = {
                    "context": DIRECT_CONTEXTS[mode],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "mode": mode,
                    "snapshot_id": snapshot["snapshot_id"],
                }
                model_path = temporary_models / f"{mode}_{name}.json"
                with model_path.open("xb") as file:
                    file.write(canonical_json_bytes(model) + b"\n")
                model_reports[name] = model_report
                validation_scores[name] = scores
                model_hashes[name] = sha256_file(model_path)
                del model
                gc.collect()

            ensemble = _ensemble_scores(
                validation_scores["v1"],
                validation_scores["v2"],
            )
            threshold = (
                0.5
                if mode == "main"
                else calibrate_semantic_threshold(
                    validation["decisions"],
                    ensemble,
                    validation["bounds"],
                )
            )
            exact_accuracy = semantic_set_accuracy(
                validation["decisions"],
                ensemble,
                validation["bounds"],
                threshold=threshold,
            )
            first_option_accuracy = semantic_set_accuracy(
                validation["decisions"],
                _first_option_scores(validation["decisions"]),
                validation["bounds"],
                threshold=0.5,
            )
            gate = VALIDATION_GATES[mode]
            passed = exact_accuracy >= gate
            mode_reports[mode] = {
                "context": DIRECT_CONTEXTS[mode],
                "status": "PASS" if passed else "REJECT_VALIDATION_GATE",
                "validation_gate": gate,
                "threshold": threshold,
                "validation_exact_set_accuracy": exact_accuracy,
                "first_option_baseline_accuracy": first_option_accuracy,
                "improvement_over_first_option": (
                    exact_accuracy - first_option_accuracy
                ),
                "models": model_reports,
                "model_sha256": model_hashes,
                "row_counts": counts,
            }
            if mode != "discard":
                all_required_modes_passed &= passed
            correct = round(exact_accuracy * validation_count)
            total_correct += correct
            total_decisions += validation_count
            print(
                f"mode={mode} validation_exact={exact_accuracy:.4f} "
                f"first_option={first_option_accuracy:.4f} "
                f"threshold={threshold:.6f}",
                flush=True,
            )
            del (
                data,
                train,
                validation,
                validation_scores,
                ensemble,
            )
            gc.collect()

        overall_accuracy = (
            total_correct / total_decisions if total_decisions else 0.0
        )
        report = {
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": sha256_file(snapshot_path),
            "decision_dataset_sha256": extraction["output_sha256"]["decisions"],
            "episode_dataset_sha256": extraction["output_sha256"]["episodes"],
            "minimum_validation_decisions": MINIMUM_VALIDATION_DECISIONS,
            "validation_gates": VALIDATION_GATES,
            "modes": mode_reports,
            "overall_validation": {
                "correct": total_correct,
                "decisions": total_decisions,
                "exact_set_accuracy": overall_accuracy,
                "gate": 0.72,
            },
            "decision": (
                "PASS"
                if all_required_modes_passed and overall_accuracy >= 0.72
                else "REJECT_VALIDATION_GATE"
            ),
        }
        os.replace(temporary_models, output_dir)
        with report_path.open("xb") as file:
            file.write(canonical_json_bytes(report) + b"\n")

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
