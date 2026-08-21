from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.constants import (  # noqa: E402
    ECE_BINS,
    MAX_VALUE_ECE,
    MIN_VALUE_AUC,
)
from rolling_policy.features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    audit_feature_names,
    visible_state_features,
)
from rolling_policy.hashing import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from rolling_policy.metrics import equal_frequency_ece, roc_auc  # noqa: E402
from rolling_policy.schema import parse_utc_datetime  # noqa: E402
from rolling_policy.tree_model import (  # noqa: E402
    fit_exported_classifier,
    predict_exported,
    predict_exported_raw,
    set_platt_calibration,
)


MODEL_SPECS = {
    "v1": {
        "seed": 1701,
        "configs": (
            {
                "max_iter": 140,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 70,
                "learning_rate": 0.06,
                "l2_regularization": 2.0,
            },
            {
                "max_iter": 180,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 90,
                "learning_rate": 0.045,
                "l2_regularization": 3.0,
            },
        ),
    },
    "v2": {
        "seed": 2909,
        "configs": (
            {
                "max_iter": 150,
                "max_leaf_nodes": 23,
                "min_samples_leaf": 60,
                "learning_rate": 0.055,
                "l2_regularization": 2.5,
            },
            {
                "max_iter": 200,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 110,
                "learning_rate": 0.04,
                "l2_regularization": 4.0,
            },
        ),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as file:
        file.write(canonical_json_bytes(value) + b"\n")


def _twelve_hour_bucket(text: str) -> int:
    created = parse_utc_datetime(text)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return int((created - epoch).total_seconds() // (12 * 60 * 60))


def _episode_weights(
    episodes_path: Path,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metadata: dict[str, dict[str, Any]] = {}
    group_counts: Counter[tuple[str, str, int, int]] = Counter()
    with episodes_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in {"train", "validation"}:
                continue
            episode_id = str(row["episode_id"])
            target_seat = int(row["target_seat"])
            source_id = f"{episode_id}:{target_seat}"
            if source_id in metadata:
                raise ValueError(f"duplicate episode/seat row: {source_id}")
            metadata[source_id] = {
                "split": split,
                "team_id": str(row["team_id"]),
                "target_seat": target_seat,
                "create_time_utc": str(row["create_time_utc"]),
            }
            group_counts[
                (
                    split,
                    str(row["team_id"]),
                    int(row["target_seat"]),
                    _twelve_hour_bucket(str(row["create_time_utc"])),
                )
            ] += 1
    weights = {}
    for source_id, row in metadata.items():
        key = (
            row["split"],
            row["team_id"],
            row["target_seat"],
            _twelve_hour_bucket(row["create_time_utc"]),
        )
        weights[source_id] = 1.0 / group_counts[key]
    return weights, metadata


def _load_visible_rows(
    decisions_path: Path,
    episode_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    per_episode: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    with decisions_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            split = str(row.get("split"))
            if split == "holdout":
                counts["holdout_rows_skipped_before_label_access"] += 1
                continue
            if split not in {"train", "validation"}:
                raise ValueError(f"unknown split on decision line {line_number}")
            if not row.get("single_choice_main"):
                continue
            episode_id = str(row["episode_id"])
            source_id = f"{episode_id}:{int(row['target_seat'])}"
            if source_id not in episode_weights:
                raise ValueError(f"missing episode metadata for {source_id}")
            won = row.get("won")
            if not isinstance(won, bool):
                raise ValueError(f"non-binary outcome for {episode_id}")
            record = {
                "episode_id": episode_id,
                "source_id": source_id,
                "split": split,
                "created": parse_utc_datetime(str(row["create_time_utc"])),
                "label": int(won),
                "features": visible_state_features(row["observation"]),
            }
            records.append(record)
            per_episode[source_id] += 1
            counts[split] += 1
    for record in records:
        record["weight"] = (
            episode_weights[record["source_id"]]
            / per_episode[record["source_id"]]
        )
    return records, dict(counts)


def _normalize(weights: list[float]) -> list[float]:
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


def _seed_keeps(seed: int, episode_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < 92


def _fit_platt(
    raw_scores: list[float],
    labels: list[int],
    weights: list[float],
) -> tuple[float, float]:
    if len(set(labels)) != 2:
        raise ValueError("calibration requires both outcome classes")
    classifier = LogisticRegression(
        C=100.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=0,
    )
    classifier.fit(
        np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1),
        np.asarray(labels, dtype=np.int8),
        sample_weight=np.asarray(_normalize(weights), dtype=np.float64),
    )
    slope = float(classifier.coef_.reshape(-1)[0])
    intercept = float(classifier.intercept_.reshape(-1)[0])
    if slope <= 0.0:
        raise ValueError("value model has non-positive calibration slope")
    return slope, intercept


def _subset(
    records: list[dict[str, Any]],
    *,
    split: str,
    before: datetime | None = None,
    at_or_after: datetime | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, float]], list[int], list[float]]:
    selected = [
        record
        for record in records
        if record["split"] == split
        and (before is None or record["created"] < before)
        and (at_or_after is None or record["created"] >= at_or_after)
        and (seed is None or _seed_keeps(seed, record["source_id"]))
    ]
    if not selected:
        raise ValueError(f"empty {split} subset")
    return (
        [record["features"] for record in selected],
        [record["label"] for record in selected],
        _normalize([record["weight"] for record in selected]),
    )


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    return {
        "auc": roc_auc(labels, probabilities),
        "ece": equal_frequency_ece(labels, probabilities, bins=ECE_BINS),
    }


def _train_one(
    name: str,
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    validation_midpoint: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(spec["seed"])
    train_x, train_y, train_w = _subset(records, split="train", seed=seed)
    calibration_x, calibration_y, calibration_w = _subset(
        records,
        split="validation",
        before=validation_midpoint,
    )
    selection_x, selection_y, _ = _subset(
        records,
        split="validation",
        at_or_after=validation_midpoint,
    )
    candidate_reports = []
    candidates = []
    for config_index, config in enumerate(spec["configs"]):
        model, exporter = fit_exported_classifier(
            train_x,
            train_y,
            train_w,
            random_seed=seed,
            **config,
        )
        native = exporter.pop("_native_probabilities")
        if exporter["max_export_error"] >= 1e-6:
            raise ValueError(f"{name} export parity failed")
        slope, intercept = _fit_platt(
            predict_exported_raw(model, calibration_x),
            calibration_y,
            calibration_w,
        )
        calibrated = set_platt_calibration(
            model,
            slope=slope,
            intercept=intercept,
        )
        selection_probabilities = predict_exported(calibrated, selection_x)
        report = {
            "config_index": config_index,
            "config": config,
            "selection": _metrics(selection_y, selection_probabilities),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "max_export_error": exporter["max_export_error"],
            "native_train_probability_count": len(native),
        }
        candidate_reports.append(report)
        candidates.append((calibrated, report))
        print(
            f"{name} candidate={config_index} "
            f"selection_auc={report['selection']['auc']:.4f} "
            f"selection_ece={report['selection']['ece']:.4f}",
            flush=True,
        )

    chosen_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index][1]["selection"]["auc"]
            - candidates[index][1]["selection"]["ece"],
            candidates[index][1]["selection"]["auc"],
        ),
    )
    chosen_base = candidates[chosen_index][0]
    chosen_base.pop("calibration", None)
    validation_x, validation_y, validation_w = _subset(
        records,
        split="validation",
    )
    final_slope, final_intercept = _fit_platt(
        predict_exported_raw(chosen_base, validation_x),
        validation_y,
        validation_w,
    )
    final_model = set_platt_calibration(
        chosen_base,
        slope=final_slope,
        intercept=final_intercept,
    )
    final_probabilities = predict_exported(final_model, validation_x)
    audit_feature_names(final_model["feature_names"])
    return final_model, {
        "name": name,
        "seed": seed,
        "training_rows": len(train_x),
        "validation_rows": len(validation_x),
        "chosen_config_index": chosen_index,
        "candidates": candidate_reports,
        "full_validation_after_final_calibration": _metrics(
            validation_y,
            final_probabilities,
        ),
        "final_calibration": {
            "slope": final_slope,
            "intercept": final_intercept,
        },
        "feature_count": len(final_model["feature_names"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    model_dir = snapshot_dir / "value_models"
    report_path = snapshot_dir / "value_training_report.json"
    holdout_plan_path = snapshot_dir / "holdout_plan.json"
    if model_dir.exists() or report_path.exists() or holdout_plan_path.exists():
        raise SystemExit("value training outputs already exist for this snapshot")

    snapshot = _read_json(snapshot_path)
    extraction = _read_json(snapshot_dir / "extraction_report.json")
    if snapshot["snapshot_id"] != extraction["snapshot_id"]:
        raise SystemExit("snapshot/extraction identity mismatch")
    decisions_path = snapshot_dir / "public" / "decisions.jsonl"
    episodes_path = snapshot_dir / "public" / "episodes.jsonl"
    if sha256_file(decisions_path) != extraction["output_sha256"]["decisions"]:
        raise SystemExit("decision dataset hash mismatch")
    if sha256_file(episodes_path) != extraction["output_sha256"]["episodes"]:
        raise SystemExit("episode dataset hash mismatch")

    episode_weights, _ = _episode_weights(episodes_path)
    records, row_counts = _load_visible_rows(decisions_path, episode_weights)
    validation_start = parse_utc_datetime(snapshot["validation_start_utc"])
    holdout_start = parse_utc_datetime(snapshot["holdout_start_utc"])
    validation_midpoint = validation_start + (
        holdout_start - validation_start
    ) / 2
    print(f"loaded visible rows: {json.dumps(row_counts, sort_keys=True)}", flush=True)

    with tempfile.TemporaryDirectory(
        prefix="rolling-value-",
        dir=snapshot_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_models = temporary / "value_models"
        temporary_models.mkdir()
        reports = {}
        model_hashes = {}
        model_names = {}
        for name, spec in MODEL_SPECS.items():
            model, report = _train_one(
                name,
                spec,
                records,
                validation_midpoint,
            )
            model_path = temporary_models / f"{name}.json"
            _write_json(model_path, model)
            reports[name] = report
            model_hashes[name] = sha256_file(model_path)
            model_names[name] = f"value_models/{name}.json"

        feature_schema = {
            name: {
                "feature_names": json.loads(
                    (temporary_models / f"{name}.json").read_text()
                )["feature_names"]
            }
            for name in MODEL_SPECS
        }
        schema_sha256 = sha256_bytes(canonical_json_bytes(feature_schema))
        training_report = {
            "snapshot_id": snapshot["snapshot_id"],
            "decision_dataset_sha256": extraction["output_sha256"]["decisions"],
            "episode_dataset_sha256": extraction["output_sha256"]["episodes"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": schema_sha256,
            "row_counts": row_counts,
            "validation_midpoint_utc": validation_midpoint.isoformat().replace(
                "+00:00", "Z"
            ),
            "models": reports,
            "model_sha256": model_hashes,
        }
        _write_json(temporary / "value_training_report.json", training_report)
        holdout_plan = {
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": sha256_file(snapshot_path),
            "decision_dataset_sha256": extraction["output_sha256"]["decisions"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": schema_sha256,
            "models": model_names,
            "model_sha256": model_hashes,
            "thresholds": {
                "minimum_auc": MIN_VALUE_AUC,
                "maximum_ece": MAX_VALUE_ECE,
                "ece_bins": ECE_BINS,
                "both_outcome_classes_required": True,
            },
            "holdout_opened": False,
        }
        _write_json(temporary / "holdout_plan.json", holdout_plan)

        os.replace(temporary_models, model_dir)
        shutil.copyfile(temporary / "value_training_report.json", report_path)
        with holdout_plan_path.open("xb") as file:
            file.write((temporary / "holdout_plan.json").read_bytes())

    print(json.dumps(training_report, indent=2, sort_keys=True), flush=True)
    print(f"sealed holdout plan: {holdout_plan_path}", flush=True)


if __name__ == "__main__":
    main()
