from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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
from rolling_policy.metrics import (  # noqa: E402
    calibration_bins,
    equal_frequency_ece,
    roc_auc,
)
from rolling_policy.tree_model import predict_exported  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    report_path = snapshot_dir / "holdout_report.json"
    if report_path.exists():
        raise SystemExit("holdout was already opened for this snapshot")

    snapshot = _read_object(snapshot_path)
    plan = _read_object(snapshot_dir / "holdout_plan.json")
    if plan.get("holdout_opened") is not False:
        raise SystemExit("holdout plan is not sealed")
    if plan.get("snapshot_id") != snapshot.get("snapshot_id"):
        raise SystemExit("holdout plan snapshot mismatch")
    if plan.get("snapshot_sha256") != sha256_file(snapshot_path):
        raise SystemExit("snapshot hash mismatch")
    if plan.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise SystemExit("unknown feature schema version")
    expected_thresholds = {
        "minimum_auc": MIN_VALUE_AUC,
        "maximum_ece": MAX_VALUE_ECE,
        "ece_bins": ECE_BINS,
        "both_outcome_classes_required": True,
    }
    if plan.get("thresholds") != expected_thresholds:
        raise SystemExit("sealed holdout thresholds were changed")
    decisions_path = snapshot_dir / "public" / "decisions.jsonl"
    if plan.get("decision_dataset_sha256") != sha256_file(decisions_path):
        raise SystemExit("holdout decision dataset hash mismatch")

    models = {}
    feature_schema = {}
    for name, relative_path in plan["models"].items():
        path = snapshot_dir / str(relative_path)
        if plan["model_sha256"].get(name) != sha256_file(path):
            raise SystemExit(f"{name} model hash mismatch")
        model = _read_object(path)
        audit_feature_names(model["feature_names"])
        models[name] = model
        feature_schema[name] = {"feature_names": model["feature_names"]}
    if plan.get("feature_schema_sha256") != sha256_bytes(
        canonical_json_bytes(feature_schema)
    ):
        raise SystemExit("feature schema hash mismatch")

    features = []
    labels = []
    source_labels: dict[str, int] = {}
    game_ids: set[str] = set()
    team_counts: Counter[str] = Counter()
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("split") != "holdout" or not row.get("single_choice_main"):
                continue
            won = row.get("won")
            if not isinstance(won, bool):
                raise ValueError("holdout contains a non-binary outcome")
            label = int(won)
            episode_id = str(row["episode_id"])
            source_id = f"{episode_id}:{int(row['target_seat'])}"
            previous = source_labels.setdefault(source_id, label)
            if previous != label:
                raise ValueError(f"inconsistent outcome for episode/seat {source_id}")
            game_ids.add(episode_id)
            labels.append(label)
            features.append(visible_state_features(row["observation"]))
            team_counts[str(row["team_id"])] += 1
    if len(set(labels)) != 2:
        raise SystemExit("holdout does not contain both outcome classes")

    thresholds = plan["thresholds"]
    model_reports = {}
    passed = True
    for name, model in models.items():
        probabilities = predict_exported(model, features)
        auc = roc_auc(labels, probabilities)
        ece = equal_frequency_ece(
            labels,
            probabilities,
            bins=int(thresholds["ece_bins"]),
        )
        model_passed = (
            auc >= float(thresholds["minimum_auc"])
            and ece <= float(thresholds["maximum_ece"])
        )
        passed = passed and model_passed
        model_reports[name] = {
            "auc": auc,
            "ece": ece,
            "passed": model_passed,
            "calibration_bins": calibration_bins(
                labels,
                probabilities,
                bins=int(thresholds["ece_bins"]),
            ),
        }

    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "holdout_plan_sha256": sha256_file(snapshot_dir / "holdout_plan.json"),
        "holdout_opened": True,
        "decision": "PASS" if passed else "REJECT_VALUE_GATE",
        "thresholds": thresholds,
        "holdout": {
            "decision_rows": len(labels),
            "games": len(game_ids),
            "episode_seat_perspectives": len(source_labels),
            "wins": sum(labels),
            "losses": len(labels) - sum(labels),
            "episode_seat_wins": sum(source_labels.values()),
            "episode_seat_losses": len(source_labels) - sum(source_labels.values()),
            "team_decision_rows": dict(sorted(team_counts.items())),
            "both_outcome_classes": len(set(labels)) == 2,
        },
        "models": model_reports,
    }
    with report_path.open("xb") as file:
        file.write(canonical_json_bytes(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
