from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.extract import option_signature  # noqa: E402
from rolling_policy.features import (  # noqa: E402
    visible_action_features,
    visible_state_features,
)
from rolling_policy.hashing import canonical_json_bytes, sha256_file  # noqa: E402
from rolling_policy.imitation import (  # noqa: E402
    ImitationDecision,
    confident_consensus_metrics,
)
from rolling_policy.tree_model import predict_exported  # noqa: E402


TARGET_COVERAGE = 0.50


def _load_validation(
    decisions_path: Path,
    mode: str,
) -> tuple[list[ImitationDecision], list[dict[str, float]], int]:
    decisions = []
    features = []
    skipped_holdout = 0
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split == "holdout":
                skipped_holdout += 1
                continue
            eligible = (
                bool(row.get("single_choice_main"))
                if mode == "main"
                else bool(
                    not row.get("forced")
                    and int(row.get("min_count", -1)) == 1
                    and int(row.get("max_count", -1)) == 1
                    and int(row.get("context", 0)) != 0
                )
            )
            if split != "validation" or not eligible:
                continue
            observation = row["observation"]
            seat = int(row["target_seat"])
            signatures = tuple(
                option_signature(observation, option, seat)
                for option in observation["select"]["option"]
            )
            selected = frozenset(
                tuple(int(value) for value in signature)
                for signature in row["selected_signature"]
            )
            decisions.append(
                ImitationDecision(
                    decision_id=str(row["decision_id"]),
                    option_signatures=signatures,
                    selected_signatures=selected,
                )
            )
            state_features = visible_state_features(observation)
            features.extend(
                visible_action_features(
                    observation,
                    option,
                    base_features=state_features,
                )
                for option in observation["select"]["option"]
            )
    return decisions, features, skipped_holdout


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mode", choices=("main", "followup"), default="main")
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    prefix = "" if args.mode == "main" else "followup_"
    report_path = snapshot_dir / f"{prefix}imitation_confidence_report.json"
    gate_path = snapshot_dir / f"{prefix}imitation_confidence_gate.json"
    if report_path.exists() or gate_path.exists():
        raise SystemExit("imitation confidence audit already exists")

    training = _load_object(
        snapshot_dir / f"{prefix}imitation_training_report.json"
    )
    decisions_path = snapshot_dir / "public" / "decisions.jsonl"
    if training["decision_dataset_sha256"] != sha256_file(decisions_path):
        raise SystemExit("decision dataset hash mismatch")
    models = {}
    for name in ("clone_v1", "clone_v2"):
        model_path = (
            snapshot_dir / f"{prefix}imitation_models" / f"{name}.json"
        )
        if training["model_sha256"][name] != sha256_file(model_path):
            raise SystemExit(f"{name} model hash mismatch")
        models[name] = _load_object(model_path)

    decisions, features, skipped_holdout = _load_validation(
        decisions_path,
        args.mode,
    )
    print(
        f"loaded validation_decisions={len(decisions)} "
        f"validation_options={len(features)} "
        f"holdout_rows_skipped={skipped_holdout}",
        flush=True,
    )
    scores = {
        name: predict_exported(model, features)
        for name, model in models.items()
    }
    confidence = confident_consensus_metrics(
        decisions,
        scores["clone_v1"],
        scores["clone_v2"],
        target_coverage=TARGET_COVERAGE,
    )
    gates = training["gates"]
    passed = (
        confidence["coverage"] >= gates["minimum_consensus_coverage"]
        and confidence["covered_accuracy"]
        >= gates["minimum_consensus_accuracy"]
    )
    report = {
        "snapshot_id": training["snapshot_id"],
        "mode": args.mode,
        "model_sha256": training["model_sha256"],
        "selection_rule": {
            "target_coverage": TARGET_COVERAGE,
            "threshold_uses_labels": False,
            "confidence": "minimum of the two top-vs-runner-up probability margins",
        },
        "gates": {
            "minimum_consensus_coverage": gates[
                "minimum_consensus_coverage"
            ],
            "minimum_consensus_accuracy": gates[
                "minimum_consensus_accuracy"
            ],
        },
        "confidence_validation": confidence,
        "holdout_rows_skipped_before_label_access": skipped_holdout,
        "decision": "PASS" if passed else "REJECT_CONFIDENCE_GATE",
    }
    with report_path.open("xb") as file:
        file.write(canonical_json_bytes(report) + b"\n")
    if passed:
        gate = {
            "snapshot_id": training["snapshot_id"],
            "model_sha256": training["model_sha256"],
            "minimum_margin": confidence["minimum_margin"],
            "validation_coverage": confidence["coverage"],
            "validation_covered_accuracy": confidence["covered_accuracy"],
            "future_labels_used": False,
        }
        with gate_path.open("xb") as file:
            file.write(canonical_json_bytes(gate) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
