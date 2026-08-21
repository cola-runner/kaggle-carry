from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
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
    balanced_option_weights,
    consensus_semantic_metrics,
    semantic_accuracy,
)
from rolling_policy.schema import parse_utc_datetime  # noqa: E402
from rolling_policy.tree_model import (  # noqa: E402
    fit_exported_classifier,
    predict_exported,
)


GATES = {
    "minimum_model_accuracy": 0.50,
    "minimum_first_option_improvement": 0.08,
    "minimum_consensus_coverage": 0.40,
    "minimum_consensus_accuracy": 0.65,
}
MODEL_SPECS = {
    "clone_v1": {
        "seed": 1701,
        "configs": (
            {
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 80,
                "learning_rate": 0.06,
                "l2_regularization": 3.0,
            },
            {
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 120,
                "learning_rate": 0.04,
                "l2_regularization": 4.0,
            },
        ),
    },
    "clone_v2": {
        "seed": 2909,
        "configs": (
            {
                "max_iter": 120,
                "max_leaf_nodes": 23,
                "min_samples_leaf": 70,
                "learning_rate": 0.055,
                "l2_regularization": 3.5,
            },
            {
                "max_iter": 160,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 130,
                "learning_rate": 0.04,
                "l2_regularization": 5.0,
            },
        ),
    },
}


def _source_id(row: dict[str, Any]) -> str:
    return f"{row['episode_id']}:{int(row['target_seat'])}"


def _episode_weights(
    episodes_path: Path,
) -> dict[str, float]:
    rows = {}
    group_counts: Counter[tuple[str, str, int, int]] = Counter()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with episodes_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in {"train", "validation"}:
                continue
            source_id = _source_id(row)
            if source_id in rows:
                raise ValueError(f"duplicate episode/seat: {source_id}")
            created = parse_utc_datetime(row["create_time_utc"])
            bucket = int((created - epoch).total_seconds() // (12 * 60 * 60))
            key = (
                split,
                str(row["team_id"]),
                int(row["target_seat"]),
                bucket,
            )
            rows[source_id] = key
            group_counts[key] += 1
    return {
        source_id: 1.0 / group_counts[key]
        for source_id, key in rows.items()
    }


def _eligible(row: dict[str, Any], mode: str) -> bool:
    if mode == "main":
        return bool(row.get("single_choice_main"))
    return bool(
        not row.get("forced")
        and int(row.get("min_count", -1)) == 1
        and int(row.get("max_count", -1)) == 1
        and int(row.get("context", 0)) != 0
    )


def _decision_counts(decisions_path: Path, mode: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split == "holdout":
                continue
            if split in {"train", "validation"} and _eligible(row, mode):
                counts[_source_id(row)] += 1
    return counts


def _load_examples(
    decisions_path: Path,
    episode_weights: dict[str, float],
    decision_counts: Counter[str],
    mode: str,
) -> dict[str, Any]:
    data = {
        split: {
            "features": [],
            "labels": [],
            "weights": [],
            "source_ids": [],
            "decisions": [],
        }
        for split in ("train", "validation")
    }
    skipped_holdout = 0
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split == "holdout":
                skipped_holdout += 1
                continue
            if split not in data or not _eligible(row, mode):
                continue
            observation = row["observation"]
            seat = int(row["target_seat"])
            selected = frozenset(
                tuple(int(value) for value in signature)
                for signature in row["selected_signature"]
            )
            signatures = tuple(
                option_signature(observation, option, seat)
                for option in observation["select"]["option"]
            )
            labels = [int(signature in selected) for signature in signatures]
            source_id = _source_id(row)
            decision_weight = (
                episode_weights[source_id] / decision_counts[source_id]
            )
            option_weights = balanced_option_weights(
                labels,
                decision_weight=decision_weight,
            )
            state_features = visible_state_features(observation)
            data[split]["decisions"].append(
                ImitationDecision(
                    decision_id=str(row["decision_id"]),
                    option_signatures=signatures,
                    selected_signatures=selected,
                )
            )
            for option, label, weight in zip(
                observation["select"]["option"],
                labels,
                option_weights,
                strict=True,
            ):
                data[split]["features"].append(
                    visible_action_features(
                        observation,
                        option,
                        base_features=state_features,
                    )
                )
                data[split]["labels"].append(label)
                data[split]["weights"].append(weight)
                data[split]["source_ids"].append(source_id)
    data["skipped_holdout_rows_before_label_access"] = skipped_holdout
    return data


def _keep(seed: int, source_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < 92


def _normalized(weights: list[float]) -> list[float]:
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


def _first_option_scores(
    decisions: list[ImitationDecision],
) -> list[float]:
    scores = []
    for decision in decisions:
        scores.extend(
            [1.0] + [0.0] * (len(decision.option_signatures) - 1)
        )
    return scores


def _train_one(
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
    train_x = [train["features"][index] for index in indices]
    train_y = [train["labels"][index] for index in indices]
    train_w = _normalized([train["weights"][index] for index in indices])
    candidates = []
    for config_index, config in enumerate(spec["configs"]):
        model, exporter = fit_exported_classifier(
            train_x,
            train_y,
            train_w,
            random_seed=seed,
            return_native_probabilities=False,
            **config,
        )
        if exporter["max_export_error"] >= 1e-6:
            raise ValueError(f"{name} export parity failed")
        scores = predict_exported(model, validation["features"])
        accuracy = semantic_accuracy(validation["decisions"], scores)
        candidate = {
            "config_index": config_index,
            "config": config,
            "validation_semantic_accuracy": accuracy,
            "max_export_error": exporter["max_export_error"],
        }
        candidates.append((model, scores, candidate))
        print(
            f"{name} candidate={config_index} "
            f"validation_accuracy={accuracy:.4f}",
            flush=True,
        )
    chosen_index = max(
        range(len(candidates)),
        key=lambda index: candidates[index][2]["validation_semantic_accuracy"],
    )
    model, scores, _ = candidates[chosen_index]
    return model, scores, {
        "name": name,
        "seed": seed,
        "training_option_rows": len(train_x),
        "chosen_config_index": chosen_index,
        "candidates": [candidate for _, _, candidate in candidates],
        "validation_semantic_accuracy": semantic_accuracy(
            validation["decisions"],
            scores,
        ),
        "feature_count": len(model["feature_names"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mode", choices=("main", "followup"), default="main")
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    prefix = "" if args.mode == "main" else "followup_"
    output_dir = snapshot_dir / f"{prefix}imitation_models"
    report_path = snapshot_dir / f"{prefix}imitation_training_report.json"
    forward_plan_path = snapshot_dir / f"{prefix}imitation_forward_plan.json"
    if output_dir.exists() or report_path.exists() or forward_plan_path.exists():
        raise SystemExit("imitation training outputs already exist")

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

    episode_weights = _episode_weights(episodes_path)
    decision_counts = _decision_counts(decisions_path, args.mode)
    data = _load_examples(
        decisions_path,
        episode_weights,
        decision_counts,
        args.mode,
    )
    train = data["train"]
    validation = data["validation"]
    first_accuracy = semantic_accuracy(
        validation["decisions"],
        _first_option_scores(validation["decisions"]),
    )
    print(
        f"loaded train_decisions={len(train['decisions'])} "
        f"train_options={len(train['features'])} "
        f"validation_decisions={len(validation['decisions'])} "
        f"validation_options={len(validation['features'])} "
        f"first_option_accuracy={first_accuracy:.4f}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="rolling-imitation-",
        dir=snapshot_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_models = temporary / "imitation_models"
        temporary_models.mkdir()
        model_reports = {}
        chosen_scores = {}
        model_hashes = {}
        for name, spec in MODEL_SPECS.items():
            model, scores, report = _train_one(
                name,
                spec,
                train,
                validation,
            )
            path = temporary_models / f"{name}.json"
            with path.open("xb") as file:
                file.write(canonical_json_bytes(model) + b"\n")
            model_reports[name] = report
            chosen_scores[name] = scores
            model_hashes[name] = sha256_file(path)

        consensus = consensus_semantic_metrics(
            validation["decisions"],
            chosen_scores["clone_v1"],
            chosen_scores["clone_v2"],
        )
        individual_pass = all(
            report["validation_semantic_accuracy"]
            >= GATES["minimum_model_accuracy"]
            and report["validation_semantic_accuracy"] - first_accuracy
            >= GATES["minimum_first_option_improvement"]
            for report in model_reports.values()
        )
        passed = (
            individual_pass
            and consensus["coverage"] >= GATES["minimum_consensus_coverage"]
            and consensus["covered_accuracy"]
            >= GATES["minimum_consensus_accuracy"]
        )
        report = {
            "snapshot_id": snapshot["snapshot_id"],
            "mode": args.mode,
            "decision_dataset_sha256": extraction["output_sha256"]["decisions"],
            "episode_dataset_sha256": extraction["output_sha256"]["episodes"],
            "gates": GATES,
            "first_option_baseline_accuracy": first_accuracy,
            "models": model_reports,
            "model_sha256": model_hashes,
            "consensus_validation": consensus,
            "row_counts": {
                "train_decisions": len(train["decisions"]),
                "train_options": len(train["features"]),
                "validation_decisions": len(validation["decisions"]),
                "validation_options": len(validation["features"]),
                "holdout_rows_skipped_before_label_access": data[
                    "skipped_holdout_rows_before_label_access"
                ],
            },
            "decision": "PASS" if passed else "REJECT_IMITATION_GATE",
        }
        forward_plan = {
            "source_snapshot_id": snapshot["snapshot_id"],
            "mode": args.mode,
            "source_snapshot_sha256": sha256_file(snapshot_path),
            "model_sha256": model_hashes,
            "earliest_nonoverlap_cutoff_utc": (
                parse_utc_datetime(snapshot["cutoff_utc"])
                + timedelta(hours=12)
            )
            .isoformat()
            .replace("+00:00", "Z"),
            "prospective_window_must_start_after_utc": snapshot["cutoff_utc"],
            "minimum_future_decisions": 5_000,
            "minimum_consensus_coverage": GATES[
                "minimum_consensus_coverage"
            ],
            "minimum_consensus_accuracy": GATES[
                "minimum_consensus_accuracy"
            ],
            "opened": False,
        }
        os.replace(temporary_models, output_dir)
        with report_path.open("xb") as file:
            file.write(canonical_json_bytes(report) + b"\n")
        with forward_plan_path.open("xb") as file:
            file.write(canonical_json_bytes(forward_plan) + b"\n")

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
