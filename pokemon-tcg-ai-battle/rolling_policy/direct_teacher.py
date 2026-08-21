from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extract import option_signature
from .features import visible_action_features, visible_state_features
from .imitation import ImitationDecision, balanced_option_weights
from .schema import parse_utc_datetime


DIRECT_CONTEXTS = {
    "main": 0,
    "bench": 5,
    "search": 7,
    "discard": 8,
}


def type_routed_prediction(
    decision: ImitationDecision,
    option_scores: list[float],
    *,
    classes: list[int],
    type_probabilities: list[float],
) -> frozenset[tuple[int, ...]]:
    if len(option_scores) != len(decision.option_signatures):
        raise ValueError("decision score count mismatch")
    if len(classes) != len(type_probabilities) or not classes:
        raise ValueError("type probability count mismatch")
    if any(not math.isfinite(float(value)) for value in type_probabilities):
        raise ValueError("type probabilities must be finite")
    legal_types = {
        signature[0]
        for signature in decision.option_signatures
    }
    candidates = [
        (float(probability), -int(option_type), int(option_type))
        for option_type, probability in zip(
            classes,
            type_probabilities,
            strict=True,
        )
        if int(option_type) in legal_types
    ]
    if candidates:
        selected_type = max(candidates)[2]
        indices = [
            index
            for index, signature in enumerate(decision.option_signatures)
            if signature[0] == selected_type
        ]
    else:
        indices = list(range(len(decision.option_signatures)))
    by_signature: dict[tuple[int, ...], float] = {}
    for index in indices:
        score = float(option_scores[index])
        if not math.isfinite(score):
            raise ValueError("option scores must be finite")
        signature = decision.option_signatures[index]
        by_signature[signature] = max(
            score,
            by_signature.get(signature, -math.inf),
        )
    chosen = min(
        by_signature,
        key=lambda signature: (-by_signature[signature], signature),
    )
    return frozenset({chosen})


def direct_semantic_signature(
    signature: tuple[int, ...],
    mode: str,
) -> tuple[int, ...]:
    if mode not in DIRECT_CONTEXTS:
        raise ValueError(f"unknown direct-teacher mode: {mode}")
    if mode != "main":
        return signature
    if len(signature) != 15:
        raise ValueError("main action signature must contain 15 integers")
    option_type = signature[0]
    if option_type == 7:
        return (option_type, signature[1])
    if option_type in {8, 9}:
        return (
            option_type,
            signature[1],
            signature[3],
            signature[4],
            signature[9],
            signature[10],
        )
    if option_type == 10:
        return (
            option_type,
            signature[1],
            signature[2],
            signature[7],
            signature[8],
        )
    if option_type in {12, 14}:
        return (option_type,)
    if option_type == 13:
        return (option_type, signature[5])
    return signature


def direct_type_features(observation: dict[str, Any]) -> dict[str, float]:
    features = visible_state_features(observation)
    current = observation.get("current")
    current = current if isinstance(current, dict) else {}
    seat = int(current.get("yourIndex", 0))
    select = observation.get("select")
    select = select if isinstance(select, dict) else {}
    options = select.get("option")
    options = options if isinstance(options, list) else []
    action_counts: Counter[tuple[int, ...]] = Counter()
    type_counts: Counter[int] = Counter()
    for option in options:
        if not isinstance(option, dict):
            continue
        signature = direct_semantic_signature(
            option_signature(observation, option, seat),
            "main",
        )
        action_counts[signature] += 1
        type_counts[signature[0]] += 1
    for option_type, count in type_counts.items():
        features[f"available:type={option_type}"] = float(count)
    for signature, count in action_counts.items():
        encoded = ":".join(str(value) for value in signature)
        features[f"available:action={encoded}"] = float(count)
    features["n:available_action_types"] = float(len(type_counts))
    features["n:available_semantic_actions"] = float(len(action_counts))
    return features


def _source_id(row: dict[str, Any]) -> str:
    return f"{row['episode_id']}:{int(row['target_seat'])}"


def eligible_direct_decision(row: dict[str, Any], mode: str) -> bool:
    try:
        context = DIRECT_CONTEXTS[mode]
    except KeyError as error:
        raise ValueError(f"unknown direct-teacher mode: {mode}") from error
    if bool(row.get("forced")) or int(row.get("context", -1)) != context:
        return False
    minimum = int(row.get("min_count", -1))
    maximum = int(row.get("max_count", -1))
    if minimum < 0 or maximum < minimum:
        return False
    if mode == "main":
        return (
            bool(row.get("single_choice_main"))
            and minimum == 1
            and maximum == 1
        )
    return True


def direct_option_weights(
    labels: list[int | bool],
    *,
    decision_weight: float,
) -> list[float]:
    clean = [int(label) for label in labels]
    if not clean or any(label not in {0, 1} for label in clean):
        raise ValueError("option labels must be a non-empty binary list")
    if not math.isfinite(decision_weight) or decision_weight <= 0.0:
        raise ValueError("decision weight must be finite and positive")
    positives = sum(clean)
    if positives in {0, len(clean)}:
        return [decision_weight / len(clean)] * len(clean)
    return balanced_option_weights(clean, decision_weight=decision_weight)


def _episode_weights(episodes_path: Path) -> dict[str, float]:
    groups: dict[str, tuple[str, str, int, int]] = {}
    group_counts: Counter[tuple[str, str, int, int]] = Counter()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with episodes_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in {"train", "validation"}:
                continue
            source_id = _source_id(row)
            if source_id in groups:
                raise ValueError(f"duplicate episode/seat: {source_id}")
            created = parse_utc_datetime(row["create_time_utc"])
            bucket = int((created - epoch).total_seconds() // (12 * 60 * 60))
            group = (
                split,
                str(row["team_id"]),
                int(row["target_seat"]),
                bucket,
            )
            groups[source_id] = group
            group_counts[group] += 1
    return {
        source_id: 1.0 / group_counts[group]
        for source_id, group in groups.items()
    }


def _decision_counts(decisions_path: Path, mode: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split"))
            if split == "holdout":
                continue
            if split in {"train", "validation"} and eligible_direct_decision(
                row,
                mode,
            ):
                counts[_source_id(row)] += 1
    return counts


def build_direct_examples(
    decisions_path: Path,
    episodes_path: Path,
    mode: str,
) -> dict[str, Any]:
    episode_weights = _episode_weights(episodes_path)
    decision_counts = _decision_counts(decisions_path, mode)
    data: dict[str, Any] = {
        split: {
            "features": [],
            "labels": [],
            "weights": [],
            "source_ids": [],
            "decision_source_ids": [],
            "decision_features": [],
            "decision_labels": [],
            "decision_weights": [],
            "decisions": [],
            "bounds": [],
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
            if split not in data or not eligible_direct_decision(row, mode):
                continue

            observation = row["observation"]
            seat = int(row["target_seat"])
            signatures = tuple(
                direct_semantic_signature(
                    option_signature(observation, option, seat),
                    mode,
                )
                for option in observation["select"]["option"]
            )
            selected = frozenset(
                direct_semantic_signature(
                    tuple(int(value) for value in signature),
                    mode,
                )
                for signature in row["selected_signature"]
            )
            labels = [int(signature in selected) for signature in signatures]
            source_id = _source_id(row)
            try:
                decision_weight = (
                    episode_weights[source_id] / decision_counts[source_id]
                )
            except (KeyError, ZeroDivisionError) as error:
                raise ValueError(
                    f"missing direct-teacher weight for {source_id}"
                ) from error
            option_weights = direct_option_weights(
                labels,
                decision_weight=decision_weight,
            )
            decision = ImitationDecision(
                decision_id=str(row["decision_id"]),
                option_signatures=signatures,
                selected_signatures=selected,
            )
            minimum = int(row["min_count"])
            maximum = int(row["max_count"])
            state_features = visible_state_features(observation)

            data[split]["decisions"].append(decision)
            data[split]["bounds"].append((minimum, maximum))
            data[split]["decision_source_ids"].append(source_id)
            if mode == "main":
                data[split]["decision_features"].append(
                    direct_type_features(observation)
                )
                data[split]["decision_labels"].append(
                    next(iter(selected))[0]
                )
                data[split]["decision_weights"].append(decision_weight)
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


load_direct_examples = build_direct_examples
