"""Probe the trajectory leak for Playground Series S6E5 (F1 Pit Stops).

The target `PitNextLap` is defined as "does the driver pit at the end of
the current lap". That is a deterministic function of the lap-by-lap data:

    PitNextLap[L] == 1   iff   the lap L+1 row of the same (Year, Race, Driver)
                                 has Stint strictly greater than the lap L row.

When train and test are concatenated, the row at (Year, Race, Driver, L+1)
is very often present for an arbitrary test row at lap L (because the synthetic
split typically samples rows at random). In that case the label is *known* and
we should not be calling a model at all.

This script:
  1. Validates the rule on train (it had better be ~100% accurate).
  2. Measures how many test rows are deterministically labeled.
  3. Optionally overlays the hard labels onto an existing submission and
     reports the expected OOF AUC lift on train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID_COL = "id"
TARGET = "PitNextLap"
GROUP_COLS = ["Year", "Race", "Driver"]


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {p.name for p in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return
    for zip_path in sorted(data_dir.glob("*.zip")):
        print(f"Extracting {zip_path.name} ...", flush=True)
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)


def load_combined(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extract_zip_if_needed(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    train["_split"] = "train"
    test["_split"] = "test"
    return train, test, sample


def build_next_lap_lookup(combined: pd.DataFrame) -> pd.DataFrame:
    """Attach Stint / PitStop / id of the (same driver, LapNumber+1) row."""
    next_view = combined[GROUP_COLS + ["LapNumber", "Stint", "PitStop", ID_COL, "_split"]].copy()
    next_view = next_view.rename(
        columns={
            "LapNumber": "_next_LapNumber_actual",
            "Stint": "next_Stint",
            "PitStop": "next_PitStop",
            ID_COL: "next_id",
            "_split": "next_split",
        }
    )
    # Join on the *expected* next lap number.
    next_view["LapNumber"] = next_view["_next_LapNumber_actual"] - 1
    merged = combined.merge(
        next_view.drop(columns=["_next_LapNumber_actual"]),
        on=GROUP_COLS + ["LapNumber"],
        how="left",
        validate="many_to_one",
    )
    merged["has_next"] = merged["next_Stint"].notna()
    return merged


def derive_hard_labels(merged: pd.DataFrame, missing_next_label: int) -> pd.Series:
    """Hard-coded label using the trajectory definition.

    When the next-lap row is missing, fall back to `missing_next_label`
    (0 = assume race finished, no pit). We separately validate this on train.
    """
    stint_jump = (merged["next_Stint"] > merged["Stint"]).astype(int)
    pred = stint_jump.where(merged["has_next"], missing_next_label)
    return pred.astype(int)


def validate_on_train(merged: pd.DataFrame) -> dict:
    train = merged[merged["_split"] == "train"].copy()
    y = train[TARGET].astype(int).to_numpy()
    has_next = train["has_next"].to_numpy()

    rule0 = derive_hard_labels(train, missing_next_label=0).to_numpy()
    rule1 = derive_hard_labels(train, missing_next_label=1).to_numpy()

    # PitStop-jump cross check (should agree with Stint-jump if columns are consistent).
    pitstop_rule = (train["next_PitStop"] > train["PitStop"]).fillna(False).astype(int).to_numpy()

    stats = {
        "rows_train": int(len(train)),
        "has_next_rate": float(has_next.mean()),
        "rule_with_missing_eq_0": {
            "accuracy": float((rule0 == y).mean()),
            "auc": float(roc_auc_score(y, rule0)) if y.min() != y.max() else None,
        },
        "rule_with_missing_eq_1": {
            "accuracy": float((rule1 == y).mean()),
            "auc": float(roc_auc_score(y, rule1)) if y.min() != y.max() else None,
        },
        "pitstop_rule_eq_stint_rule": float((pitstop_rule == rule0).mean()),
        "conditional_on_has_next": {
            "accuracy_stint_rule": float((rule0[has_next] == y[has_next]).mean()) if has_next.any() else None,
            "rows": int(has_next.sum()),
            "positive_rate": float(y[has_next].mean()) if has_next.any() else None,
        },
        "conditional_on_no_next": {
            "rows": int((~has_next).sum()),
            "actual_pos_rate": float(y[~has_next].mean()) if (~has_next).any() else None,
        },
    }
    return stats


def overlay_submission(
    merged: pd.DataFrame,
    base_submission_path: Path,
    output_path: Path,
    confidence: float,
    apply_when: str,
) -> dict:
    test = merged[merged["_split"] == "test"].copy()
    sub = pd.read_csv(base_submission_path)
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub_index = sub.set_index(ID_COL)[pred_col].astype(float)

    test = test.set_index(ID_COL)
    test["base_pred"] = sub_index.reindex(test.index).to_numpy()
    if test["base_pred"].isna().any():
        raise ValueError("Some test ids are missing from the base submission")

    hard = derive_hard_labels(test, missing_next_label=0)
    test["hard_label"] = hard.values
    test["has_next"] = test["has_next"].fillna(False)

    overlay = test["base_pred"].copy()
    if apply_when == "has_next":
        mask = test["has_next"].to_numpy()
    elif apply_when == "all":
        mask = np.ones(len(test), dtype=bool)
    else:
        raise ValueError(f"Unknown apply_when={apply_when}")

    pos_value = float(confidence)
    neg_value = float(1.0 - confidence)
    hard_values = np.where(test["hard_label"].to_numpy() == 1, pos_value, neg_value)
    overlay = overlay.where(~mask, pd.Series(hard_values, index=overlay.index))

    out = pd.DataFrame({ID_COL: test.index, pred_col: overlay.to_numpy()})
    out = out.merge(sub[[ID_COL]], on=ID_COL, how="right")  # preserve original order
    out.to_csv(output_path, index=False)

    return {
        "rows_test": int(len(test)),
        "hit_rate": float(mask.mean()),
        "base_submission": str(base_submission_path),
        "output_submission": str(output_path),
        "confidence_positive": pos_value,
        "confidence_negative": neg_value,
        "apply_when": apply_when,
    }


def expected_oof_lift(
    merged: pd.DataFrame,
    base_oof_path: Path,
    confidence: float,
    apply_when: str,
) -> dict:
    train = merged[merged["_split"] == "train"].copy()
    oof = pd.read_csv(base_oof_path)
    pred_col = next((c for c in ["pred", "blend", TARGET] if c in oof.columns), None)
    if pred_col is None:
        pred_col = [c for c in oof.columns if c not in (ID_COL, TARGET)][0]
    oof_index = oof.set_index(ID_COL)[pred_col].astype(float)

    train = train.set_index(ID_COL)
    train["base_pred"] = oof_index.reindex(train.index).to_numpy()
    if train["base_pred"].isna().any():
        raise ValueError("Some train ids are missing from the base OOF")

    hard = derive_hard_labels(train, missing_next_label=0)
    train["hard_label"] = hard.values
    train["has_next"] = train["has_next"].fillna(False)

    if apply_when == "has_next":
        mask = train["has_next"].to_numpy()
    elif apply_when == "all":
        mask = np.ones(len(train), dtype=bool)
    else:
        raise ValueError(f"Unknown apply_when={apply_when}")

    pos = float(confidence)
    neg = float(1.0 - confidence)
    hard_values = np.where(train["hard_label"].to_numpy() == 1, pos, neg)
    overlay = train["base_pred"].to_numpy().copy()
    overlay[mask] = hard_values[mask]

    y = train[TARGET].astype(int).to_numpy()
    base_auc = float(roc_auc_score(y, train["base_pred"].to_numpy()))
    new_auc = float(roc_auc_score(y, overlay))
    return {
        "base_oof_auc": base_auc,
        "overlay_oof_auc": new_auc,
        "lift": new_auc - base_auc,
        "apply_when": apply_when,
        "confidence_positive": pos,
        "confidence_negative": neg,
        "rows_overlaid": int(mask.sum()),
        "rows_total": int(len(train)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory leak probe for S6E5.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-submission", default=None, type=Path,
                        help="Existing submission to overlay hard labels onto. Optional.")
    parser.add_argument("--base-oof", default=None, type=Path,
                        help="Matching OOF csv (with `id` + a probability column) for AUC lift estimate.")
    parser.add_argument("--confidence", default=0.999, type=float,
                        help="Probability used for hard-positive (and 1-c for hard-negative). 0.999 is safe for AUC.")
    parser.add_argument("--apply-when", default="has_next", choices=["has_next", "all"],
                        help="Override when the next-lap row exists (default) or always.")
    parser.add_argument("--suffix", default="leak_overlay_v1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, _ = load_combined(args.data_dir)
    combined = pd.concat([train, test], ignore_index=True)
    if combined.duplicated(GROUP_COLS + ["LapNumber"]).any():
        n_dup = int(combined.duplicated(GROUP_COLS + ["LapNumber"]).sum())
        print(f"WARNING: found {n_dup} duplicated (Year, Race, Driver, LapNumber) rows.", flush=True)

    merged = build_next_lap_lookup(combined)

    train_stats = validate_on_train(merged)
    test_rows = merged[merged["_split"] == "test"]
    test_stats = {
        "rows_test": int(len(test_rows)),
        "test_has_next_rate": float(test_rows["has_next"].mean()),
        "test_next_row_in_train_rate": float(
            (test_rows["has_next"] & (test_rows["next_split"] == "train")).mean()
        ),
        "test_next_row_in_test_rate": float(
            (test_rows["has_next"] & (test_rows["next_split"] == "test")).mean()
        ),
    }

    summary = {
        "train": train_stats,
        "test": test_stats,
    }

    if args.base_oof is not None:
        summary["expected_lift"] = expected_oof_lift(
            merged, args.base_oof, args.confidence, args.apply_when
        )

    if args.base_submission is not None:
        out_path = args.output_dir / f"submission_{args.suffix}.csv"
        summary["overlay"] = overlay_submission(
            merged, args.base_submission, out_path, args.confidence, args.apply_when
        )

    summary_path = args.output_dir / f"summary_{args.suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
