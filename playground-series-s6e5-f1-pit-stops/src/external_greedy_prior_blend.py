from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID_COL = "id"
TARGET = "PitNextLap"
PRED_COLS = ["pred", "blend", "catboost_realmlp_external"]


DEFAULT_STEPS = [
    ("ry_lap_position_stint", "Race,Year,LapNumber,Position,Stint", 0.00525),
    ("ry_lap_stint_tyre_comp", "Race,Year,LapNumber,Stint,TyreLife,Compound", 0.00325),
    ("ry_lap_pos_pit_comp", "Race,Year,LapNumber,Position,PitStop,Compound", 0.002),
]


def prediction_column(df: pd.DataFrame) -> str:
    for col in PRED_COLS:
        if col in df.columns:
            return col
    candidates = [col for col in df.columns if col not in {ID_COL, TARGET}]
    if candidates:
        return candidates[-1]
    if TARGET in df.columns:
        return TARGET
    raise ValueError(f"Could not infer prediction column from {list(df.columns)}")


def parse_step(raw: str) -> tuple[str, list[str], float]:
    name, keys_raw, alpha_raw = raw.split(":", 2)
    return name, [key.strip() for key in keys_raw.split(",") if key.strip()], float(alpha_raw)


def apply_step(values: np.ndarray, prior: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(prior)
    out = values.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * prior[mask]
    return out, mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a fixed sequence of tiny external-key prior blends.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--base-oof", required=True, type=Path)
    parser.add_argument("--base-submission", required=True, type=Path)
    parser.add_argument("--step", action="append", default=None, help="name:key1,key2:alpha")
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="external_greedy_prior_blend")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    base_oof_df = pd.read_csv(args.base_oof)
    base_sub_df = pd.read_csv(args.base_submission)
    base_oof_col = prediction_column(base_oof_df)
    base_sub_col = prediction_column(base_sub_df)

    y = train[TARGET].astype(int).to_numpy()
    oof_pred = base_oof_df[base_oof_col].astype(float).to_numpy()
    test_pred = base_sub_df[base_sub_col].astype(float).to_numpy()
    base_auc = float(roc_auc_score(y, oof_pred))

    raw_steps = args.step
    if raw_steps is None:
        steps = [(name, [key.strip() for key in keys.split(",") if key.strip()], alpha) for name, keys, alpha in DEFAULT_STEPS]
    else:
        steps = [parse_step(raw) for raw in raw_steps]

    step_summaries = []
    for name, keys, alpha in steps:
        stats = external.groupby(keys, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
        train_prior = train.merge(stats, on=keys, how="left")["mean"].to_numpy()
        test_prior = test.merge(stats, on=keys, how="left")["mean"].to_numpy()
        before_auc = float(roc_auc_score(y, oof_pred))
        oof_pred, train_mask = apply_step(oof_pred, train_prior, alpha)
        test_pred, test_mask = apply_step(test_pred, test_prior, alpha)
        after_auc = float(roc_auc_score(y, oof_pred))
        prior_auc_matched = None
        if train_mask.any() and np.unique(y[train_mask]).size > 1:
            prior_auc_matched = float(roc_auc_score(y[train_mask], train_prior[train_mask]))
        step_summaries.append(
            {
                "name": name,
                "keys": keys,
                "alpha": alpha,
                "before_auc": before_auc,
                "after_auc": after_auc,
                "delta": after_auc - before_auc,
                "train_match_rate": float(train_mask.mean()),
                "test_match_rate": float(test_mask.mean()),
                "prior_auc_matched": prior_auc_matched,
            }
        )

    blend_auc = float(roc_auc_score(y, oof_pred))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = base_sub_df.copy()
    submission[base_sub_col] = np.clip(test_pred, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": oof_pred}).to_csv(oof_path, index=False)

    summary = {
        "base_oof": str(args.base_oof),
        "base_submission": str(args.base_submission),
        "base_auc": base_auc,
        "blend_auc": blend_auc,
        "delta_vs_base": blend_auc - base_auc,
        "steps": step_summaries,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
