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


def apply_prior(base: np.ndarray, prior: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(prior)
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * prior[mask]
    return out, mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend a tiny external exact-key prior into an existing OOF/submission.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--base-oof", required=True, type=Path)
    parser.add_argument("--base-submission", required=True, type=Path)
    parser.add_argument("--keys", default="Race,Year,LapNumber,Position,Compound")
    parser.add_argument("--alpha", default=0.00525, type=float)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="external_key_prior_blend")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    base_oof_df = pd.read_csv(args.base_oof)
    base_sub_df = pd.read_csv(args.base_submission)
    base_oof_col = prediction_column(base_oof_df)
    base_sub_col = prediction_column(base_sub_df)

    y = train[TARGET].astype(int).to_numpy()
    base_oof = base_oof_df[base_oof_col].astype(float).to_numpy()
    base_test = base_sub_df[base_sub_col].astype(float).to_numpy()
    keys = [key.strip() for key in args.keys.split(",") if key.strip()]

    stats = external.groupby(keys, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
    train_prior = train.merge(stats, on=keys, how="left")["mean"].to_numpy()
    test_prior = test.merge(stats, on=keys, how="left")["mean"].to_numpy()

    blended_oof, train_mask = apply_prior(base_oof, train_prior, args.alpha)
    blended_test, test_mask = apply_prior(base_test, test_prior, args.alpha)

    base_auc = float(roc_auc_score(y, base_oof))
    blend_auc = float(roc_auc_score(y, blended_oof))
    prior_auc_matched = None
    if train_mask.any() and np.unique(y[train_mask]).size > 1:
        prior_auc_matched = float(roc_auc_score(y[train_mask], train_prior[train_mask]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = base_sub_df.copy()
    submission[base_sub_col] = np.clip(blended_test, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": blended_oof}).to_csv(oof_path, index=False)

    summary = {
        "base_oof": str(args.base_oof),
        "base_submission": str(args.base_submission),
        "keys": keys,
        "alpha": args.alpha,
        "base_auc": base_auc,
        "blend_auc": blend_auc,
        "delta_vs_base": blend_auc - base_auc,
        "train_match_rate": float(train_mask.mean()),
        "test_match_rate": float(test_mask.mean()),
        "prior_auc_matched": prior_auc_matched,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
