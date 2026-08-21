"""AutoGluon-Tabular as a fresh independent stack member.

Public notebooks for this competition use plain CatBoost/LGB/XGB/TabM/MLP. None
that I scanned use AutoGluon. AutoGluon's internal bagged stack (multiple model
families with stacked meta) tends to be decorrelated from any single boosting
family and is a strong candidate for a "new" ensemble member.

Outputs (so it can be blended exactly like our other OOFs):
  submissions/oof_autogluon_<suffix>.csv         (id, PitNextLap, pred)
  submissions/submission_autogluon_<suffix>.csv  (id, PitNextLap)
  submissions/summary_autogluon_<suffix>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import roc_auc_score


ID_COL = "id"
TARGET = "PitNextLap"


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {p.name for p in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return
    for zip_path in sorted(data_dir.glob("*.zip")):
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(columns={"LapTime (s)": "LapTime_s"})
    total_laps = out["LapNumber"] / out["RaceProgress"].replace(0, np.nan)
    out["TotalLaps_est"] = total_laps.clip(1, 120)
    out["LapsRemaining_est"] = out["TotalLaps_est"] - out["LapNumber"]
    out["TyreLife_frac_race"] = out["TyreLife"] / out["TotalLaps_est"].replace(0, np.nan)
    out["TyreLife_frac_lap"] = out["TyreLife"] / out["LapNumber"].replace(0, np.nan)
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1)
    out["Delta_per_TyreLife"] = out["LapTime_Delta"] / (out["TyreLife"] + 1)
    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["Compound_Stint"] = out["Compound"].astype(str) + "_" + out["Stint"].astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--model-dir", default="ag_models", type=Path)
    parser.add_argument("--time-limit", default=10800, type=int, help="Total training time budget in seconds (default 3h).")
    parser.add_argument("--preset", default="high_quality", choices=["medium_quality","good_quality","high_quality","best_quality"])
    parser.add_argument("--num-bag-folds", default=5, type=int)
    parser.add_argument("--num-stack-levels", default=1, type=int)
    parser.add_argument("--use-external", action="store_true")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    if args.use_external and args.external_csv.exists():
        external = add_simple_features(pd.read_csv(args.external_csv))
        external = external[train.columns]
        train_combined = pd.concat([train, external], ignore_index=True)
        print(f"merged external: {len(external)} extra rows -> total {len(train_combined)}", flush=True)
    else:
        train_combined = train

    # Drop id from training input; keep target.
    train_input = train_combined.drop(columns=[ID_COL])
    test_input = test.drop(columns=[ID_COL])

    train_data = TabularDataset(train_input)
    test_data = TabularDataset(test_input)
    print(f"train_data: {train_data.shape}, test_data: {test_data.shape}", flush=True)
    print(f"preset: {args.preset}, time_limit: {args.time_limit}s, bag_folds: {args.num_bag_folds}, stack_levels: {args.num_stack_levels}", flush=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric="roc_auc",
        path=str(args.model_dir),
        problem_type="binary",
    ).fit(
        train_data=train_data,
        time_limit=args.time_limit,
        presets=args.preset,
        num_bag_folds=args.num_bag_folds,
        num_stack_levels=args.num_stack_levels,
        verbosity=2,
    )
    print(f"AutoGluon fit done in {time.time()-t0:.1f}s", flush=True)

    # OOF predictions on train_combined (positional). We only keep the first len(train) rows.
    oof_full = predictor.predict_proba_oof()
    if isinstance(oof_full, pd.DataFrame):
        # binary: get column for class 1
        if 1 in oof_full.columns: pcol = 1
        elif "1" in oof_full.columns: pcol = "1"
        else: pcol = oof_full.columns[-1]
        oof_full = oof_full[pcol]
    oof_train = oof_full.iloc[:len(train)].to_numpy() if hasattr(oof_full, 'iloc') else np.asarray(oof_full)[:len(train)]
    y = train[TARGET].astype(int).to_numpy()
    oof_auc = float(roc_auc_score(y, oof_train))
    print(f"OOF AUC (train portion only): {oof_auc:.6f}", flush=True)

    # Test predictions
    test_proba = predictor.predict_proba(test_data)
    if isinstance(test_proba, pd.DataFrame):
        if 1 in test_proba.columns: pcol = 1
        elif "1" in test_proba.columns: pcol = "1"
        else: pcol = test_proba.columns[-1]
        test_pred = test_proba[pcol].to_numpy()
    else:
        test_pred = np.asarray(test_proba)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_autogluon_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_autogluon_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_autogluon_{args.suffix}.json"
    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof_train}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub[pred_col] = test_pred
    sub.to_csv(sub_path, index=False)

    try:
        lb = predictor.leaderboard(silent=True).to_dict(orient="records")
    except Exception:
        lb = None
    summary = {
        "preset": args.preset,
        "time_limit": args.time_limit,
        "num_bag_folds": args.num_bag_folds,
        "num_stack_levels": args.num_stack_levels,
        "use_external": args.use_external,
        "train_rows": int(len(train_combined)),
        "oof_auc_train_portion": oof_auc,
        "fit_seconds": time.time() - t0,
        "leaderboard": lb,
        "outputs": {"oof": str(oof_path), "submission": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"oof_auc": oof_auc, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
