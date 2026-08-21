"""Side-by-side StratifiedKFold vs GroupKFold sanity probe.

Runs a single CatBoost model on the same feature recipe as `simple_external_cv.py`
(minus the external augmentation) under two different fold schemes and reports
the OOF AUC. Big OOF drop under group splits => the current pipeline's CV is
optimistic and meta-stack gains may be fitting fold leakage rather than signal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


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


def run_one(
    cv_name: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    iterations: int,
    seed: int,
) -> dict:
    print(f"\n========== {cv_name} ==========")
    oof = np.zeros(len(y))
    fold_scores: list[float] = []
    fold_pos_rates: list[tuple[float, float]] = []

    for fold, (tri, vai) in enumerate(splits, start=1):
        t0 = time.time()
        train_pos = float(y[tri].mean())
        valid_pos = float(y[vai].mean())
        fold_pos_rates.append((train_pos, valid_pos))
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed + fold,
            od_type="Iter",
            od_wait=200,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            X.iloc[tri],
            y[tri],
            cat_features=cat_cols,
            eval_set=(X.iloc[vai], y[vai]),
            use_best_model=True,
        )
        pred = model.predict_proba(X.iloc[vai])[:, 1]
        oof[vai] = pred
        score = float(roc_auc_score(y[vai], pred))
        fold_scores.append(score)
        print(
            f"[{cv_name}] fold {fold}  AUC={score:.6f}  "
            f"train_pos={train_pos:.4f}  valid_pos={valid_pos:.4f}  "
            f"t={time.time()-t0:.1f}s  best_iter={model.tree_count_}"
        )

    overall = float(roc_auc_score(y, oof))
    print(f"[{cv_name}] OOF AUC: {overall:.6f}")
    return {
        "cv": cv_name,
        "oof_auc": overall,
        "fold_scores": fold_scores,
        "fold_pos_rates": fold_pos_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="StratifiedKFold vs GroupKFold sanity probe.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--iterations", default=1500, type=int,
                        help="Fewer iterations than the full pipeline; we want a fast probe.")
    parser.add_argument("--group-col", default="Year_Race",
                        help="Group key for GroupKFold. Use 'Year_Race' or 'Year_Race_Driver'.")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    print(f"train shape: {train_raw.shape}", flush=True)

    train = add_simple_features(train_raw)
    y = train[TARGET].astype(int).to_numpy()

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    for c in cat_cols:
        train[c] = train[c].astype(str)
    X = train[feature_cols]

    print(f"features: {len(feature_cols)}, categorical: {cat_cols}", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    skf_splits = list(skf.split(X, y))

    if args.group_col == "Year_Race":
        groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).to_numpy()
    elif args.group_col == "Year_Race_Driver":
        groups = (
            train["Year"].astype(str) + "_" + train["Race"].astype(str) + "_" + train["Driver"].astype(str)
        ).to_numpy()
    else:
        raise ValueError(f"Unknown --group-col {args.group_col}")

    gkf = GroupKFold(n_splits=args.n_splits)
    gkf_splits = list(gkf.split(X, y, groups=groups))

    print(f"\nStratifiedKFold: {args.n_splits} folds, random_state={args.seed}")
    print(f"GroupKFold ({args.group_col}): {args.n_splits} folds, n_groups={len(np.unique(groups))}")

    strat = run_one("StratifiedKFold", skf_splits, X, y, cat_cols, args.iterations, args.seed)
    group = run_one(f"GroupKFold({args.group_col})", gkf_splits, X, y, cat_cols, args.iterations, args.seed)

    summary = {
        "n_features": len(feature_cols),
        "cat_cols": cat_cols,
        "iterations_cap": args.iterations,
        "stratified": strat,
        "group": group,
        "delta_strat_minus_group": strat["oof_auc"] - group["oof_auc"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_cv_sanity_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "stratified_oof_auc": strat["oof_auc"],
        "group_oof_auc": group["oof_auc"],
        "delta": summary["delta_strat_minus_group"],
        "interpretation": (
            "If delta is large positive, the StratifiedKFold OOF is inflated by "
            "fold leakage and group-aware tuning is required for any meta gain to "
            "translate to LB."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
