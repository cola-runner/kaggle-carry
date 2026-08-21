"""CatBoost with two genuinely new signal sources nobody in public uses:

1. Track physical properties (`trk_*`): pit-lane loss, abrasion rating, downforce
   level, DRS zones, typical pit-stops-per-race, etc. From hand-curated F1 data.
2. Neighborhood features (`nbr_*`): does the lap N+1/N-1/N+2/... row exist in
   train+test for this (Year, Race, Driver)? Plus per-group rank/coverage stats.
   These require concatenating train+test which standard pipelines skip.

Same StratifiedKFold(seed=2026, 5-fold) as the GBDT baseline so OOF blends
directly with my CatBoost + public OOFs.
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
from sklearn.model_selection import StratifiedKFold

from track_features import attach_track_features, TRACK_COLS


ID_COL = "id"
TARGET = "PitNextLap"
GRP = ["Year", "Race", "Driver"]


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


def add_neighborhood_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train2 = train.copy()
    test2 = test.copy()
    train2["_split"] = "tr"
    test2["_split"] = "te"
    all_df = pd.concat([train2, test2], ignore_index=True)

    # Per-group counts and rank
    grp_counts = all_df.groupby(GRP).size().rename("nbr_grp_count")
    all_df = all_df.merge(grp_counts.reset_index(), on=GRP, how="left")
    all_df = all_df.sort_values(GRP + ["LapNumber"])
    all_df["nbr_rank_in_grp"] = all_df.groupby(GRP).cumcount()
    all_df["nbr_rank_frac"] = all_df["nbr_rank_in_grp"] / (all_df["nbr_grp_count"] - 1).replace(0, np.nan)
    all_df = all_df.sort_values(ID_COL).reset_index(drop=True)

    # Existence of LapNumber+/-k neighbors
    key = list(zip(*[all_df[c].astype(str).to_numpy() for c in GRP], all_df["LapNumber"].to_numpy()))
    key_set = set(key)

    def has_off(off):
        return np.fromiter(
            ((y, r, d, l + off) in key_set for (y, r, d, l) in key),
            dtype=np.int8, count=len(key),
        )

    for off, name in [(1, "nbr_has_next1"), (-1, "nbr_has_lag1"),
                      (2, "nbr_has_next2"), (-2, "nbr_has_lag2"),
                      (3, "nbr_has_next3"), (-3, "nbr_has_lag3")]:
        all_df[name] = has_off(off).astype(int)

    # Distance to nearest neighbor in either direction (smaller = denser data)
    all_df["nbr_neighbor_density"] = (
        all_df["nbr_has_next1"] + all_df["nbr_has_lag1"]
        + 0.5 * (all_df["nbr_has_next2"] + all_df["nbr_has_lag2"])
        + 0.33 * (all_df["nbr_has_next3"] + all_df["nbr_has_lag3"])
    )

    out_train = all_df[all_df["_split"] == "tr"].drop(columns=["_split"]).reset_index(drop=True)
    out_test = all_df[all_df["_split"] == "te"].drop(columns=["_split"]).reset_index(drop=True)
    return out_train, out_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--iterations", default=2500, type=int)
    parser.add_argument("--use-external", action="store_true")
    parser.add_argument("--use-track", action="store_true")
    parser.add_argument("--use-neighborhood", action="store_true")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)

    if args.use_track:
        train = attach_track_features(train)
        test = attach_track_features(test)
        print(f"+ {len(TRACK_COLS)+3} track features", flush=True)
    if args.use_neighborhood:
        train, test = add_neighborhood_features(train, test)
        # Realign with id order
        train = train.sort_values(ID_COL).reset_index(drop=True)
        test = test.sort_values(ID_COL).reset_index(drop=True)
        print(f"+ neighborhood features", flush=True)

    y = train[TARGET].astype(int).to_numpy()

    if args.use_external and args.external_csv.exists():
        ext = pd.read_csv(args.external_csv)
        ext = add_simple_features(ext)
        if args.use_track:
            ext = attach_track_features(ext)
        if args.use_neighborhood:
            # external rows don't have neighbor info from train+test; fill with 0/global mean
            for c in [c for c in train.columns if c.startswith("nbr_")]:
                if c not in ext.columns:
                    ext[c] = 0 if c.startswith("nbr_has") else float(train[c].median())
        external = ext[[c for c in train.columns if c != ID_COL]].copy()
        y_ext = external[TARGET].astype(int).to_numpy()
        external = external.drop(columns=[TARGET])
        print(f"external rows: {len(external)}", flush=True)
    else:
        external = None
        y_ext = None

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    for c in cat_cols:
        train[c] = train[c].fillna("_missing_").astype(str).replace({"nan": "_missing_"})
        test[c] = test[c].fillna("_missing_").astype(str).replace({"nan": "_missing_"})
        if external is not None:
            external[c] = external[c].fillna("_missing_").astype(str).replace({"nan": "_missing_"})
    X = train[feature_cols]
    X_test = test[feature_cols]
    if external is not None:
        X_external = external[feature_cols]

    print(f"features={len(feature_cols)}, cats={len(cat_cols)}", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    fold_scores = []

    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        if external is not None:
            X_train = pd.concat([X.iloc[tri], X_external], ignore_index=True)
            y_train = np.concatenate([y[tri], y_ext])
        else:
            X_train = X.iloc[tri]
            y_train = y[tri]

        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.seed + fold,
            od_type="Iter",
            od_wait=200,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            X_train, y_train,
            cat_features=cat_cols,
            eval_set=(X.iloc[vai], y[vai]),
            use_best_model=True,
        )
        valid_pred = model.predict_proba(X.iloc[vai])[:, 1]
        oof[vai] = valid_pred
        test_pred += model.predict_proba(X_test)[:, 1] / args.n_splits
        s = float(roc_auc_score(y[vai], valid_pred))
        fold_scores.append(s)
        print(f"[fold {fold}] AUC={s:.6f}  t={time.time()-t0:.1f}s  best_iter={model.tree_count_}", flush=True)

    overall = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {overall:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_catboost_trknbr_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_catboost_trknbr_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_catboost_trknbr_{args.suffix}.json"
    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pc = [c for c in sub.columns if c != ID_COL][0]
    sub[pc] = np.clip(test_pred, 0, 1)
    sub.to_csv(sub_path, index=False)

    summary = {
        "use_external": args.use_external, "use_track": args.use_track, "use_neighborhood": args.use_neighborhood,
        "n_features": len(feature_cols), "n_cats": len(cat_cols), "oof_auc": overall, "fold_scores": fold_scores,
        "outputs": {"oof": str(oof_path), "submission": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
