"""Pseudo-labeling for CatBoost on the simple-external feature recipe.

Pipeline:
  1. Read the test-set predictions from a prior CatBoost submission (e.g. the
     baseline at OOF AUC 0.9514).
  2. Mark every test row whose predicted probability is below `--neg-thr` as a
     pseudo-negative and every row above `--pos-thr` as a pseudo-positive.
  3. Train CatBoost on (competition train) + (external) + (pseudo-labeled test)
     with the same StratifiedKFold split as the baseline. Validation slice stays
     pure competition train, so OOF AUC remains comparable.
  4. Report OOF AUC change and write a fresh OOF + submission.
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
    parser.add_argument("--prior-submission", default="submissions/submission_final_push_v2.csv", type=Path,
                        help="CSV of test-set probabilities to use for pseudo-labels.")
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--iterations", default=2500, type=int)
    parser.add_argument("--pos-thr", default=0.90, type=float)
    parser.add_argument("--neg-thr", default=0.04, type=float)
    parser.add_argument("--pseudo-weight", default=0.6, type=float,
                        help="Sample weight for pseudo-labeled rows vs 1.0 for real labels.")
    parser.add_argument("--suffix", default="pseudo_v1")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    external_raw = pd.read_csv(args.external_csv)

    prior = pd.read_csv(args.prior_submission)
    pred_col_prior = [c for c in prior.columns if c != ID_COL][0]
    prior_map = prior.set_index(ID_COL)[pred_col_prior]
    test_raw["_prior"] = test_raw[ID_COL].map(prior_map).astype(float)
    assert test_raw["_prior"].notna().all(), "prior submission missing some test ids"

    pos_mask = test_raw["_prior"] >= args.pos_thr
    neg_mask = test_raw["_prior"] <= args.neg_thr
    pseudo_pos = int(pos_mask.sum())
    pseudo_neg = int(neg_mask.sum())
    print(f"Pseudo-positives at >= {args.pos_thr}: {pseudo_pos}", flush=True)
    print(f"Pseudo-negatives at <= {args.neg_thr}: {pseudo_neg}", flush=True)
    print(f"Pseudo rows / total test: {(pseudo_pos+pseudo_neg)/len(test_raw):.3f}", flush=True)

    pseudo_rows = test_raw.loc[pos_mask | neg_mask].copy()
    pseudo_rows[TARGET] = (pseudo_rows["_prior"] >= args.pos_thr).astype(int)
    pseudo_rows = pseudo_rows.drop(columns=["_prior"])

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw.drop(columns=["_prior"]))
    external = add_simple_features(external_raw)
    pseudo = add_simple_features(pseudo_rows)

    y = train[TARGET].astype(int).to_numpy()
    y_ext = external[TARGET].astype(int).to_numpy()
    y_pseudo = pseudo[TARGET].astype(int).to_numpy()

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    for frame in (train, test, external, pseudo):
        for c in cat_cols:
            frame[c] = frame[c].fillna("_missing_").astype(str).replace({"nan": "_missing_"})
    print(f"features: {len(feature_cols)}, cat: {cat_cols}", flush=True)

    X = train[feature_cols]
    X_test = test[feature_cols]
    X_external = external[feature_cols]
    X_pseudo = pseudo[feature_cols]

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    fold_scores: list[float] = []

    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        # Build training pool: competition train_fold + external + pseudo labels.
        X_train = pd.concat([X.iloc[tri], X_external, X_pseudo], ignore_index=True)
        y_train = np.concatenate([y[tri], y_ext, y_pseudo])
        weight = np.concatenate([
            np.ones(len(tri)),
            np.ones(len(X_external)),
            np.full(len(X_pseudo), args.pseudo_weight),
        ])

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
            sample_weight=weight,
            eval_set=(X.iloc[vai], y[vai]),
            use_best_model=True,
        )
        valid_pred = model.predict_proba(X.iloc[vai])[:, 1]
        oof[vai] = valid_pred
        test_pred += model.predict_proba(X_test)[:, 1] / args.n_splits
        score = float(roc_auc_score(y[vai], valid_pred))
        fold_scores.append(score)
        print(f"[fold {fold}]  AUC={score:.6f}  t={time.time()-t0:.1f}s  best_iter={model.tree_count_}  n_train={len(X_train)}", flush=True)

    overall = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {overall:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_catboost_pseudo_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_catboost_pseudo_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_catboost_pseudo_{args.suffix}.json"

    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub[pred_col] = np.clip(test_pred, 0.0, 1.0)
    sub.to_csv(sub_path, index=False)

    summary = {
        "oof_auc": overall,
        "fold_scores": fold_scores,
        "iterations_cap": args.iterations,
        "pseudo_pos": pseudo_pos,
        "pseudo_neg": pseudo_neg,
        "pos_thr": args.pos_thr,
        "neg_thr": args.neg_thr,
        "pseudo_weight": args.pseudo_weight,
        "prior_submission": str(args.prior_submission),
        "outputs": {"oof": str(oof_path), "submission": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
