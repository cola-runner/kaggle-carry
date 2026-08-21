"""CatBoost with yekenot's feature engineering — 5-seed averaging + external data.

Matches the StratifiedKFold(seed=2026, 5 folds) used by other own-trained members.
The yekenot FE adds floor-factorized cats, KBins bins, count encodings, interaction
keys + arithmetic interactions, and OOF-safe TargetEncoder applied inside the fold loop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from yekenot_features import YekenotFE


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int, help="StratifiedKFold seed (match other own-trained OOFs).")
    parser.add_argument("--seeds", default="2026,1337,42,7,101", help="Comma-separated CatBoost seeds to average.")
    parser.add_argument("--iterations", default=2500, type=int)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--te-smooth", default="auto")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    orig = pd.read_csv(args.external_csv).drop(columns=["Normalized_TyreLife"], errors="ignore")
    y_orig = orig[TARGET].astype(int).to_numpy()
    orig = orig.drop(columns=[TARGET])

    X_raw = train_raw.drop(columns=[ID_COL, TARGET])
    y = train_raw[TARGET].astype(int).to_numpy()
    X_test_raw = test_raw.drop(columns=[ID_COL])

    fe = YekenotFE()
    X = fe.fit_transform(X_raw)
    X_test = fe.transform(X_test_raw)
    X_orig = fe.transform(orig)
    print(f"after FE: X={X.shape}  X_test={X_test.shape}  X_orig={X_orig.shape}", flush=True)
    print(f"combo_names: {fe.combo_names}", flush=True)
    combo_names = fe.combo_names

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"cat_cols count: {len(cat_cols)}", flush=True)
    for c in cat_cols:
        for d in (X, X_test, X_orig):
            d[c] = d[c].fillna("_missing_").astype(str)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    skf_orig = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_summaries = []

    fold_splits_orig = list(skf_orig.split(X_orig, y_orig))
    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        ori_tri, _ = fold_splits_orig[fold - 1]
        X_tr = pd.concat([X.iloc[tri], X_orig.iloc[ori_tri]], ignore_index=True)
        y_tr = np.concatenate([y[tri], y_orig[ori_tri]])
        X_va = X.iloc[vai].reset_index(drop=True)
        y_va = y[vai]
        X_te = X_test.copy()

        # OOF-safe TargetEncoder on combo keys
        te = TargetEncoder(cv=args.n_splits, smooth="auto", shuffle=True, random_state=args.seed)
        tr_enc = te.fit_transform(X_tr[combo_names], y_tr)
        va_enc = te.transform(X_va[combo_names])
        te_enc = te.transform(X_te[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = tr_enc
        X_va[te_names] = va_enc
        X_te[te_names] = te_enc

        seed_valid = []
        seed_test = []
        per_seed = []
        for sd in seeds:
            model = CatBoostClassifier(
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=5.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=sd + fold,
                od_type="Iter",
                od_wait=200,
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
            )
            model.fit(
                X_tr, y_tr,
                cat_features=cat_cols,
                eval_set=(X_va, y_va),
                use_best_model=True,
            )
            v_pred = model.predict_proba(X_va)[:, 1]
            t_pred = model.predict_proba(X_te)[:, 1]
            seed_valid.append(v_pred)
            seed_test.append(t_pred)
            per_seed.append(float(roc_auc_score(y_va, v_pred)))
        avg_valid = np.mean(seed_valid, axis=0)
        avg_test = np.mean(seed_test, axis=0)
        oof[vai] = avg_valid
        test_pred += avg_test / args.n_splits
        auc = float(roc_auc_score(y_va, avg_valid))
        print(f"[fold {fold}] avg AUC={auc:.6f}  per-seed: {per_seed}  t={time.time()-t0:.1f}s", flush=True)
        fold_summaries.append({"fold": fold, "avg_auc": auc, "per_seed": per_seed, "seconds": time.time() - t0})

    overall = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {overall:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_catboost_yekenot_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_catboost_yekenot_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_catboost_yekenot_{args.suffix}.json"

    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pc = [c for c in sub.columns if c != ID_COL][0]
    sub[pc] = np.clip(test_pred, 0, 1)
    sub.to_csv(sub_path, index=False)

    summary = {
        "oof_auc": overall, "seeds": seeds, "n_splits": args.n_splits,
        "fold_summaries": fold_summaries,
        "outputs": {"oof": str(oof_path), "sub": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": overall, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
