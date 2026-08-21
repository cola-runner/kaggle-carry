"""pytabkit RealMLP_TD_Classifier — our own NN member, aligned to existing fold scheme.

pytabkit's RealMLP is what Mikhail Naumov (LB 0.95475 / top 4) uses. Public
notebooks that use pytabkit (yekenot, masayakawamata) hit OOF 0.950-0.953. We
train our own version with:
  - StratifiedKFold(seed=2026, 5 folds) — matches public_blend_v1 so OOFs line up.
  - External F1 strategy dataset concatenated into each training fold (val stays
    pure competition train so OOF is honest).
  - Same simple feature recipe as our other members for direct comparability.

Outputs:
  submissions/oof_pytabkit_realmlp_<suffix>.csv         (id, PitNextLap, pred)
  submissions/submission_pytabkit_realmlp_<suffix>.csv  (id, PitNextLap)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
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


def prepare(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].fillna("_missing_").astype(str)
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out[cat_cols + num_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--n-cv", default=5, type=int, help="pytabkit n_cv (internal CV within each fold).")
    parser.add_argument("--n-refit", default=0, type=int)
    parser.add_argument("--use-external", action="store_true")
    parser.add_argument("--device", default="auto",
                        help="'mps', 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    external_raw = pd.read_csv(args.external_csv) if args.use_external and args.external_csv.exists() else None

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    if external_raw is not None:
        external = add_simple_features(external_raw)
    else:
        external = None

    y = train[TARGET].astype(int).to_numpy()
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"features: {len(feature_cols)} ({len(cat_cols)} cat, {len(num_cols)} num)", flush=True)

    train_prepped = prepare(train, cat_cols, num_cols)
    test_prepped = prepare(test, cat_cols, num_cols)
    if external is not None:
        external_prepped = prepare(external, cat_cols, num_cols)
        y_ext = external[TARGET].astype(int).to_numpy()
    else:
        external_prepped = None
        y_ext = None

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_summaries = []

    for fold, (tri, vai) in enumerate(skf.split(train_prepped, y), start=1):
        t0 = time.time()
        print(f"\n=== fold {fold}/{args.n_splits} ===", flush=True)

        X_tr = train_prepped.iloc[tri].reset_index(drop=True)
        y_tr = y[tri]
        if external_prepped is not None:
            X_tr = pd.concat([X_tr, external_prepped], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_ext])

        X_va = train_prepped.iloc[vai].reset_index(drop=True)
        y_va = y[vai]

        clf = RealMLP_TD_Classifier(
            n_cv=args.n_cv,
            n_refit=args.n_refit,
            val_metric_name="cross_entropy",
            random_state=args.seed + fold,
            device=args.device,
            verbosity=1,
        )
        clf.fit(X_tr, y_tr, cat_col_names=cat_cols, val_idxs=None)

        val_pred = clf.predict_proba(X_va)[:, 1]
        fold_test_pred = clf.predict_proba(test_prepped)[:, 1]

        oof[vai] = val_pred
        test_pred += fold_test_pred / args.n_splits
        auc = float(roc_auc_score(y_va, val_pred))
        print(f"[fold {fold}] AUC={auc:.6f}  t={time.time()-t0:.1f}s", flush=True)
        fold_summaries.append({"fold": fold, "auc": auc, "seconds": time.time() - t0})

    overall = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {overall:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_pytabkit_realmlp_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_pytabkit_realmlp_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_pytabkit_realmlp_{args.suffix}.json"

    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pc = [c for c in sub.columns if c != ID_COL][0]
    sub[pc] = np.clip(test_pred, 0.0, 1.0)
    sub.to_csv(sub_path, index=False)

    summary = {
        "n_splits": args.n_splits, "seed": args.seed,
        "n_cv": args.n_cv, "n_refit": args.n_refit,
        "use_external": args.use_external,
        "oof_auc": overall, "fold_summaries": fold_summaries,
        "outputs": {"oof": str(oof_path), "submission": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": overall, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
