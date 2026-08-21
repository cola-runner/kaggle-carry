"""pytabkit RealMLP_TD_Classifier with yekenot's exact hyperparameters + FE.

Reference: `kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit` (OOF 0.9540).
We use the same FE recipe + the same `params` dict, the only knob to dial down
for our compute budget is `n_ens` (yekenot = 24; default here = 12 to fit our
MPS budget; --n-ens 8 if even faster needed).

Aligned to StratifiedKFold(seed=2026) so OOF stacks with our other own-trained
models. After fold 1, prints AUC immediately — if < `--min-fold1-auc`, the
script exits without running the remaining folds (saves time on a broken run).
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
import torch
from pytabkit import RealMLP_TD_Classifier
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


def build_params(n_ens: int, random_state: int) -> dict:
    """yekenot's exact RealMLP_TD_Classifier hyperparameters."""
    return {
        "random_state": random_state,
        "verbosity": 1,
        "val_metric_name": "1-auc_ovr",
        "n_ens": n_ens,
        "n_epochs": 6,
        "batch_size": 256,
        "use_early_stopping": False,
        "early_stopping_additive_patience": 10,
        "early_stopping_multiplicative_patience": 1,
        "lr": 0.01,
        "wd": 0.016,
        "sq_mom": 0.99,
        "lr_sched": "lin_cos_log_15",
        "first_layer_lr_factor": 0.25,
        "embedding_size": 6,
        "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128],
        "act": "silu",
        "p_drop": 0.05,
        "p_drop_sched": "invsqrtp1e-3",
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_act_name": "gelu",
        "plr_lr_factor": 0.1151,
        "plr_sigma": 2.33,
        "ls_eps": 0.01,
        "ls_eps_sched": "sqrt_cos",
        "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": ["one_hot", "median_center", "robust_scale",
                 "smooth_clip", "embedding", "l2_normalize"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--n-ens", default=12, type=int, help="ensemble members per fold; yekenot=24")
    parser.add_argument("--min-fold1-auc", default=0.950, type=float,
                        help="Abort the run if fold 1 OOF AUC is below this.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}", flush=True)

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
    combo_names = fe.combo_names
    print(f"FE done: X={X.shape}  X_test={X_test.shape}  X_orig={X_orig.shape}", flush=True)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"cat cols: {len(cat_cols)}", flush=True)

    for c in cat_cols:
        for d in (X, X_test, X_orig):
            d[c] = d[c].fillna("_missing_").astype(str)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    skf_orig = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_splits_orig = list(skf_orig.split(X_orig, y_orig))

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_summaries = []

    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        print(f"\n=== fold {fold}/{args.n_splits} ===", flush=True)
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

        params = build_params(args.n_ens, random_state=args.seed + fold)
        print(f"params n_ens={params['n_ens']}", flush=True)
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        v_pred = model.predict_proba(X_va)[:, 1]
        t_pred = model.predict_proba(X_te)[:, 1]
        oof[vai] = v_pred
        test_pred += t_pred / args.n_splits
        auc = float(roc_auc_score(y_va, v_pred))
        secs = time.time() - t0
        print(f"[fold {fold}] AUC={auc:.6f}  t={secs:.1f}s", flush=True)
        fold_summaries.append({"fold": fold, "auc": auc, "seconds": secs})

        if fold == 1 and auc < args.min_fold1_auc:
            print(f"!! fold-1 AUC {auc:.6f} below threshold {args.min_fold1_auc} — aborting run.", flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / f"summary_pytabkit_yekenot_{args.suffix}_ABORTED.json").write_text(
                json.dumps({"fold1_auc": auc, "threshold": args.min_fold1_auc, "fold_summaries": fold_summaries}, indent=2),
                encoding="utf-8",
            )
            sys.exit(0)

    overall = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {overall:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_pytabkit_yekenot_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_pytabkit_yekenot_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_pytabkit_yekenot_{args.suffix}.json"
    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pc = [c for c in sub.columns if c != ID_COL][0]
    sub[pc] = np.clip(test_pred, 0, 1)
    sub.to_csv(sub_path, index=False)
    summary = {
        "oof_auc": overall, "n_ens": args.n_ens, "device": device, "seed": args.seed,
        "fold_summaries": fold_summaries,
        "outputs": {"oof": str(oof_path), "sub": str(sub_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": overall, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
