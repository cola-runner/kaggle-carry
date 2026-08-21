from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost_normproxy_cv import (
    ID_COL,
    NORM_TARGET,
    TARGET,
    add_norm_features,
    add_simple_features,
    align_lgbm_categories,
    categorical_columns,
    extract_zip_if_needed,
    load_external,
)


TE_KEYS = [
    ["Race", "Year", "Compound", "Stint", "TyreLife"],
    ["Race", "Year", "Compound", "Stint", "TyreLife", "PitStop"],
    ["Race", "Year", "Stint", "TyreLife"],
    ["Race", "Year", "Compound", "TyreLife"],
    ["Race", "Year", "Compound", "Stint", "RaceProgress"],
    ["Race", "Year", "Compound", "Stint", "LapNumber"],
    ["Race", "Year", "TyreLife"],
    ["Compound", "Stint", "TyreLife"],
]


def te_name(keys: list[str], smooth: float) -> str:
    suffix = str(smooth).replace(".", "p")
    return "te_" + "_".join(keys) + f"_s{suffix}"


def encode_target_mean(
    source: pd.DataFrame,
    source_y: np.ndarray,
    target: pd.DataFrame,
    keys: list[str],
    smooth: float,
) -> np.ndarray:
    global_mean = float(np.mean(source_y))
    stats_frame = source[keys].copy()
    stats_frame["_target"] = source_y
    stats = stats_frame.groupby(keys, dropna=False)["_target"].agg(["sum", "count"])
    encoded = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
    target_index = pd.MultiIndex.from_frame(target[keys])
    values = encoded.reindex(target_index).to_numpy()
    return np.nan_to_num(values, nan=global_mean)


def add_fold_target_encoding(
    fold_train: pd.DataFrame,
    fold_valid: pd.DataFrame,
    fold_test: pd.DataFrame,
    external: pd.DataFrame,
    y_fold_train: np.ndarray,
    y_external: np.ndarray,
    smooth_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    source = pd.concat([fold_train, external], ignore_index=True)
    source_y = np.concatenate([y_fold_train, y_external])

    X_train = fold_train.copy()
    X_valid = fold_valid.copy()
    X_test = fold_test.copy()
    X_external = external.copy()
    te_cols: list[str] = []

    for smooth in smooth_values:
        for keys in TE_KEYS:
            col = te_name(keys, smooth)
            X_train[col] = encode_target_mean(source, source_y, fold_train, keys, smooth)
            X_valid[col] = encode_target_mean(source, source_y, fold_valid, keys, smooth)
            X_test[col] = encode_target_mean(source, source_y, fold_test, keys, smooth)
            X_external[col] = encode_target_mean(source, source_y, external, keys, smooth)
            te_cols.append(col)

    return X_train, X_valid, X_test, X_external, te_cols


def load_norm_cache(cache_path: Path, expected: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing norm proxy cache: {cache_path}")
    cached = np.load(cache_path)
    train_norm = cached["train_norm"]
    test_norm = cached["test_norm"]
    external_norm = cached["external_norm"]
    if (len(train_norm), len(test_norm), len(external_norm)) != expected:
        raise ValueError("Norm proxy cache row counts do not match current data.")
    return train_norm, test_norm, external_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM with fold-safe target encodings.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--norm-cache", default="submissions/normproxy_seed2026_folds5.npz", type=Path)
    parser.add_argument("--no-norm-proxy", action="store_true")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--smooth", default="5,20", help="Comma-separated smoothing values for target encoding.")
    parser.add_argument("--n-estimators", default=8000, type=int)
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    competition_cols = [col for col in test_raw.columns if col in train_raw.columns and col != ID_COL]
    external_raw = load_external(args.external_csv, competition_cols)

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    external = add_simple_features(external_raw)
    y = train[TARGET].astype(int).to_numpy()
    y_external = external[TARGET].astype(int).to_numpy()

    if not args.no_norm_proxy:
        norm_values = load_norm_cache(args.norm_cache, (len(train), len(test), len(external)))
        train = add_norm_features(train, norm_values[0])
        test = add_norm_features(test, norm_values[1])
        external = add_norm_features(external, norm_values[2])

    train["Source"] = "synthetic"
    test["Source"] = "synthetic"
    external["Source"] = "external"

    features = [col for col in test.columns if col in train.columns and col not in [ID_COL, TARGET, NORM_TARGET]]
    features = list(dict.fromkeys(features))
    cat_cols = categorical_columns([train, test, external], features)
    align_lgbm_categories([train, test, external], cat_cols)
    smooth_values = [float(value) for value in args.smooth.split(",") if value.strip()]

    print(f"Base features: {len(features)}", flush=True)
    print(f"Categoricals: {cat_cols}", flush=True)
    print(f"TE columns per fold: {len(TE_KEYS) * len(smooth_values)}", flush=True)

    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train[features], y), start=1):
        print(f"[lgbm_te] fold {fold}", flush=True)
        fold_train = train.iloc[tr_idx][features]
        fold_valid = train.iloc[va_idx][features]
        fold_test = test[features]
        fold_external = external[features]

        X_train, X_valid, X_test, X_external, te_cols = add_fold_target_encoding(
            fold_train,
            fold_valid,
            fold_test,
            fold_external,
            y[tr_idx],
            y_external,
            smooth_values,
        )
        X_model = pd.concat([X_train, X_external], ignore_index=True)
        y_model = np.concatenate([y[tr_idx], y_external])
        model_features = features + te_cols

        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=args.n_estimators,
            learning_rate=0.02,
            num_leaves=95,
            min_child_samples=70,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.03,
            reg_lambda=1.2,
            random_state=args.seed + 5000 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_model[model_features],
            y_model,
            eval_set=[(X_valid[model_features], y[va_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(250), lgb.log_evaluation(500)],
        )
        valid_pred = model.predict_proba(X_valid[model_features])[:, 1]
        oof[va_idx] = valid_pred
        test_pred += model.predict_proba(X_test[model_features])[:, 1] / args.n_splits
        score = roc_auc_score(y[va_idx], valid_pred)
        print(f"[lgbm_te] fold {fold} AUC: {score:.8f}", flush=True)
        fold_scores.append(float(score))

    oof_auc = roc_auc_score(y, oof)
    print(f"[lgbm_te] OOF AUC: {oof_auc:.8f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "norm" if not args.no_norm_proxy else "nonorm"
    smooth_suffix = args.smooth.replace(",", "_").replace(".", "p")
    submission_path = args.output_dir / f"submission_lgbm_te_{suffix}_s{smooth_suffix}.csv"
    oof_path = args.output_dir / f"oof_lgbm_te_{suffix}_s{smooth_suffix}.csv"
    summary_path = args.output_dir / f"summary_lgbm_te_{suffix}_s{smooth_suffix}.json"

    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != ID_COL][0]
    submission[prediction_col] = np.clip(test_pred, 0, 1)
    submission.to_csv(submission_path, index=False)

    pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y, "pred": oof}).to_csv(oof_path, index=False)

    summary = {
        "base_features": len(features),
        "target_encoding_features": len(TE_KEYS) * len(smooth_values),
        "target_encoding_keys": TE_KEYS,
        "smooth_values": smooth_values,
        "categorical_features": cat_cols,
        "external_rows": int(len(external)),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "fold_scores": fold_scores,
        "oof_auc": float(oof_auc),
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
