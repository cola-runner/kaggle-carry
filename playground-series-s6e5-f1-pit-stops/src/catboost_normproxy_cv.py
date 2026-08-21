from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


ID_COL = "id"
TARGET = "PitNextLap"
NORM_TARGET = "Normalized_TyreLife"


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {path.name for path in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return

    for zip_path in sorted(data_dir.glob("*.zip")):
        print(f"Extracting {zip_path.name} ...", flush=True)
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    existing = {path.name for path in data_dir.glob("*.csv")}
    missing = required - existing
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {sorted(missing)}")


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
    out["Driver_Race"] = out["Driver"].astype(str) + "_" + out["Race"].astype(str)
    out["Driver_Compound"] = out["Driver"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Stint"] = out["Race"].astype(str) + "_" + out["Stint"].astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def load_external(external_csv: Path, competition_feature_cols: list[str]) -> pd.DataFrame:
    external = pd.read_csv(external_csv)
    needed = competition_feature_cols + [TARGET, NORM_TARGET]
    missing = [col for col in needed if col not in external.columns]
    if missing:
        raise ValueError(f"External data is missing columns: {missing}")
    return external[needed].copy()


def categorical_columns(frames: list[pd.DataFrame], features: list[str]) -> list[str]:
    cat_cols: list[str] = []
    for col in features:
        if any(
            pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col])
            for frame in frames
        ):
            cat_cols.append(col)
    return cat_cols


def align_lgbm_categories(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    for col in cat_cols:
        categories = pd.Index(pd.concat([frame[col].astype(str) for frame in frames], ignore_index=True).unique())
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[col] = frame[col].astype(str).astype(dtype)


def make_norm_proxy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    n_splits: int,
    seed: int,
    cache_path: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float]]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        print(f"Loaded norm proxy cache: {cache_path}", flush=True)
        return (
            cached["train_norm"],
            cached["test_norm"],
            cached["external_norm"],
            float(cached["rmse"]),
            [float(x) for x in cached["fold_rmse"]],
        )

    train_norm = np.zeros(len(train))
    test_norm = np.zeros(len(test))
    external_norm = np.zeros(len(external))
    fold_rmse: list[float] = []

    train_lgb = train.copy()
    test_lgb = test.copy()
    external_lgb = external.copy()
    align_lgbm_categories([train_lgb, test_lgb, external_lgb], cat_cols)

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(external_lgb), start=1):
        print(f"[norm] fold {fold}", flush=True)
        model = LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=3000,
            learning_rate=0.025,
            num_leaves=127,
            min_child_samples=50,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=1.0,
            random_state=seed + 3000 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            external_lgb.iloc[tr_idx][features],
            external_lgb.iloc[tr_idx][NORM_TARGET],
            eval_set=[(external_lgb.iloc[va_idx][features], external_lgb.iloc[va_idx][NORM_TARGET])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
        )
        valid_pred = model.predict(external_lgb.iloc[va_idx][features])
        external_norm[va_idx] = valid_pred
        train_norm += model.predict(train_lgb[features]) / n_splits
        test_norm += model.predict(test_lgb[features]) / n_splits
        rmse = float(np.sqrt(mean_squared_error(external_lgb.iloc[va_idx][NORM_TARGET], valid_pred)))
        print(f"[norm] fold {fold} RMSE: {rmse:.8f}", flush=True)
        fold_rmse.append(float(rmse))

    overall_rmse = float(np.sqrt(mean_squared_error(external[NORM_TARGET], external_norm)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_norm=train_norm,
        test_norm=test_norm,
        external_norm=external_norm,
        rmse=np.array(overall_rmse),
        fold_rmse=np.array(fold_rmse),
    )
    print(f"[norm] overall RMSE: {overall_rmse:.8f}", flush=True)
    print(f"Saved norm proxy cache: {cache_path}", flush=True)
    return train_norm, test_norm, external_norm, float(overall_rmse), fold_rmse


def add_norm_features(df: pd.DataFrame, pred_norm: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    pred = np.clip(pred_norm, 0, 1)
    out["Pred_Normalized_TyreLife"] = pred
    out["PredNorm_x_TyreLife"] = pred * out["TyreLife"]
    out["PredNorm_x_Progress"] = pred * out["RaceProgress"]
    out["PredNorm_to_Remainder"] = pred / (1.0 - out["RaceProgress"] + 1e-3)
    out["TyreLife_minus_PredNormScaled"] = out["TyreLife"] - pred * out["TotalLaps_est"]
    out["PredNorm_bin"] = pd.cut(
        pred,
        bins=[-0.001, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.001],
        labels=["00_10", "10_20", "20_35", "35_50", "50_65", "65_80", "80_100"],
    ).astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def train_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    y: pd.Series,
    y_external: pd.Series,
    features: list[str],
    cat_cols: list[str],
    n_splits: int,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train_cat = train.copy()
    test_cat = test.copy()
    external_cat = external.copy()

    for frame in [train_cat, test_cat, external_cat]:
        for col in cat_cols:
            frame[col] = frame[col].astype(str)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_cat))
    test_pred = np.zeros(len(test_cat))
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_cat[features], y), start=1):
        print(f"[catboost] fold {fold}", flush=True)
        X_train = pd.concat([train_cat.iloc[tr_idx][features], external_cat[features]], ignore_index=True)
        y_train = pd.concat([y.iloc[tr_idx], y_external], ignore_index=True)
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.045,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed + 4000 + fold,
            od_type="Iter",
            od_wait=250,
            verbose=300,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(train_cat.iloc[va_idx][features], y.iloc[va_idx]),
            use_best_model=True,
        )
        valid_pred = model.predict_proba(train_cat.iloc[va_idx][features])[:, 1]
        oof[va_idx] = valid_pred
        test_pred += model.predict_proba(test_cat[features])[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], valid_pred)
        print(f"[catboost] fold {fold} AUC: {score:.8f}", flush=True)
        scores.append(float(score))

    return oof, test_pred, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost with learned Normalized_TyreLife proxy.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--catboost-iterations", default=3000, type=int)
    parser.add_argument("--force-norm-cache", action="store_true")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    competition_cols = [col for col in test_raw.columns if col in train_raw.columns and col != ID_COL]
    external_raw = load_external(args.external_csv, competition_cols)

    train_base = add_simple_features(train_raw)
    test_base = add_simple_features(test_raw)
    external_base = add_simple_features(external_raw)
    y = train_base[TARGET].astype(int)
    y_external = external_base[TARGET].astype(int)

    norm_features = [col for col in test_base.columns if col in train_base.columns and col in external_base.columns]
    norm_features = [col for col in norm_features if col not in [ID_COL, TARGET, NORM_TARGET]]
    norm_cat_cols = categorical_columns([train_base, test_base, external_base], norm_features)
    print(f"[norm] features: {len(norm_features)}", flush=True)
    print(f"[norm] categoricals: {norm_cat_cols}", flush=True)

    cache_path = args.output_dir / f"normproxy_seed{args.seed}_folds{args.n_splits}.npz"
    train_norm, test_norm, external_norm, norm_rmse, norm_fold_rmse = make_norm_proxy(
        train_base,
        test_base,
        external_base,
        norm_features,
        norm_cat_cols,
        args.n_splits,
        args.seed,
        cache_path,
        args.force_norm_cache,
    )

    train = add_norm_features(train_base, train_norm)
    test = add_norm_features(test_base, test_norm)
    external = add_norm_features(external_base, external_norm)
    train["Source"] = "synthetic"
    test["Source"] = "synthetic"
    external["Source"] = "external"

    features = [col for col in test.columns if col in train.columns and col not in [ID_COL, TARGET, NORM_TARGET]]
    features = list(dict.fromkeys(features))
    cat_cols = categorical_columns([train, test, external], features)
    print(f"[catboost] features: {len(features)}", flush=True)
    print(f"[catboost] categoricals: {cat_cols}", flush=True)

    oof, test_pred, fold_scores = train_catboost(
        train,
        test,
        external,
        y,
        y_external,
        features,
        cat_cols,
        args.n_splits,
        args.seed,
        args.catboost_iterations,
    )
    oof_auc = roc_auc_score(y, oof)
    print(f"[catboost] OOF AUC: {oof_auc:.8f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission_catboost_normproxy_external.csv"
    oof_path = args.output_dir / "oof_catboost_normproxy_external.csv"
    summary_path = args.output_dir / "summary_catboost_normproxy_external.json"

    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != ID_COL][0]
    submission[prediction_col] = np.clip(test_pred, 0, 1)
    submission.to_csv(submission_path, index=False)

    pd.DataFrame(
        {
            ID_COL: train_raw[ID_COL],
            TARGET: y,
            "pred": oof,
            "pred_norm": train_norm,
        }
    ).to_csv(oof_path, index=False)

    summary = {
        "norm_proxy": {
            "rmse": norm_rmse,
            "fold_rmse": norm_fold_rmse,
            "cache": str(cache_path),
        },
        "features": len(features),
        "categorical_features": cat_cols,
        "external_rows": int(len(external)),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "catboost_iterations": args.catboost_iterations,
        "fold_scores": fold_scores,
        "oof_auc": float(oof_auc),
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
