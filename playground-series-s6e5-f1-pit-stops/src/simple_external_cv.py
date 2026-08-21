from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COL = "id"
TARGET = "PitNextLap"


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
    return out.replace([np.inf, -np.inf], np.nan)


def load_external(external_csv: Path, competition_feature_cols: list[str]) -> pd.DataFrame:
    external = pd.read_csv(external_csv)
    needed = competition_feature_cols + [TARGET]
    missing = [col for col in needed if col not in external.columns]
    if missing:
        raise ValueError(f"External data is missing columns: {missing}")
    return external[needed].copy()


def prepare_frames(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    external_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    external = add_simple_features(external_raw)

    train["Source"] = "synthetic"
    test["Source"] = "synthetic"
    external["Source"] = "external"

    y = train[TARGET].astype(int)
    y_external = external[TARGET].astype(int)
    features = [col for col in test.columns if col in train.columns and col not in [ID_COL, TARGET]]
    features = list(dict.fromkeys(features))
    cat_cols = [col for col in features if train[col].dtype == "object" or col == "Source"]
    return train, test, external, y, y_external, features, cat_cols


def align_categories(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    for col in cat_cols:
        categories = pd.Index(pd.concat([frame[col].astype(str) for frame in frames], ignore_index=True).unique())
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[col] = frame[col].astype(str).astype(dtype)


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    y: pd.Series,
    y_external: pd.Series,
    features: list[str],
    cat_cols: list[str],
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train_lgb = train.copy()
    test_lgb = test.copy()
    external_lgb = external.copy()
    align_categories([train_lgb, test_lgb, external_lgb], cat_cols)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_lgb))
    test_pred = np.zeros(len(test_lgb))
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_lgb[features], y), start=1):
        print(f"[lgbm] fold {fold}", flush=True)
        X_train = pd.concat([train_lgb.iloc[tr_idx][features], external_lgb[features]], ignore_index=True)
        y_train = pd.concat([y.iloc[tr_idx], y_external], ignore_index=True)
        for col in cat_cols:
            X_train[col] = X_train[col].astype(train_lgb[col].dtype)

        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=7000,
            learning_rate=0.025,
            num_leaves=63,
            min_child_samples=60,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.02,
            reg_lambda=1.0,
            random_state=seed + 1000 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(train_lgb.iloc[va_idx][features], y.iloc[va_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(250), lgb.log_evaluation(500)],
        )
        valid_pred = model.predict_proba(train_lgb.iloc[va_idx][features])[:, 1]
        oof[va_idx] = valid_pred
        test_pred += model.predict_proba(test_lgb[features])[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], valid_pred)
        print(f"[lgbm] fold {fold} AUC: {score:.6f}", flush=True)
        scores.append(float(score))
    return oof, test_pred, scores


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
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed + 2000 + fold,
            od_type="Iter",
            od_wait=200,
            verbose=250,
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
        print(f"[catboost] fold {fold} AUC: {score:.6f}", flush=True)
        scores.append(float(score))
    return oof, test_pred, scores


def optimize_blend(oofs: dict[str, np.ndarray], y: pd.Series) -> tuple[dict[str, float], float]:
    if len(oofs) == 1:
        name = next(iter(oofs))
        return {name: 1.0}, roc_auc_score(y, oofs[name])

    names = list(oofs)
    best_score = -1.0
    best_weights: dict[str, float] = {}
    for weight in np.linspace(0, 1, 21):
        weights = {names[0]: float(weight), names[1]: float(1 - weight)}
        pred = sum(weights[name] * oofs[name] for name in names)
        score = roc_auc_score(y, pred)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Current best local pipeline for S6E5.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--models", default="catboost,lgbm")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--catboost-iterations", default=2500, type=int)
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    competition_cols = [col for col in test_raw.columns if col in train_raw.columns and col != ID_COL]
    external_raw = load_external(args.external_csv, competition_cols)

    train, test, external, y, y_external, features, cat_cols = prepare_frames(train_raw, test_raw, external_raw)
    print(f"Features: {len(features)}", flush=True)
    print(f"Categorical: {cat_cols}", flush=True)
    print(f"External rows: {len(external)}", flush=True)

    selected_models = [model.strip() for model in args.models.split(",") if model.strip()]
    oofs: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    scores: dict[str, object] = {}

    if "catboost" in selected_models:
        oof, pred, fold_scores = train_catboost(
            train, test, external, y, y_external, features, cat_cols, args.n_splits, args.seed, args.catboost_iterations
        )
        oofs["catboost"] = oof
        test_preds["catboost"] = pred
        scores["catboost"] = {"fold_scores": fold_scores, "oof_auc": float(roc_auc_score(y, oof))}

    if "lgbm" in selected_models:
        oof, pred, fold_scores = train_lgbm(
            train, test, external, y, y_external, features, cat_cols, args.n_splits, args.seed
        )
        oofs["lgbm"] = oof
        test_preds["lgbm"] = pred
        scores["lgbm"] = {"fold_scores": fold_scores, "oof_auc": float(roc_auc_score(y, oof))}

    weights, blend_auc = optimize_blend(oofs, y)
    blend_oof = sum(weights[name] * oofs[name] for name in weights)
    blend_test = sum(weights[name] * test_preds[name] for name in weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(weights)
    submission_path = args.output_dir / f"submission_simple_external_{suffix}.csv"
    oof_path = args.output_dir / f"oof_simple_external_{suffix}.csv"
    summary_path = args.output_dir / f"summary_simple_external_{suffix}.json"

    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != ID_COL][0]
    submission[prediction_col] = np.clip(blend_test, 0, 1)
    submission.to_csv(submission_path, index=False)

    oof_df = pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y, **oofs, "blend": blend_oof})
    oof_df.to_csv(oof_path, index=False)

    summary = {
        "features": len(features),
        "categorical_features": cat_cols,
        "external_rows": int(len(external)),
        "n_splits": args.n_splits,
        "models": scores,
        "blend_weights": weights,
        "blend_oof_auc": float(blend_auc),
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
