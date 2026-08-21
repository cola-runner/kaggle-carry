from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


TARGET = "PitNextLap"
ID_COL = "id"


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {path.name for path in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return

    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Could not find Kaggle csv files or zip in {data_dir}")

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} ...")
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)


def load_data(data_dir: Path, max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extract_zip_if_needed(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if max_rows:
        train = train.sample(max_rows, random_state=42).sort_index().reset_index(drop=True)
    return train, test, sample


def clean_column_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def safe_divide(a: pd.Series, b: pd.Series | np.ndarray | float) -> pd.Series:
    out = a.astype(float) / pd.Series(b, index=a.index).replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def make_fixed_bin(series: pd.Series, bins: list[float], name: str) -> pd.Series:
    values = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return values.fillna(-1).astype(int).astype(str).radd(f"{name}_")


def add_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    y = train[TARGET].copy()
    train_x = train.drop(columns=[TARGET]).copy()
    test_x = test.copy()
    n_train = len(train_x)

    all_df = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    all_df = all_df.rename(columns={"LapTime (s)": "LapTime_s"})

    base_cat_cols = ["Driver", "Compound", "Race"]
    for col in base_cat_cols:
        all_df[col] = all_df[col].astype(str)

    all_df["TotalLaps_est"] = safe_divide(all_df["LapNumber"], all_df["RaceProgress"])
    all_df["TotalLaps_est"] = all_df["TotalLaps_est"].clip(1, 120)
    all_df["LapsRemaining_est"] = all_df["TotalLaps_est"] - all_df["LapNumber"]
    all_df["TyreLife_frac_race"] = safe_divide(all_df["TyreLife"], all_df["TotalLaps_est"])
    all_df["TyreLife_frac_lap"] = safe_divide(all_df["TyreLife"], all_df["LapNumber"])
    all_df["TyreLife_to_remaining"] = safe_divide(all_df["TyreLife"], all_df["LapsRemaining_est"] + 1)
    all_df["Lap_to_total_laps"] = safe_divide(all_df["LapNumber"], all_df["TotalLaps_est"])
    all_df["Stint_to_lap"] = safe_divide(all_df["Stint"], all_df["LapNumber"])

    all_df["Abs_Position_Change"] = all_df["Position_Change"].abs()
    all_df["LapTime_Delta_abs"] = all_df["LapTime_Delta"].abs()
    all_df["LapTime_Delta_sign"] = np.sign(all_df["LapTime_Delta"])
    all_df["Cumulative_Degradation_abs"] = all_df["Cumulative_Degradation"].abs()
    all_df["Cumulative_Degradation_sign"] = np.sign(all_df["Cumulative_Degradation"])
    all_df["LogAbs_LapTime_Delta"] = np.sign(all_df["LapTime_Delta"]) * np.log1p(all_df["LapTime_Delta"].abs())
    all_df["LogAbs_Degradation"] = np.sign(all_df["Cumulative_Degradation"]) * np.log1p(
        all_df["Cumulative_Degradation"].abs()
    )

    all_df["Degradation_per_TyreLife"] = safe_divide(all_df["Cumulative_Degradation"], all_df["TyreLife"] + 1)
    all_df["Degradation_per_Lap"] = safe_divide(all_df["Cumulative_Degradation"], all_df["LapNumber"] + 1)
    all_df["Delta_per_TyreLife"] = safe_divide(all_df["LapTime_Delta"], all_df["TyreLife"] + 1)
    all_df["Delta_per_Lap"] = safe_divide(all_df["LapTime_Delta"], all_df["LapNumber"] + 1)

    all_df["Position_x_RaceProgress"] = all_df["Position"] * all_df["RaceProgress"]
    all_df["Position_x_TyreLife"] = all_df["Position"] * all_df["TyreLife"]
    all_df["Stint_x_TyreLife"] = all_df["Stint"] * all_df["TyreLife"]
    all_df["PitStop_x_TyreLife"] = all_df["PitStop"] * all_df["TyreLife"]
    all_df["PitStop_x_Stint"] = all_df["PitStop"] * all_df["Stint"]
    all_df["LateRace_TyreLife"] = (all_df["RaceProgress"] > 0.65).astype(int) * all_df["TyreLife"]

    all_df["Race_Year"] = all_df["Race"] + "_" + all_df["Year"].astype(str)
    all_df["Race_Compound"] = all_df["Race"] + "_" + all_df["Compound"]
    all_df["Race_Compound_Year"] = all_df["Race"] + "_" + all_df["Compound"] + "_" + all_df["Year"].astype(str)
    all_df["Driver_Race"] = all_df["Driver"] + "_" + all_df["Race"]
    all_df["Driver_Compound"] = all_df["Driver"] + "_" + all_df["Compound"]
    all_df["Driver_Year"] = all_df["Driver"] + "_" + all_df["Year"].astype(str)
    all_df["Compound_Stint"] = all_df["Compound"] + "_S" + all_df["Stint"].astype(str)
    all_df["Race_Stint"] = all_df["Race"] + "_S" + all_df["Stint"].astype(str)
    all_df["Race_PitStop"] = all_df["Race"] + "_P" + all_df["PitStop"].astype(str)

    all_df["TyreLife_bin"] = make_fixed_bin(all_df["TyreLife"], [-1, 3, 6, 10, 15, 22, 30, 45, 80, 200], "tyre")
    all_df["LapNumber_bin"] = make_fixed_bin(all_df["LapNumber"], [-1, 5, 12, 20, 30, 42, 55, 70, 120], "lap")
    all_df["RaceProgress_bin"] = make_fixed_bin(
        all_df["RaceProgress"], [-0.01, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.01], "progress"
    )
    all_df["Position_bin"] = make_fixed_bin(all_df["Position"], [0, 3, 6, 10, 14, 20, 99], "pos")
    all_df["LapsRemaining_bin"] = make_fixed_bin(
        all_df["LapsRemaining_est"], [-20, 0, 3, 8, 15, 25, 40, 120], "remaining"
    )

    cat_cols = [
        "Driver",
        "Compound",
        "Race",
        "Race_Year",
        "Race_Compound",
        "Race_Compound_Year",
        "Driver_Race",
        "Driver_Compound",
        "Driver_Year",
        "Compound_Stint",
        "Race_Stint",
        "Race_PitStop",
        "TyreLife_bin",
        "LapNumber_bin",
        "RaceProgress_bin",
        "Position_bin",
        "LapsRemaining_bin",
    ]

    for col in cat_cols:
        counts = all_df[col].value_counts(dropna=False)
        all_df[f"{col}_freq"] = all_df[col].map(counts).astype(float) / len(all_df)

    stat_keys = ["Race", "Driver", "Compound", "Race_Compound", "Race_Year", "Compound_Stint"]
    stat_values = ["LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "TyreLife", "Position", "RaceProgress"]
    for key in stat_keys:
        grouped = all_df.groupby(key, observed=True)
        for value in stat_values:
            mean = grouped[value].transform("mean")
            all_df[f"{value}_minus_{key}_mean"] = all_df[value] - mean
            all_df[f"{value}_ratio_{key}_mean"] = safe_divide(all_df[value], mean)

    all_df = all_df.replace([np.inf, -np.inf], np.nan).copy()
    all_df.columns = [clean_column_name(col) for col in all_df.columns]
    cat_cols = [clean_column_name(col) for col in cat_cols]

    train_features = all_df.iloc[:n_train].reset_index(drop=True)
    test_features = all_df.iloc[n_train:].reset_index(drop=True)
    train_features[TARGET] = y.reset_index(drop=True).astype(int)
    return train_features, test_features, cat_cols


def add_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
    smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    global_mean = y_train.mean()
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    te_cols: list[str] = []

    for col in columns:
        stats = y_train.groupby(X_train[col]).agg(["mean", "count"])
        encoded = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        new_col = f"{col}_target_mean"
        X_train[new_col] = X_train[col].map(encoded).fillna(global_mean).astype(float)
        X_valid[new_col] = X_valid[col].map(encoded).fillna(global_mean).astype(float)
        X_test[new_col] = X_test[col].map(encoded).fillna(global_mean).astype(float)
        te_cols.append(new_col)

    return X_train, X_valid, X_test, te_cols


def as_lightgbm_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].astype("category")
    return out


def as_catboost_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].astype(str)
    return out


def make_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=5000,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=6.0,
        random_strength=0.6,
        bagging_temperature=0.3,
        border_count=254,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        od_type="Iter",
        od_wait=250,
        verbose=250,
        allow_writing_files=False,
        thread_count=-1,
    )


def make_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=5000,
        learning_rate=0.025,
        max_depth=6,
        min_child_weight=12,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=3.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=16,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=250,
    )


def optimize_blend(oof_predictions: dict[str, np.ndarray], y: pd.Series) -> tuple[dict[str, float], float]:
    names = list(oof_predictions)
    if len(names) == 1:
        return {names[0]: 1.0}, roc_auc_score(y, oof_predictions[names[0]])

    best_score = -1.0
    best_weights: dict[str, float] = {}
    grid = np.linspace(0, 1, 21)

    if len(names) == 2:
        first, second = names
        for w in grid:
            pred = w * oof_predictions[first] + (1 - w) * oof_predictions[second]
            score = roc_auc_score(y, pred)
            if score > best_score:
                best_score = score
                best_weights = {first: float(w), second: float(1 - w)}
        return best_weights, best_score

    for w0 in grid:
        for w1 in grid:
            if w0 + w1 > 1:
                continue
            w2 = 1 - w0 - w1
            weights = {names[0]: w0, names[1]: w1, names[2]: w2}
            pred = sum(weights[name] * oof_predictions[name] for name in names)
            score = roc_auc_score(y, pred)
            if score > best_score:
                best_score = score
                best_weights = {name: float(weight) for name, weight in weights.items()}
    return best_weights, best_score


def train_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    folds: StratifiedKFold,
    seed: int,
    te_cols: list[str],
    smooth: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_scores: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        print(f"\n[{model_name}] fold {fold}")
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        fold_test = X_test.copy()

        X_train, X_valid, fold_test, generated_te_cols = add_target_encoding(
            X_train, y_train, X_valid, fold_test, te_cols, smooth
        )
        model_cat_cols = cat_cols

        if model_name == "lgbm":
            model = make_lgbm(seed + fold)
            X_train_model = as_lightgbm_frame(X_train, model_cat_cols)
            X_valid_model = as_lightgbm_frame(X_valid, model_cat_cols)
            X_test_model = as_lightgbm_frame(fold_test, model_cat_cols)
            model.fit(
                X_train_model,
                y_train,
                eval_set=[(X_valid_model, y_valid)],
                categorical_feature=model_cat_cols,
                callbacks=[lgb.early_stopping(250), lgb.log_evaluation(250)],
            )
            valid_pred = model.predict_proba(X_valid_model)[:, 1]
            test_fold_pred = model.predict_proba(X_test_model)[:, 1]

        elif model_name == "catboost":
            model = make_catboost(seed + fold)
            X_train_model = as_catboost_frame(X_train, model_cat_cols)
            X_valid_model = as_catboost_frame(X_valid, model_cat_cols)
            X_test_model = as_catboost_frame(fold_test, model_cat_cols)
            model.fit(
                X_train_model,
                y_train,
                cat_features=model_cat_cols,
                eval_set=(X_valid_model, y_valid),
                use_best_model=True,
            )
            valid_pred = model.predict_proba(X_valid_model)[:, 1]
            test_fold_pred = model.predict_proba(X_test_model)[:, 1]

        elif model_name == "xgboost":
            model = make_xgboost(seed + fold)
            X_train_model = as_lightgbm_frame(X_train, model_cat_cols)
            X_valid_model = as_lightgbm_frame(X_valid, model_cat_cols)
            X_test_model = as_lightgbm_frame(fold_test, model_cat_cols)
            model.fit(X_train_model, y_train, eval_set=[(X_valid_model, y_valid)], verbose=250)
            valid_pred = model.predict_proba(X_valid_model)[:, 1]
            test_fold_pred = model.predict_proba(X_test_model)[:, 1]

        else:
            raise ValueError(f"Unknown model: {model_name}")

        score = roc_auc_score(y_valid, valid_pred)
        print(f"[{model_name}] fold {fold} AUC: {score:.6f} with {len(generated_te_cols)} TE features")
        oof[valid_idx] = valid_pred
        test_pred += test_fold_pred / folds.n_splits
        fold_scores.append(float(score))

    overall = roc_auc_score(y, oof)
    print(f"\n[{model_name}] OOF AUC: {overall:.6f}")
    return oof, test_pred, fold_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Stronger CV pipeline for Kaggle S6E5.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--models", default="lgbm,catboost", help="Comma-separated: lgbm,catboost,xgboost")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--te-smooth", default=80.0, type=float)
    parser.add_argument("--max-rows", default=None, type=int, help="Debug only: sample training rows")
    args = parser.parse_args()

    train_raw, test_raw, sample = load_data(args.data_dir, args.max_rows)
    train, test, cat_cols = add_feature_engineering(train_raw, test_raw)
    y = train.pop(TARGET).astype(int)
    X = train.drop(columns=[ID_COL])
    X_test = test.drop(columns=[ID_COL])

    te_cols = [
        "Driver",
        "Race",
        "Compound",
        "Race_Year",
        "Race_Compound",
        "Race_Compound_Year",
        "Driver_Race",
        "Driver_Compound",
        "Compound_Stint",
        "Race_Stint",
    ]
    te_cols = [col for col in te_cols if col in X.columns]

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    print(f"Train features: {X.shape}")
    print(f"Test features:  {X_test.shape}")
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Target encoding features per fold: {len(te_cols)}")
    print(f"Models: {models}")

    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    model_scores: dict[str, dict[str, object]] = {}

    for model_name in models:
        oof, test_pred, fold_scores = train_model(
            model_name=model_name,
            X=X,
            y=y,
            X_test=X_test,
            cat_cols=cat_cols,
            folds=folds,
            seed=args.seed,
            te_cols=te_cols,
            smooth=args.te_smooth,
        )
        oof_predictions[model_name] = oof
        test_predictions[model_name] = test_pred
        model_scores[model_name] = {
            "fold_scores": fold_scores,
            "oof_auc": float(roc_auc_score(y, oof)),
        }

    weights, blend_auc = optimize_blend(oof_predictions, y)
    blend_oof = sum(weights[name] * oof_predictions[name] for name in weights)
    blend_test = sum(weights[name] * test_predictions[name] for name in weights)
    blend_test = np.clip(blend_test, 0.0, 1.0)
    print(f"\nBlend weights: {weights}")
    print(f"Blend OOF AUC: {blend_auc:.6f}")
    print(f"Blend check AUC: {roc_auc_score(y, blend_oof):.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != ID_COL][0]
    submission[prediction_col] = blend_test

    suffix = "_".join(models)
    output_path = args.output_dir / f"submission_strong_cv_{suffix}.csv"
    oof_path = args.output_dir / f"oof_strong_cv_{suffix}.csv"
    summary_path = args.output_dir / f"summary_strong_cv_{suffix}.json"

    submission.to_csv(output_path, index=False)
    pd.DataFrame({ID_COL: train_raw[ID_COL].values, TARGET: y.values, **oof_predictions, "blend": blend_oof}).to_csv(
        oof_path, index=False
    )

    summary = {
        "competition": "playground-series-s6e5",
        "rows_train": int(len(train_raw)),
        "rows_test": int(len(test_raw)),
        "features": int(X.shape[1]),
        "categorical_features": int(len(cat_cols)),
        "target_encoding_features": int(len(te_cols)),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "models": model_scores,
        "blend_weights": weights,
        "blend_oof_auc": float(blend_auc),
        "output": str(output_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote submission: {output_path}")
    print(f"Wrote OOF:        {oof_path}")
    print(f"Wrote summary:    {summary_path}")


if __name__ == "__main__":
    main()
