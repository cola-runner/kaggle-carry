from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COL = "id"
TARGET = "PitNextLap"
LAPTIME_COL = "LapTime (s)"


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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan).astype(float)
    values = numerator.astype(float) / denominator
    return values.replace([np.inf, -np.inf], np.nan)


def load_external(external_csv: Path, competition_feature_cols: list[str]) -> pd.DataFrame:
    external = pd.read_csv(external_csv)
    needed = competition_feature_cols + [TARGET]
    missing = [col for col in needed if col not in external.columns]
    if missing:
        raise ValueError(f"External data is missing columns: {missing}")
    return external[needed].copy()


def qcut_as_category(series: pd.Series, q: int, name: str) -> pd.Series:
    codes = pd.qcut(series.rank(method="first"), q=q, labels=False, duplicates="drop")
    return codes.fillna(-1).astype(int).astype(str).radd(f"{name}_")


def add_realmlp_style_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    external_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    train_x = train_raw.drop(columns=[TARGET]).copy()
    test_x = test_raw.copy()
    external_x = external_raw.drop(columns=[TARGET]).copy()

    train_x["Source"] = "competition"
    test_x["Source"] = "competition"
    external_x["Source"] = "external"

    n_train = len(train_x)
    n_test = len(test_x)
    all_df = pd.concat([train_x, test_x, external_x], axis=0, ignore_index=True)

    all_df["_LapNumber_/_RaceProgress"] = safe_divide(all_df["LapNumber"], all_df["RaceProgress"])
    all_df["_TyreLife_/_LapNumber"] = safe_divide(all_df["TyreLife"], all_df["LapNumber"])

    numeric_for_floor = [
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "TyreLife",
        "Position",
        LAPTIME_COL,
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Position_Change",
        "_LapNumber_/_RaceProgress",
        "_TyreLife_/_LapNumber",
    ]
    floor_cat_cols: list[str] = []
    for col in numeric_for_floor:
        new_col = f"{col}_"
        all_df[new_col] = np.floor(all_df[col].astype(float)).astype("Int64").astype(str)
        floor_cat_cols.append(new_col)

    for col in ["Driver", "Compound", "Race", "Year", "PitStop"]:
        new_col = f"{col}_count"
        counts = all_df[col].astype(str).value_counts(dropna=False)
        all_df[new_col] = all_df[col].astype(str).map(counts).astype(float)

    all_df["RaceProgress_200_quantile_bin_"] = qcut_as_category(
        all_df["RaceProgress"], q=200, name="RaceProgress"
    )
    all_df["LapTime (s)_7_quantile_bin_"] = qcut_as_category(all_df[LAPTIME_COL], q=7, name="LapTime_s")
    all_df["Race_Compound_"] = all_df["Race"].astype(str) + "_" + all_df["Compound"].astype(str)
    all_df["Race_Year_"] = all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)

    object_cat_cols = ["Driver", "Compound", "Race", "Source"]
    extra_cat_cols = [
        "RaceProgress_200_quantile_bin_",
        "LapTime (s)_7_quantile_bin_",
        "Race_Compound_",
        "Race_Year_",
    ]
    cat_cols = object_cat_cols + floor_cat_cols + extra_cat_cols
    for col in cat_cols:
        all_df[col] = all_df[col].astype(str)

    train = all_df.iloc[:n_train].reset_index(drop=True)
    test = all_df.iloc[n_train : n_train + n_test].reset_index(drop=True)
    external = all_df.iloc[n_train + n_test :].reset_index(drop=True)

    y = train_raw[TARGET].astype(int).reset_index(drop=True)
    y_external = external_raw[TARGET].astype(int).reset_index(drop=True)
    features = [col for col in all_df.columns if col != ID_COL]
    return train, test, external, y, y_external, features, cat_cols


def add_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
    smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    global_mean = float(y_train.mean())
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    generated: list[str] = []

    for col in columns:
        stats = y_train.groupby(X_train[col].astype(str)).agg(["mean", "count"])
        encoded = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        new_col = f"_{col}_TE"
        X_train[new_col] = X_train[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        X_valid[new_col] = X_valid[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        X_test[new_col] = X_test[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        generated.append(new_col)

    return X_train, X_valid, X_test, generated


def as_catboost_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].astype(str)
    return out


def make_catboost(seed: int, iterations: int, learning_rate: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        od_type="Iter",
        od_wait=250,
        verbose=300,
        allow_writing_files=False,
        thread_count=-1,
    )


def train_catboost_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    y: pd.Series,
    y_external: pd.Series,
    features: list[str],
    cat_cols: list[str],
    te_cols: list[str],
    n_splits: int,
    seed: int,
    iterations: int,
    learning_rate: float,
    te_smooth: float,
    max_folds: int | None,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(train), np.nan, dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    scores: list[float] = []
    completed_folds: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train[features], y), start=1):
        if max_folds is not None and fold > max_folds:
            break

        print(f"\n[catboost-realmlp] fold {fold}", flush=True)
        X_train = pd.concat([train.iloc[tr_idx][features], external[features]], ignore_index=True)
        y_train = pd.concat([y.iloc[tr_idx], y_external], ignore_index=True)
        X_valid = train.iloc[va_idx][features].reset_index(drop=True)
        y_valid = y.iloc[va_idx].reset_index(drop=True)
        X_test = test[features].copy()

        X_train, X_valid, X_test, generated_te_cols = add_target_encoding(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            X_test=X_test,
            columns=te_cols,
            smooth=te_smooth,
        )
        model_features = features + generated_te_cols
        model = make_catboost(seed + fold, iterations=iterations, learning_rate=learning_rate)
        model.fit(
            as_catboost_frame(X_train[model_features], cat_cols),
            y_train,
            cat_features=cat_cols,
            eval_set=(as_catboost_frame(X_valid[model_features], cat_cols), y_valid),
            use_best_model=True,
        )

        valid_pred = model.predict_proba(as_catboost_frame(X_valid[model_features], cat_cols))[:, 1]
        test_fold_pred = model.predict_proba(as_catboost_frame(X_test[model_features], cat_cols))[:, 1]
        oof[va_idx] = valid_pred
        test_pred += test_fold_pred
        score = roc_auc_score(y_valid, valid_pred)
        print(f"[catboost-realmlp] fold {fold} AUC: {score:.9f}", flush=True)
        scores.append(float(score))
        completed_folds.append(fold)

    if completed_folds:
        test_pred /= len(completed_folds)
    return oof, test_pred, scores, completed_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost on RealMLP-style public features plus external F1 data.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=7070, type=int)
    parser.add_argument("--iterations", default=3000, type=int)
    parser.add_argument("--learning-rate", default=0.045, type=float)
    parser.add_argument("--te-smooth", default=80.0, type=float)
    parser.add_argument("--max-folds", default=None, type=int)
    parser.add_argument("--suffix", default="catboost_realmlp_external")
    args = parser.parse_args()

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    competition_cols = [col for col in test_raw.columns if col in train_raw.columns and col != ID_COL]
    external_raw = load_external(args.external_csv, competition_cols)

    train, test, external, y, y_external, features, cat_cols = add_realmlp_style_features(
        train_raw, test_raw, external_raw
    )
    te_cols = ["Race_Compound_", "Race_Year_"]
    print(f"Features before TE: {len(features)}", flush=True)
    print(f"Categorical features: {len(cat_cols)}", flush=True)
    print(f"Target encoding columns: {te_cols}", flush=True)
    print(f"External rows: {len(external)}", flush=True)

    oof, test_pred, fold_scores, completed_folds = train_catboost_cv(
        train=train,
        test=test,
        external=external,
        y=y,
        y_external=y_external,
        features=features,
        cat_cols=cat_cols,
        te_cols=te_cols,
        n_splits=args.n_splits,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        te_smooth=args.te_smooth,
        max_folds=args.max_folds,
    )

    completed_mask = ~np.isnan(oof)
    partial_auc = float(roc_auc_score(y[completed_mask], oof[completed_mask])) if completed_mask.any() else None
    full_oof_auc = float(roc_auc_score(y, oof)) if completed_mask.all() else None
    print(f"\nCompleted folds: {completed_folds}", flush=True)
    print(f"Fold scores: {fold_scores}", flush=True)
    if partial_auc is not None:
        print(f"Completed-fold OOF AUC: {partial_auc:.9f}", flush=True)
    if full_oof_auc is not None:
        print(f"Full OOF AUC: {full_oof_auc:.9f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_tag = "full" if completed_mask.all() else f"folds{len(completed_folds)}"
    suffix = f"{args.suffix}_{fold_tag}_seed{args.seed}"
    submission_path = args.output_dir / f"submission_{suffix}.csv"
    oof_path = args.output_dir / f"oof_{suffix}.csv"
    summary_path = args.output_dir / f"summary_{suffix}.json"

    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != ID_COL][0]
    submission[prediction_col] = np.clip(test_pred, 0, 1)
    submission.to_csv(submission_path, index=False)

    pd.DataFrame({ID_COL: train_raw[ID_COL].values, TARGET: y.values, "catboost_realmlp_external": oof}).to_csv(
        oof_path, index=False
    )

    summary = {
        "competition": "playground-series-s6e5",
        "features_before_te": int(len(features)),
        "categorical_features": int(len(cat_cols)),
        "target_encoding_columns": te_cols,
        "external_rows": int(len(external)),
        "n_splits": int(args.n_splits),
        "completed_folds": completed_folds,
        "seed": int(args.seed),
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "te_smooth": float(args.te_smooth),
        "fold_scores": fold_scores,
        "completed_fold_oof_auc": partial_auc,
        "full_oof_auc": full_oof_auc,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote submission: {submission_path}", flush=True)
    print(f"Wrote OOF:        {oof_path}", flush=True)
    print(f"Wrote summary:    {summary_path}", flush=True)


if __name__ == "__main__":
    main()
