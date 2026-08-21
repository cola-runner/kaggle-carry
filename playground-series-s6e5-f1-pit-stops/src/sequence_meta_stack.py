from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COL = "id"
TARGET = "PitNextLap"
PREDICTION_COLS = ["pred", "blend", "catboost_realmlp_external", TARGET]


def prediction_column(df: pd.DataFrame) -> str:
    for col in PREDICTION_COLS:
        if col in df.columns and col != ID_COL:
            return col
    candidates = [col for col in df.columns if col not in {ID_COL, TARGET}]
    if not candidates:
        raise ValueError(f"Could not infer prediction column from {list(df.columns)}")
    return candidates[-1]


def load_oof(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = prediction_column(df)
    if df[col].isna().any():
        raise ValueError(f"{path} has missing OOF predictions")
    return df[col].astype(float).to_numpy()


def load_submission_pred(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(path)
    col = prediction_column(df)
    return df, df[col].astype(float).to_numpy(), col


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def add_group_features(all_df: pd.DataFrame, keys: list[str], prefix: str) -> None:
    group = all_df.groupby(keys, dropna=False)
    all_df[f"{prefix}_n"] = group[ID_COL].transform("size").astype(float)
    for col in ["LapNumber", "TyreLife", "RaceProgress", "LapTime (s)", "Cumulative_Degradation"]:
        all_df[f"{prefix}_{col}_min"] = group[col].transform("min")
        all_df[f"{prefix}_{col}_max"] = group[col].transform("max")
        all_df[f"{prefix}_{col}_mean"] = group[col].transform("mean")
        all_df[f"{prefix}_{col}_std"] = group[col].transform("std").fillna(0)
        span = (all_df[f"{prefix}_{col}_max"] - all_df[f"{prefix}_{col}_min"]).replace(0, np.nan)
        all_df[f"{prefix}_{col}_to_max"] = all_df[f"{prefix}_{col}_max"] - all_df[col]
        all_df[f"{prefix}_{col}_from_min"] = all_df[col] - all_df[f"{prefix}_{col}_min"]
        all_df[f"{prefix}_{col}_seen_progress"] = (all_df[col] - all_df[f"{prefix}_{col}_min"]) / span

    all_df[f"{prefix}_lap_rank_pct"] = group["LapNumber"].rank(method="average", pct=True)
    all_df[f"{prefix}_tyre_rank_pct"] = group["TyreLife"].rank(method="average", pct=True)
    all_df[f"{prefix}_race_progress_rank_pct"] = group["RaceProgress"].rank(method="average", pct=True)


def add_neighbor_features(all_df: pd.DataFrame, keys: list[str], prefix: str) -> None:
    sort_cols = keys + ["LapNumber", "TyreLife", ID_COL]
    ordered = all_df.sort_values(sort_cols).copy()
    group = ordered.groupby(keys, dropna=False, sort=False)
    for col in ["LapNumber", "TyreLife", "Stint", "PitStop", "Position", "RaceProgress", "LapTime (s)"]:
        ordered[f"{prefix}_prev_{col}"] = group[col].shift(1)
        ordered[f"{prefix}_next_{col}"] = group[col].shift(-1)
        ordered[f"{prefix}_prev_delta_{col}"] = ordered[col] - ordered[f"{prefix}_prev_{col}"]
        ordered[f"{prefix}_next_delta_{col}"] = ordered[f"{prefix}_next_{col}"] - ordered[col]

    neighbor_cols = [col for col in ordered.columns if col.startswith(f"{prefix}_")]
    all_df[neighbor_cols] = ordered.sort_index()[neighbor_cols]


def add_target_encoding(
    train_part: pd.DataFrame,
    y_part: np.ndarray,
    valid_part: pd.DataFrame,
    test_part: pd.DataFrame,
    keys_list: list[list[str]],
    smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_part.copy()
    valid_out = valid_part.copy()
    test_out = test_part.copy()
    global_mean = float(np.mean(y_part))
    generated: list[str] = []

    source = train_part.copy()
    source["_target"] = y_part
    for keys in keys_list:
        name = "_".join(keys)
        stats = source.groupby(keys, dropna=False)["_target"].agg(["sum", "count"]).reset_index()
        stats[f"te_{name}"] = (stats["sum"] + global_mean * smooth) / (stats["count"] + smooth)
        col = f"te_{name}"
        generated.append(col)
        train_out = train_out.merge(stats[keys + [col]], on=keys, how="left")
        valid_out = valid_out.merge(stats[keys + [col]], on=keys, how="left")
        test_out = test_out.merge(stats[keys + [col]], on=keys, how="left")
        for frame in [train_out, valid_out, test_out]:
            frame[col] = frame[col].fillna(global_mean).astype(float)

    return train_out, valid_out, test_out, generated


def build_base_frame(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = train_raw.drop(columns=[TARGET]).copy()
    test_x = test_raw.copy()
    train_x["Source"] = "train"
    test_x["Source"] = "test"
    n_train = len(train_x)
    all_df = pd.concat([train_x, test_x], ignore_index=True)

    all_df["Race_Year"] = all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)
    all_df["Race_Compound"] = all_df["Race"].astype(str) + "_" + all_df["Compound"].astype(str)
    all_df["Driver_Race"] = all_df["Driver"].astype(str) + "_" + all_df["Race"].astype(str)
    all_df["Driver_Race_Year"] = (
        all_df["Driver"].astype(str) + "_" + all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)
    )
    all_df["Driver_Compound"] = all_df["Driver"].astype(str) + "_" + all_df["Compound"].astype(str)
    all_df["Race_Stint"] = all_df["Race"].astype(str) + "_" + all_df["Stint"].astype(str)
    all_df["Race_Compound_Stint"] = all_df["Race_Compound"] + "_" + all_df["Stint"].astype(str)

    total_laps = all_df["LapNumber"] / all_df["RaceProgress"].replace(0, np.nan)
    all_df["TotalLaps_est"] = total_laps.clip(1, 120)
    all_df["LapsRemaining_est"] = all_df["TotalLaps_est"] - all_df["LapNumber"]
    all_df["TyreLife_per_LapNumber"] = all_df["TyreLife"] / all_df["LapNumber"].replace(0, np.nan)
    all_df["LapTime_x_RaceProgress"] = all_df["LapTime (s)"] * all_df["RaceProgress"]
    all_df["Degradation_per_TyreLife"] = all_df["Cumulative_Degradation"] / all_df["TyreLife"].replace(0, np.nan)

    group_specs = [
        (["Year", "Race", "Driver"], "yr_race_driver"),
        (["Year", "Race", "Driver", "Stint"], "yr_race_driver_stint"),
        (["Year", "Race", "Compound"], "yr_race_compound"),
        (["Year", "Race", "Stint", "Compound"], "yr_race_stint_compound"),
        (["Year", "Race", "PitStop", "Compound"], "yr_race_pit_compound"),
    ]
    for keys, prefix in group_specs:
        add_group_features(all_df, keys, prefix)

    add_neighbor_features(all_df, ["Year", "Race", "Driver"], "seq_driver_race")
    add_neighbor_features(all_df, ["Year", "Race", "Driver", "Stint"], "seq_driver_stint")

    cat_cols = [
        "Driver",
        "Compound",
        "Race",
        "Source",
        "Race_Year",
        "Race_Compound",
        "Driver_Race",
        "Driver_Race_Year",
        "Driver_Compound",
        "Race_Stint",
        "Race_Compound_Stint",
    ]
    for col in cat_cols:
        all_df[col] = all_df[col].astype("category")

    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    train = all_df.iloc[:n_train].reset_index(drop=True)
    test = all_df.iloc[n_train:].reset_index(drop=True)
    return train, test, cat_cols


def make_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=5000,
        learning_rate=0.018,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.08,
        reg_lambda=1.6,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-aware meta stack for the F1 pit stop competition.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=9097, type=int)
    parser.add_argument("--suffix", default="sequence_meta_stack")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    y = train_raw[TARGET].astype(int).to_numpy()

    pred_specs = {
        "best4": (
            args.output_dir / "oof_segment_best4_year_cat7070.csv",
            args.output_dir / "submission_segment_best4_year_cat7070.csv",
        ),
        "best0": (
            args.output_dir / "oof_blend_logstack_realmlp_catboost_seed7070.csv",
            args.output_dir / "submission_blend_logstack_realmlp_catboost_seed7070.csv",
        ),
        "cat7070": (
            args.output_dir / "oof_catboost_realmlp_external_full_seed7070.csv",
            args.output_dir / "submission_catboost_realmlp_external_full_seed7070.csv",
        ),
        "norm": (
            args.output_dir / "oof_lgbm_normproxy_external.csv",
            args.output_dir / "submission_lgbm_normproxy_external.csv",
        ),
        "seed42": (
            args.output_dir / "oof_blend_seed42_cat.csv",
            args.output_dir / "submission_blend_seed42_cat.csv",
        ),
        "te": (
            args.output_dir / "oof_blend_oof_search_te.csv",
            args.output_dir / "submission_blend_oof_search_te.csv",
        ),
        "logstack": (
            args.output_dir / "oof_logstack_sweep.csv",
            args.output_dir / "submission_logstack_sweep.csv",
        ),
    }

    train, test, cat_cols = build_base_frame(train_raw, test_raw)
    for name, (oof_path, sub_path) in pred_specs.items():
        train[f"pred_{name}"] = load_oof(oof_path)
        _, test_pred, _ = load_submission_pred(sub_path)
        test[f"pred_{name}"] = test_pred
        train[f"logit_{name}"] = logit(train[f"pred_{name}"].to_numpy())
        test[f"logit_{name}"] = logit(test[f"pred_{name}"].to_numpy())
        train[f"rank_{name}"] = rank_pct(train[f"pred_{name}"].to_numpy())
        test[f"rank_{name}"] = rank_pct(test[f"pred_{name}"].to_numpy())

    for left, right in [("best4", "norm"), ("best4", "cat7070"), ("best4", "seed42"), ("best4", "te")]:
        train[f"diff_{left}_{right}"] = train[f"pred_{left}"] - train[f"pred_{right}"]
        test[f"diff_{left}_{right}"] = test[f"pred_{left}"] - test[f"pred_{right}"]

    te_keys = [
        ["Year"],
        ["Year", "Race"],
        ["Year", "Race", "Compound"],
        ["Year", "Race", "Stint", "Compound"],
        ["Year", "Race", "TyreLife", "Stint", "Compound", "PitStop"],
    ]

    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores: list[float] = []
    feature_importance: dict[str, float] = {}

    base_features = [col for col in train.columns if col != ID_COL]
    for fold, (tr_idx, va_idx) in enumerate(folds.split(train[base_features], y), start=1):
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train.iloc[va_idx].reset_index(drop=True)
        fold_test = test.copy()
        fold_y = y[tr_idx]

        fold_train, fold_valid, fold_test, te_cols = add_target_encoding(
            fold_train, fold_y, fold_valid, fold_test, te_keys, smooth=80.0
        )
        features = base_features + te_cols
        model = make_model(args.seed + fold)
        model.fit(
            fold_train[features],
            fold_y,
            eval_set=[(fold_valid[features], y[va_idx])],
            eval_metric="auc",
            categorical_feature=[col for col in cat_cols if col in features],
            callbacks=[early_stopping(250), log_evaluation(250)],
        )
        valid_pred = model.predict_proba(fold_valid[features])[:, 1]
        oof[va_idx] = valid_pred
        test_pred += model.predict_proba(fold_test[features])[:, 1] / args.n_splits
        score = float(roc_auc_score(y[va_idx], valid_pred))
        fold_scores.append(score)
        for name, value in zip(features, model.feature_importances_):
            feature_importance[name] = feature_importance.get(name, 0.0) + float(value)
        print(f"[sequence-meta] fold {fold} AUC: {score:.9f}", flush=True)

    meta_auc = float(roc_auc_score(y, oof))
    base_auc = float(roc_auc_score(y, train["pred_best4"].to_numpy()))
    best_blend_score = -1.0
    best_alpha = 0.0
    best_oof = None
    best_test = None
    for alpha in np.linspace(0, 1, 1001):
        pred = alpha * train["pred_best4"].to_numpy() + (1 - alpha) * oof
        score = roc_auc_score(y, pred)
        if score > best_blend_score:
            best_blend_score = float(score)
            best_alpha = float(alpha)
            best_oof = pred
            best_test = alpha * test["pred_best4"].to_numpy() + (1 - alpha) * test_pred

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = sample.copy()
    pred_col = [col for col in submission.columns if col != ID_COL][0]
    submission[pred_col] = np.clip(best_test, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train_raw[ID_COL].values, TARGET: y, "pred": best_oof, "meta_pred": oof}).to_csv(
        oof_path, index=False
    )

    top_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:40]
    summary = {
        "base_oof_auc": base_auc,
        "meta_oof_auc": meta_auc,
        "blend_oof_auc": best_blend_score,
        "alpha_base": best_alpha,
        "alpha_meta": float(1 - best_alpha),
        "fold_scores": fold_scores,
        "features": len(base_features) + len(te_keys),
        "top_importance": top_importance,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
