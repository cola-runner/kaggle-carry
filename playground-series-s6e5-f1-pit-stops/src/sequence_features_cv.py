from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


ID_COL = "id"
TARGET = "PitNextLap"
LAPTIME_RAW = "LapTime (s)"
LAPTIME = "LapTime_s"
GROUP_COLS = ["Year", "Race", "Driver"]

EXTERNAL_KEY_SETS = [
    ("yrd_lap", ["Year", "Race", "Driver", "LapNumber"]),
    ("yrd_lap_compound", ["Year", "Race", "Driver", "LapNumber", "Compound"]),
    ("yrl_position_stint", ["Year", "Race", "LapNumber", "Position", "Stint"]),
    ("yrl_position_compound", ["Year", "Race", "LapNumber", "Position", "Compound"]),
    ("yrl_stint_tyre_compound", ["Year", "Race", "LapNumber", "Stint", "TyreLife", "Compound"]),
    ("yrl_position_pit_compound", ["Year", "Race", "LapNumber", "Position", "PitStop", "Compound"]),
    ("yrl_position_stint_compound", ["Year", "Race", "LapNumber", "Position", "Stint", "Compound"]),
]

SEQUENCE_NUMERIC_COLS = [
    LAPTIME,
    "LapTime_Delta",
    "Cumulative_Degradation",
    "TyreLife",
    "Position",
    "Position_Change",
    "RaceProgress",
    "PitStop",
    "Stint",
]


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


def clean_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    return re.sub(r"_+", "_", cleaned).strip("_")


def safe_divide(a: pd.Series, b: pd.Series | np.ndarray | float) -> pd.Series:
    out = a.astype(float) / pd.Series(b, index=a.index).replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def make_bin(series: pd.Series, bins: list[float], prefix: str) -> pd.Series:
    values = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return values.fillna(-1).astype(int).astype(str).radd(f"{prefix}_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={LAPTIME_RAW: LAPTIME}).copy()


def add_row_features(all_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = all_df.copy()
    cat_cols = ["Driver", "Compound", "Race"]
    for col in cat_cols:
        out[col] = out[col].astype(str)

    out["TotalLaps_est"] = safe_divide(out["LapNumber"], out["RaceProgress"]).clip(1, 120)
    out["LapsRemaining_est"] = out["TotalLaps_est"] - out["LapNumber"]
    out["TyreLife_frac_race"] = safe_divide(out["TyreLife"], out["TotalLaps_est"])
    out["TyreLife_frac_lap"] = safe_divide(out["TyreLife"], out["LapNumber"])
    out["TyreLife_to_remaining"] = safe_divide(out["TyreLife"], out["LapsRemaining_est"] + 1)
    out["Lap_to_total_laps"] = safe_divide(out["LapNumber"], out["TotalLaps_est"])
    out["Stint_to_lap"] = safe_divide(out["Stint"], out["LapNumber"])

    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["LapTime_Delta_abs"] = out["LapTime_Delta"].abs()
    out["LapTime_Delta_sign"] = np.sign(out["LapTime_Delta"])
    out["Cumulative_Degradation_abs"] = out["Cumulative_Degradation"].abs()
    out["Cumulative_Degradation_sign"] = np.sign(out["Cumulative_Degradation"])
    out["LogAbs_LapTime_Delta"] = np.sign(out["LapTime_Delta"]) * np.log1p(out["LapTime_Delta"].abs())
    out["LogAbs_Degradation"] = np.sign(out["Cumulative_Degradation"]) * np.log1p(
        out["Cumulative_Degradation"].abs()
    )
    out["Degradation_per_TyreLife"] = safe_divide(out["Cumulative_Degradation"], out["TyreLife"] + 1)
    out["Degradation_per_Lap"] = safe_divide(out["Cumulative_Degradation"], out["LapNumber"] + 1)
    out["Delta_per_TyreLife"] = safe_divide(out["LapTime_Delta"], out["TyreLife"] + 1)
    out["Delta_per_Lap"] = safe_divide(out["LapTime_Delta"], out["LapNumber"] + 1)

    out["Position_x_RaceProgress"] = out["Position"] * out["RaceProgress"]
    out["Position_x_TyreLife"] = out["Position"] * out["TyreLife"]
    out["Stint_x_TyreLife"] = out["Stint"] * out["TyreLife"]
    out["PitStop_x_TyreLife"] = out["PitStop"] * out["TyreLife"]
    out["PitStop_x_Stint"] = out["PitStop"] * out["Stint"]
    out["LateRace_TyreLife"] = (out["RaceProgress"] > 0.65).astype(int) * out["TyreLife"]

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Driver_Race_Year"] = out["Driver"] + "_" + out["Race"] + "_" + out["Year"].astype(str)
    out["Race_Compound"] = out["Race"] + "_" + out["Compound"]
    out["Race_Compound_Year"] = out["Race"] + "_" + out["Compound"] + "_" + out["Year"].astype(str)
    out["Driver_Compound"] = out["Driver"] + "_" + out["Compound"]
    out["Driver_Year"] = out["Driver"] + "_" + out["Year"].astype(str)
    out["Compound_Stint"] = out["Compound"] + "_S" + out["Stint"].astype(str)
    out["Race_Stint"] = out["Race"] + "_S" + out["Stint"].astype(str)
    out["Race_PitStop"] = out["Race"] + "_P" + out["PitStop"].astype(str)

    out["TyreLife_bin"] = make_bin(out["TyreLife"], [-1, 3, 6, 10, 15, 22, 30, 45, 80, 200], "tyre")
    out["LapNumber_bin"] = make_bin(out["LapNumber"], [-1, 5, 12, 20, 30, 42, 55, 70, 120], "lap")
    out["RaceProgress_bin"] = make_bin(
        out["RaceProgress"], [-0.01, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.01], "progress"
    )
    out["Position_bin"] = make_bin(out["Position"], [0, 3, 6, 10, 14, 20, 99], "pos")
    out["LapsRemaining_bin"] = make_bin(out["LapsRemaining_est"], [-20, 0, 3, 8, 15, 25, 40, 120], "remaining")

    extra_cats = [
        "Race_Year",
        "Driver_Race_Year",
        "Race_Compound",
        "Race_Compound_Year",
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
    cat_cols.extend(extra_cats)

    for col in cat_cols:
        counts = out[col].value_counts(dropna=False)
        out[f"{col}_freq"] = out[col].map(counts).astype(float) / len(out)

    return out, cat_cols


def add_sequence_features(all_df: pd.DataFrame, include_lead: bool) -> pd.DataFrame:
    out = all_df.copy()
    out["_seq_order"] = np.arange(len(out))
    sort_cols = GROUP_COLS + ["LapNumber", ID_COL]
    seq = out.sort_values(sort_cols).copy()
    grouped = seq.groupby(GROUP_COLS, sort=False, dropna=False)

    seq["driver_race_lap_index"] = grouped.cumcount()
    seq["driver_race_n_laps"] = grouped["LapNumber"].transform("size")
    seq["driver_race_lap_frac"] = safe_divide(seq["driver_race_lap_index"], seq["driver_race_n_laps"] - 1)
    seq["lap_gap_lag1"] = seq["LapNumber"] - grouped["LapNumber"].shift(1)
    seq["lap_gap_lead1"] = grouped["LapNumber"].shift(-1) - seq["LapNumber"]

    for col in SEQUENCE_NUMERIC_COLS:
        base = clean_name(col)
        seq[f"{base}_lag1"] = grouped[col].shift(1)
        seq[f"{base}_lag2"] = grouped[col].shift(2)
        seq[f"{base}_diff_lag1"] = seq[col] - seq[f"{base}_lag1"]
        seq[f"{base}_diff_lag2"] = seq[col] - seq[f"{base}_lag2"]
        seq[f"{base}_roll3_mean_lag"] = grouped[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        seq[f"{base}_roll3_std_lag"] = grouped[col].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
        seq[f"{base}_roll5_mean_lag"] = grouped[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        seq[f"{base}_minus_roll3_mean_lag"] = seq[col] - seq[f"{base}_roll3_mean_lag"]
        if include_lead:
            seq[f"{base}_lead1"] = grouped[col].shift(-1)
            seq[f"{base}_lead1_diff"] = seq[f"{base}_lead1"] - seq[col]

    seq = seq.sort_values("_seq_order").drop(columns=["_seq_order"])
    return seq.reset_index(drop=True)


def add_external_lookup_features(
    all_df: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = all_df.copy()
    hit_stats: list[dict[str, object]] = []
    agg_candidates = [TARGET, "Normalized_TyreLife", LAPTIME, "LapTime_Delta", "Cumulative_Degradation", "PitStop"]

    for name, keys in EXTERNAL_KEY_SETS:
        missing = [col for col in keys if col not in external.columns or col not in out.columns]
        if missing:
            continue

        agg_spec: dict[str, list[str] | str] = {TARGET: ["mean", "count"]}
        for col in agg_candidates:
            if col in external.columns and col not in keys and col != TARGET:
                agg_spec[col] = "mean"

        stats = external.groupby(keys, dropna=False).agg(agg_spec)
        flat_cols: list[str] = []
        for col in stats.columns:
            if isinstance(col, tuple):
                base, stat = col
                if base == TARGET and stat == "mean":
                    flat_cols.append(f"ext_{name}_target_mean")
                elif base == TARGET and stat == "count":
                    flat_cols.append(f"ext_{name}_count")
                else:
                    flat_cols.append(f"ext_{name}_{clean_name(base)}_{stat}")
            else:
                flat_cols.append(f"ext_{name}_{clean_name(str(col))}")
        stats.columns = flat_cols
        stats = stats.reset_index()

        out = out.merge(stats, on=keys, how="left")
        target_col = f"ext_{name}_target_mean"
        count_col = f"ext_{name}_count"
        hit_col = f"ext_{name}_hit"
        out[hit_col] = out[target_col].notna().astype(float)
        if count_col in out.columns:
            out[count_col] = out[count_col].fillna(0).astype(float)

        train_mask = out["_split"].eq("train")
        test_mask = out["_split"].eq("test")
        hit_stats.append(
            {
                "name": name,
                "keys": keys,
                "n_groups": int(len(stats)),
                "train_hit_rate": float(out.loc[train_mask, hit_col].mean()),
                "test_hit_rate": float(out.loc[test_mask, hit_col].mean()),
                "train_mean_count": float(out.loc[train_mask, count_col].mean()) if count_col in out.columns else None,
                "test_mean_count": float(out.loc[test_mask, count_col].mean()) if count_col in out.columns else None,
            }
        )

    return out, hit_stats


def load_frames(data_dir: Path, external_csv: Path, max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extract_zip_if_needed(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    external = pd.read_csv(external_csv)
    if max_rows:
        train = train.sample(max_rows, random_state=42).sort_index().reset_index(drop=True)
    return normalize_columns(train), normalize_columns(test), normalize_columns(external)


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    include_lead: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, object]]]:
    y = train[TARGET].astype(int).reset_index(drop=True)
    train_x = train.drop(columns=[TARGET]).copy()
    test_x = test.copy()
    train_x["_split"] = "train"
    test_x["_split"] = "test"
    train_x["_row"] = np.arange(len(train_x))
    test_x["_row"] = np.arange(len(test_x))

    all_df = pd.concat([train_x, test_x], ignore_index=True)
    all_df, cat_cols = add_row_features(all_df)
    all_df = add_sequence_features(all_df, include_lead=include_lead)
    all_df, hit_stats = add_external_lookup_features(all_df, external)
    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    train_features = all_df[all_df["_split"].eq("train")].sort_values("_row").reset_index(drop=True)
    test_features = all_df[all_df["_split"].eq("test")].sort_values("_row").reset_index(drop=True)
    train_features[TARGET] = y

    drop_cols = {TARGET, "_split", "_row"}
    feature_cols = [col for col in test_features.columns if col not in drop_cols]
    cat_cols = [col for col in cat_cols if col in feature_cols]
    return train_features[feature_cols + [TARGET]], test_features[feature_cols], cat_cols, hit_stats


def add_target_encoding(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols: list[str],
    smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    global_mean = float(y_train.mean())
    x_train = x_train.copy()
    x_valid = x_valid.copy()
    x_test = x_test.copy()
    new_cols: list[str] = []

    for col in cat_cols:
        stats = y_train.groupby(x_train[col].astype(str), observed=True).agg(["mean", "count"])
        encoded = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        new_col = f"{col}_te"
        x_train[new_col] = x_train[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        x_valid[new_col] = x_valid[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        x_test[new_col] = x_test[col].astype(str).map(encoded).fillna(global_mean).astype(float)
        new_cols.append(new_col)

    return x_train, x_valid, x_test, new_cols


def align_categories(frames: list[pd.DataFrame], cat_cols: list[str]) -> None:
    for col in cat_cols:
        categories = pd.Index(
            pd.concat([frame[col].astype(str).fillna("__NA__") for frame in frames], ignore_index=True).unique()
        )
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[col] = frame[col].astype(str).fillna("__NA__").astype(dtype)


def make_splits(
    y: pd.Series,
    train_features: pd.DataFrame,
    cv_kind: str,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cv_kind == "group":
        groups = train_features["Year"].astype(str) + "_" + train_features["Race"].astype(str)
        return list(GroupKFold(n_splits=n_splits).split(train_features, y, groups=groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_features, y))


def train_lgbm(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    cat_cols: list[str],
    cv_kind: str,
    n_splits: int,
    seed: int,
    target_encoding: bool,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[dict[str, object]]]:
    y = train_features[TARGET].astype(int)
    features = [col for col in test_features.columns if col != TARGET]
    splits = make_splits(y, train_features, cv_kind=cv_kind, n_splits=n_splits, seed=seed)
    oof = np.zeros(len(train_features), dtype=float)
    test_pred = np.zeros(len(test_features), dtype=float)
    scores: list[float] = []
    fold_info: list[dict[str, object]] = []

    for fold, (trn_idx, val_idx) in enumerate(splits, start=1):
        print(f"[lgbm] fold {fold}/{n_splits}", flush=True)
        x_train = train_features.iloc[trn_idx][features].copy()
        x_valid = train_features.iloc[val_idx][features].copy()
        x_test = test_features[features].copy()
        y_train = y.iloc[trn_idx]
        y_valid = y.iloc[val_idx]
        fold_features = features.copy()

        if target_encoding:
            x_train, x_valid, x_test, te_cols = add_target_encoding(
                x_train, y_train, x_valid, x_test, cat_cols=cat_cols, smooth=40.0
            )
            fold_features.extend(te_cols)

        fold_cat_cols = [col for col in cat_cols if col in fold_features]
        align_categories([x_train, x_valid, x_test], fold_cat_cols)

        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=-1,
            min_child_samples=120,
            subsample=0.86,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.4,
            reg_lambda=5.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            x_train[fold_features],
            y_train,
            eval_set=[(x_valid[fold_features], y_valid)],
            categorical_feature=fold_cat_cols,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)],
        )
        valid_pred = model.predict_proba(x_valid[fold_features])[:, 1]
        test_pred += model.predict_proba(x_test[fold_features])[:, 1] / n_splits
        oof[val_idx] = valid_pred
        score = float(roc_auc_score(y_valid, valid_pred))
        scores.append(score)
        fold_info.append({"fold": fold, "auc": score, "best_iteration": int(model.best_iteration_ or n_estimators)})
        print(f"[lgbm] fold {fold} AUC {score:.8f}", flush=True)

    return oof, test_pred, scores, fold_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence features plus external exact lookup CV for S6E5.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="sequence_features_lgbm_v1")
    parser.add_argument("--cv", choices=["stratified", "group"], default="stratified")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--include-lead", action="store_true")
    parser.add_argument("--target-encoding", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=3500)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=96)
    args = parser.parse_args()

    train, test, external = load_frames(args.data_dir, args.external_csv, max_rows=args.max_rows)
    train_features, test_features, cat_cols, hit_stats = build_features(train, test, external, include_lead=args.include_lead)
    print(
        json.dumps(
            {
                "train_shape": train_features.shape,
                "test_shape": test_features.shape,
                "n_cat_cols": len(cat_cols),
                "include_lead": args.include_lead,
                "cv": args.cv,
                "external_hit_stats": hit_stats,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.diagnose_only:
        return

    oof, test_pred, scores, fold_info = train_lgbm(
        train_features=train_features,
        test_features=test_features,
        cat_cols=cat_cols,
        cv_kind=args.cv,
        n_splits=args.n_splits,
        seed=args.seed,
        target_encoding=args.target_encoding,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    y = train_features[TARGET].astype(int)
    overall_auc = float(roc_auc_score(y, oof))
    print(f"OOF AUC {overall_auc:.10f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    pred_col = TARGET if TARGET in sample.columns else sample.columns[-1]
    sample[pred_col] = np.clip(test_pred, 0.0, 1.0)
    sample.to_csv(submission_path, index=False)

    pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), TARGET: y.to_numpy(), "pred": oof}).to_csv(oof_path, index=False)
    summary = {
        "oof_auc": overall_auc,
        "fold_scores": scores,
        "fold_info": fold_info,
        "feature_count": int(test_features.shape[1]),
        "cat_cols": cat_cols,
        "include_lead": args.include_lead,
        "target_encoding": args.target_encoding,
        "cv": args.cv,
        "seed": args.seed,
        "external_hit_stats": hit_stats,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
