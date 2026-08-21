from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET = "PitNextLap"
PRED_COLS = ["pred", "blend", "catboost_realmlp_external"]


def prediction_column(df: pd.DataFrame) -> str:
    for col in PRED_COLS:
        if col in df.columns:
            return col
    candidates = [col for col in df.columns if col not in {ID_COL, TARGET}]
    if candidates:
        return candidates[-1]
    if TARGET in df.columns:
        return TARGET
    raise ValueError(f"Could not infer prediction column from {list(df.columns)}")


def load_oof(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = prediction_column(df)
    return df[col].astype(float).to_numpy()


def load_submission(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(path)
    col = prediction_column(df)
    return df, df[col].astype(float).to_numpy(), col


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def add_sequence_buckets(all_df: pd.DataFrame) -> None:
    yrsc = all_df.groupby(["Year", "Race", "Stint", "Compound"], dropna=False, sort=False)
    yrsc_lap_max = yrsc["LapNumber"].transform("max")
    all_df["yrsc_lap_to_end"] = yrsc_lap_max - all_df["LapNumber"]
    all_df["yrsc_group_n"] = yrsc[ID_COL].transform("size").astype(float)

    yrds = all_df.groupby(["Year", "Race", "Driver", "Stint"], dropna=False, sort=False)
    tyre_min = yrds["TyreLife"].transform("min")
    tyre_max = yrds["TyreLife"].transform("max")
    tyre_span = (tyre_max - tyre_min).replace(0, np.nan)
    all_df["yrds_tyre_progress"] = (all_df["TyreLife"] - tyre_min) / tyre_span
    all_df["yrds_tyre_to_end"] = tyre_max - all_df["TyreLife"]
    all_df["yrds_group_n"] = yrds[ID_COL].transform("size").astype(float)

    all_df["yrsc_lap_bucket"] = pd.cut(
        all_df["yrsc_lap_to_end"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 8, 13, 21, np.inf],
        labels=False,
    ).astype("Int64")
    all_df["yrds_tyre_progress_bucket"] = pd.cut(
        all_df["yrds_tyre_progress"],
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=False,
    ).astype("Int64")


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_raw.drop(columns=[TARGET]).copy()
    test_x = test_raw.copy()
    n_train = len(train_x)
    all_df = pd.concat([train_x, test_x], ignore_index=True)

    total_laps = all_df["LapNumber"] / all_df["RaceProgress"].replace(0, np.nan)
    all_df["TotalLaps_est"] = total_laps.clip(1, 120)
    all_df["LapsRemaining_est"] = all_df["TotalLaps_est"] - all_df["LapNumber"]
    all_df["TyreLife_per_LapNumber"] = all_df["TyreLife"] / all_df["LapNumber"].replace(0, np.nan)
    all_df["Degradation_per_TyreLife"] = all_df["Cumulative_Degradation"] / all_df["TyreLife"].replace(0, np.nan)
    all_df["LapTime_x_RaceProgress"] = all_df["LapTime (s)"] * all_df["RaceProgress"]
    add_sequence_buckets(all_df)

    numeric_cols = [
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "TyreLife",
        "Position",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Position_Change",
        "TotalLaps_est",
        "LapsRemaining_est",
        "TyreLife_per_LapNumber",
        "Degradation_per_TyreLife",
        "LapTime_x_RaceProgress",
        "yrsc_lap_to_end",
        "yrsc_group_n",
        "yrds_tyre_progress",
        "yrds_tyre_to_end",
        "yrds_group_n",
    ]
    cat_cols = [
        "Year",
        "Compound",
        "Stint",
        "PitStop",
        "yrsc_lap_bucket",
        "yrds_tyre_progress_bucket",
    ]
    feat = all_df[numeric_cols].copy()
    feat = pd.concat(
        [feat, pd.get_dummies(all_df[cat_cols].astype(str), prefix=cat_cols, dummy_na=True, dtype=float)],
        axis=1,
    )
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat.iloc[:n_train].reset_index(drop=True), feat.iloc[n_train:].reset_index(drop=True)


def add_prediction_features(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    output_dir: Path,
    include_risky: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, str]:
    specs = {
        "stable": (
            output_dir / "oof_segment_seqbucket_yearlog_compound_normproxy.csv",
            output_dir / "submission_segment_seqbucket_yearlog_compound_normproxy.csv",
        ),
        "seqbucket3": (
            output_dir / "oof_seqbucket3_yrds_late_tyre_prog_cat7070.csv",
            output_dir / "submission_seqbucket3_yrds_late_tyre_prog_cat7070.csv",
        ),
        "seqpipe": (
            output_dir / "oof_seq_pipeline_stable_v1.csv",
            output_dir / "submission_seq_pipeline_stable_v1.csv",
        ),
        "oldblend": (
            output_dir / "oof_blend_logstack_realmlp_catboost_seed7070.csv",
            output_dir / "submission_blend_logstack_realmlp_catboost_seed7070.csv",
        ),
        "best4": (
            output_dir / "oof_segment_best4_year_cat7070.csv",
            output_dir / "submission_segment_best4_year_cat7070.csv",
        ),
        "cat7070": (
            output_dir / "oof_catboost_realmlp_external_full_seed7070.csv",
            output_dir / "submission_catboost_realmlp_external_full_seed7070.csv",
        ),
        "norm": (
            output_dir / "oof_lgbm_normproxy_external.csv",
            output_dir / "submission_lgbm_normproxy_external.csv",
        ),
        "seed42": (
            output_dir / "oof_blend_seed42_cat.csv",
            output_dir / "submission_blend_seed42_cat.csv",
        ),
        "te": (
            output_dir / "oof_blend_oof_search_te.csv",
            output_dir / "submission_blend_oof_search_te.csv",
        ),
        "logstack": (
            output_dir / "oof_logstack_sweep.csv",
            output_dir / "submission_logstack_sweep.csv",
        ),
    }
    if include_risky:
        specs["race_te"] = (
            output_dir / "oof_segment_seqbucket_yearlog_race_te.csv",
            output_dir / "submission_segment_seqbucket_yearlog_race_te.csv",
        )
        specs["caloffset"] = (
            output_dir / "oof_caloffset_Compound_Stint.csv",
            output_dir / "submission_caloffset_Compound_Stint.csv",
        )

    oof_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    sample_submission: pd.DataFrame | None = None
    submission_col = TARGET
    for name, (oof_path, sub_path) in specs.items():
        if not oof_path.exists() or not sub_path.exists():
            continue
        oof = load_oof(oof_path)
        sub, test_pred, col = load_submission(sub_path)
        oof_preds[name] = oof
        test_preds[name] = test_pred
        sample_submission = sub
        submission_col = col

        train_feat[f"pred_{name}"] = oof
        test_feat[f"pred_{name}"] = test_pred
        train_feat[f"logit_{name}"] = logit(oof)
        test_feat[f"logit_{name}"] = logit(test_pred)
        train_feat[f"rank_{name}"] = rank_pct(oof)
        test_feat[f"rank_{name}"] = rank_pct(test_pred)

    for right in ["seqbucket3", "seqpipe", "oldblend", "best4", "cat7070", "norm", "seed42", "te", "logstack"]:
        if "stable" in oof_preds and right in oof_preds:
            train_feat[f"diff_stable_{right}"] = oof_preds["stable"] - oof_preds[right]
            test_feat[f"diff_stable_{right}"] = test_preds["stable"] - test_preds[right]

    if sample_submission is None:
        raise ValueError("No usable submission files found.")
    return train_feat, test_feat, oof_preds, test_preds, sample_submission, submission_col


def make_model(kind: str, strength: float, balanced: bool, seed: int):
    if kind == "sgd":
        return SGDClassifier(
            alpha=strength,
            average=True,
            class_weight="balanced" if balanced else None,
            early_stopping=True,
            loss="log_loss",
            max_iter=60,
            n_iter_no_change=4,
            n_jobs=-1,
            penalty="l2",
            random_state=seed,
            tol=1e-4,
            validation_fraction=0.08,
        )
    return LogisticRegression(
        C=strength,
        class_weight="balanced" if balanced else None,
        max_iter=350,
        n_jobs=-1,
        random_state=seed,
        solver="lbfgs",
    )


def crossfit(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    y: np.ndarray,
    kind: str,
    strength: float,
    balanced: bool,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test = np.zeros(len(test_feat), dtype=float)
    for fold, (trn_idx, val_idx) in enumerate(cv.split(train_feat, y), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            make_model(kind=kind, strength=strength, balanced=balanced, seed=seed + fold),
        )
        model.fit(train_feat.iloc[trn_idx], y[trn_idx])
        oof[val_idx] = model.predict_proba(train_feat.iloc[val_idx])[:, 1]
        test += model.predict_proba(test_feat)[:, 1] / n_splits
    return oof, test


def best_blend(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, max_alpha: float) -> tuple[float, float, np.ndarray]:
    best_score = float(roc_auc_score(y, base))
    best_alpha = 0.0
    best_pred = base.copy()
    for alpha in np.linspace(0.0, max_alpha, 401):
        pred = (1 - alpha) * base + alpha * candidate
        score = float(roc_auc_score(y, pred))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred = pred
    return best_alpha, best_score, best_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny cross-fitted residual stack.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="minimal_residual_stack")
    parser.add_argument("--kind", choices=["sgd", "logreg"], default="sgd")
    parser.add_argument("--strengths", default="0.00001,0.00003,0.0001,0.0003")
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--max-alpha", default=0.12, type=float)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--balanced-modes", choices=["both", "false", "true"], default="both")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    y = train_raw[TARGET].astype(int).to_numpy()

    train_feat, test_feat = build_features(train_raw, test_raw)
    train_feat, test_feat, oof_preds, test_preds, sample_submission, submission_col = add_prediction_features(
        train_feat=train_feat,
        test_feat=test_feat,
        output_dir=args.output_dir,
        include_risky=args.include_risky,
    )
    if "stable" not in oof_preds:
        raise ValueError("stable base files are required")

    base = oof_preds["stable"]
    base_test = test_preds["stable"]
    base_auc = float(roc_auc_score(y, base))
    strengths = [float(value) for value in args.strengths.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.balanced_modes == "both":
        balanced_values = [False, True]
    else:
        balanced_values = [args.balanced_modes == "true"]

    best: dict[str, object] | None = None
    results: list[dict[str, object]] = []
    meta_records: list[dict[str, object]] = []
    for seed in seeds:
        for strength in strengths:
            for balanced in balanced_values:
                meta_oof, meta_test = crossfit(
                    train_feat=train_feat,
                    test_feat=test_feat,
                    y=y,
                    kind=args.kind,
                    strength=strength,
                    balanced=balanced,
                    seed=seed,
                    n_splits=args.n_splits,
                )
                meta_auc = float(roc_auc_score(y, meta_oof))
                alpha, blend_auc, blend_oof = best_blend(y, base, meta_oof, args.max_alpha)
                blend_test = (1 - alpha) * base_test + alpha * meta_test
                row = {
                    "kind": args.kind,
                    "seed": seed,
                    "strength": strength,
                    "balanced": balanced,
                    "base_auc": base_auc,
                    "meta_auc": meta_auc,
                    "best_alpha_meta": alpha,
                    "blend_auc": blend_auc,
                    "delta_vs_base": blend_auc - base_auc,
                }
                print(json.dumps(row), flush=True)
                results.append(row)
                meta_records.append({"row": row, "meta_oof": meta_oof, "meta_test": meta_test})
                if best is None or blend_auc > best["blend_auc"]:
                    best = {
                        **row,
                        "meta_oof": meta_oof,
                        "meta_test": meta_test,
                        "blend_oof": blend_oof,
                        "blend_test": blend_test,
                    }

    if best is None:
        raise RuntimeError("No model was evaluated")

    if len(meta_records) > 1:
        mean_meta_oof = np.mean([record["meta_oof"] for record in meta_records], axis=0)
        mean_meta_test = np.mean([record["meta_test"] for record in meta_records], axis=0)
        alpha, blend_auc, blend_oof = best_blend(y, base, mean_meta_oof, args.max_alpha)
        blend_test = (1 - alpha) * base_test + alpha * mean_meta_test
        ensemble_row = {
            "kind": args.kind,
            "seed": "mean",
            "strength": args.strengths,
            "balanced": args.balanced_modes,
            "base_auc": base_auc,
            "meta_auc": float(roc_auc_score(y, mean_meta_oof)),
            "best_alpha_meta": alpha,
            "blend_auc": blend_auc,
            "delta_vs_base": blend_auc - base_auc,
            "members": len(meta_records),
        }
        print("ENSEMBLE", json.dumps(ensemble_row), flush=True)
        results.append(ensemble_row)
        if blend_auc > best["blend_auc"]:
            best = {
                **ensemble_row,
                "meta_oof": mean_meta_oof,
                "meta_test": mean_meta_test,
                "blend_oof": blend_oof,
                "blend_test": blend_test,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    meta_oof_path = args.output_dir / f"oof_meta_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = sample_submission.copy()
    submission[submission_col] = np.clip(best["blend_test"], 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y, "pred": best["blend_oof"]}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y, "pred": best["meta_oof"]}).to_csv(meta_oof_path, index=False)

    summary = {
        "base_auc": base_auc,
        "best": {key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "results": results,
        "features": list(train_feat.columns),
        "n_features": int(train_feat.shape[1]),
        "available_predictions": sorted(oof_preds),
        "output": str(submission_path),
        "oof_output": str(oof_path),
        "meta_oof_output": str(meta_oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
