from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET = "PitNextLap"
PRED_COLS = ["pred", "blend", "catboost_realmlp_external"]

PREDICTION_SPECS = {
    "ext6": ("oof_extgreedy6_residual_risky.csv", "submission_extgreedy6_residual_risky.csv"),
    "ext5": ("oof_extgreedy5_residual_risky.csv", "submission_extgreedy5_residual_risky.csv"),
    "ext4": ("oof_extgreedy4_residual_risky.csv", "submission_extgreedy4_residual_risky.csv"),
    "resid": ("oof_minimal_residual_stack_sgd_risky_seedmean.csv", "submission_minimal_residual_stack_sgd_risky_seedmean.csv"),
    "stable": (
        "oof_segment_seqbucket_yearlog_compound_normproxy.csv",
        "submission_segment_seqbucket_yearlog_compound_normproxy.csv",
    ),
    "seq3": ("oof_seqbucket3_yrds_late_tyre_prog_cat7070.csv", "submission_seqbucket3_yrds_late_tyre_prog_cat7070.csv"),
    "race_te": ("oof_segment_seqbucket_yearlog_race_te.csv", "submission_segment_seqbucket_yearlog_race_te.csv"),
    "cal": ("oof_caloffset_Compound_Stint.csv", "submission_caloffset_Compound_Stint.csv"),
}

PRIOR_KEYS = [
    ("k1", ["Race", "Year", "LapNumber", "Position", "Stint"]),
    ("k2", ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Compound"]),
    ("k3", ["Race", "Year", "LapNumber", "Position", "PitStop", "Compound"]),
    ("k4", ["Race", "Year", "LapNumber", "Stint", "Compound", "PitStop", "RaceProgress"]),
    ("k5", ["Race", "Year", "LapNumber", "Position", "Stint", "Compound", "RaceProgress"]),
    ("k6", ["Race", "Year", "LapNumber", "PitStop", "RaceProgress"]),
    ("k7", ["Race", "Year", "LapNumber", "Position", "Stint", "Compound", "PitStop", "RaceProgress"]),
    ("k8", ["Race", "Year", "LapNumber", "Compound", "PitStop", "RaceProgress"]),
    ("k9", ["Race", "Year", "LapNumber", "Position", "Compound", "PitStop", "RaceProgress"]),
    ("k10", ["Race", "Year", "LapNumber", "PitStop", "Driver", "RaceProgress"]),
    ("k11", ["Race", "Year", "LapNumber", "Compound", "PitStop", "Driver", "RaceProgress"]),
    ("k12", ["Race", "Year", "LapNumber", "Stint", "PitStop", "RaceProgress"]),
    ("k13", ["Race", "Year", "LapNumber", "RaceProgress"]),
    ("k14", ["Race", "Year", "LapNumber", "Position", "Compound", "RaceProgress"]),
    ("k15", ["Race", "Year", "LapNumber", "Stint", "Compound", "RaceProgress"]),
]


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


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def add_feature(
    train_arrays: list[np.ndarray],
    test_arrays: list[np.ndarray],
    names: list[str],
    name: str,
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> None:
    train_arrays.append(train_values.astype(float))
    test_arrays.append(test_values.astype(float))
    names.append(name)


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, str]:
    train_arrays: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    names: list[str] = []
    sample_submission: pd.DataFrame | None = None
    submission_col = TARGET

    base_oof = pd.read_csv(output_dir / "oof_extgreedy6_residual_risky.csv")["pred"].to_numpy()
    base_test_df = pd.read_csv(output_dir / "submission_extgreedy6_residual_risky.csv")
    base_test = base_test_df[prediction_column(base_test_df)].to_numpy()
    sample_submission = base_test_df
    submission_col = prediction_column(base_test_df)

    for name, (oof_file, sub_file) in PREDICTION_SPECS.items():
        oof_path = output_dir / oof_file
        sub_path = output_dir / sub_file
        if not oof_path.exists() or not sub_path.exists():
            continue
        oof_df = pd.read_csv(oof_path)
        sub_df = pd.read_csv(sub_path)
        pred = oof_df[prediction_column(oof_df)].astype(float).to_numpy()
        test_pred = sub_df[prediction_column(sub_df)].astype(float).to_numpy()
        add_feature(train_arrays, test_arrays, names, f"pred_{name}", pred, test_pred)
        add_feature(train_arrays, test_arrays, names, f"logit_{name}", logit(pred), logit(test_pred))
        add_feature(train_arrays, test_arrays, names, f"rank_{name}", rank_pct(pred), rank_pct(test_pred))
        add_feature(train_arrays, test_arrays, names, f"diff_ext6_{name}", base_oof - pred, base_test - test_pred)

    for name, keys in PRIOR_KEYS:
        stats = external.groupby(keys, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
        train_prior = train.merge(stats, on=keys, how="left")
        test_prior = test.merge(stats, on=keys, how="left")
        prior = train_prior["mean"].to_numpy()
        test_prior_values = test_prior["mean"].to_numpy()
        count = train_prior["count"].fillna(0).to_numpy()
        test_count = test_prior["count"].fillna(0).to_numpy()
        fill_value = float(np.nanmean(prior))
        add_feature(train_arrays, test_arrays, names, f"prior_{name}", prior, test_prior_values)
        add_feature(
            train_arrays,
            test_arrays,
            names,
            f"mask_{name}",
            (~np.isnan(prior)).astype(float),
            (~np.isnan(test_prior_values)).astype(float),
        )
        add_feature(train_arrays, test_arrays, names, f"count_{name}", count, test_count)
        add_feature(
            train_arrays,
            test_arrays,
            names,
            f"prior_diff_{name}",
            np.nan_to_num(prior, nan=fill_value) - base_oof,
            np.nan_to_num(test_prior_values, nan=fill_value) - base_test,
        )

    if sample_submission is None:
        raise ValueError("No sample submission was loaded")
    return np.vstack(train_arrays).T, np.vstack(test_arrays).T, names, sample_submission, submission_col


def crossfit(
    x: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    seed: int,
    alpha_sgd: float,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    for fold, (trn_idx, val_idx) in enumerate(cv.split(x, y), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha_sgd,
                average=True,
                early_stopping=True,
                validation_fraction=0.08,
                n_iter_no_change=4,
                max_iter=80,
                tol=1e-4,
                random_state=seed + fold,
                n_jobs=-1,
            ),
        )
        model.fit(x[trn_idx], y[trn_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1] / n_splits
    return oof, test_pred


def best_blend(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, max_alpha: float) -> tuple[float, float, np.ndarray]:
    best_auc = float(roc_auc_score(y, base))
    best_alpha = 0.0
    best_pred = base.copy()
    for alpha in np.linspace(0.0, max_alpha, 401):
        pred = (1.0 - alpha) * base + alpha * candidate
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)
            best_pred = pred
    return best_alpha, best_auc, best_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted meta model over external-key priors.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="external_prior_meta_stack")
    parser.add_argument("--seeds", default="17,29,41,53,67")
    parser.add_argument("--strengths", default="0.0001,0.0003")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--max-alpha", default=0.24, type=float)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    y = train[TARGET].astype(int).to_numpy()
    x, x_test, feature_names, sample_submission, submission_col = build_features(train, test, external, args.output_dir)
    base_oof = pd.read_csv(args.output_dir / "oof_extgreedy6_residual_risky.csv")["pred"].to_numpy()
    base_test = pd.read_csv(args.output_dir / "submission_extgreedy6_residual_risky.csv")[TARGET].to_numpy()
    base_auc = float(roc_auc_score(y, base_oof))
    print(f"base {base_auc} features {x.shape}", flush=True)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    strengths = [float(value) for value in args.strengths.split(",") if value.strip()]
    results: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for seed in seeds:
        for alpha_sgd in strengths:
            meta_oof, meta_test = crossfit(x, x_test, y, seed, alpha_sgd, args.n_splits)
            blend_alpha, blend_auc, blend_oof = best_blend(y, base_oof, meta_oof, args.max_alpha)
            blend_test = (1.0 - blend_alpha) * base_test + blend_alpha * meta_test
            row = {
                "seed": seed,
                "alpha_sgd": alpha_sgd,
                "base_auc": base_auc,
                "meta_auc": float(roc_auc_score(y, meta_oof)),
                "best_alpha_meta": blend_alpha,
                "blend_auc": blend_auc,
                "delta_vs_base": blend_auc - base_auc,
            }
            print(json.dumps(row), flush=True)
            results.append(row)
            records.append({"row": row, "meta_oof": meta_oof, "meta_test": meta_test})
            if best is None or blend_auc > best["blend_auc"]:
                best = {**row, "meta_oof": meta_oof, "meta_test": meta_test, "blend_oof": blend_oof, "blend_test": blend_test}

    if len(records) > 1:
        mean_meta_oof = np.mean([record["meta_oof"] for record in records], axis=0)
        mean_meta_test = np.mean([record["meta_test"] for record in records], axis=0)
        blend_alpha, blend_auc, blend_oof = best_blend(y, base_oof, mean_meta_oof, args.max_alpha)
        blend_test = (1.0 - blend_alpha) * base_test + blend_alpha * mean_meta_test
        row = {
            "seed": "mean",
            "alpha_sgd": args.strengths,
            "base_auc": base_auc,
            "meta_auc": float(roc_auc_score(y, mean_meta_oof)),
            "best_alpha_meta": blend_alpha,
            "blend_auc": blend_auc,
            "delta_vs_base": blend_auc - base_auc,
            "members": len(records),
        }
        print("ENSEMBLE", json.dumps(row), flush=True)
        results.append(row)
        if best is None or blend_auc > best["blend_auc"]:
            best = {
                **row,
                "meta_oof": mean_meta_oof,
                "meta_test": mean_meta_test,
                "blend_oof": blend_oof,
                "blend_test": blend_test,
            }

    if best is None:
        raise RuntimeError("No model was evaluated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    meta_oof_path = args.output_dir / f"oof_meta_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = sample_submission.copy()
    submission[submission_col] = np.clip(best["blend_test"], 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": best["blend_oof"]}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": best["meta_oof"]}).to_csv(meta_oof_path, index=False)

    summary = {
        "base_auc": base_auc,
        "best": {key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "results": results,
        "features": feature_names,
        "n_features": len(feature_names),
        "output": str(submission_path),
        "oof_output": str(oof_path),
        "meta_oof_output": str(meta_oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
