from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from external_prior_meta_stack import ID_COL, TARGET, logit, prediction_column, rank_pct


PREDICTION_SPECS = {
    "best": ("oof_external_prior_lgb_stack_v1_alpha0767.csv", "submission_external_prior_lgb_stack_v1_alpha0767.csv"),
    "lgb07": ("oof_external_prior_lgb_stack_v1.csv", "submission_external_prior_lgb_stack_v1.csv"),
    "sgd": ("oof_external_prior_meta_stack_v1.csv", "submission_external_prior_meta_stack_v1.csv"),
    "ext6": ("oof_extgreedy6_residual_risky.csv", "submission_extgreedy6_residual_risky.csv"),
    "ext5": ("oof_extgreedy5_residual_risky.csv", "submission_extgreedy5_residual_risky.csv"),
    "resid": ("oof_minimal_residual_stack_sgd_risky_seedmean.csv", "submission_minimal_residual_stack_sgd_risky_seedmean.csv"),
    "stable": (
        "oof_segment_seqbucket_yearlog_compound_normproxy.csv",
        "submission_segment_seqbucket_yearlog_compound_normproxy.csv",
    ),
    "race_te": ("oof_segment_seqbucket_yearlog_race_te.csv", "submission_segment_seqbucket_yearlog_race_te.csv"),
}


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


def build_wide_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
    min_match: float,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, str, int]:
    train_arrays: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    names: list[str] = []

    best_oof = pd.read_csv(output_dir / "oof_external_prior_lgb_stack_v1_alpha0767.csv")["pred"].to_numpy()
    best_sub = pd.read_csv(output_dir / "submission_external_prior_lgb_stack_v1_alpha0767.csv")
    best_test_col = prediction_column(best_sub)
    best_test = best_sub[best_test_col].to_numpy()

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
        add_feature(train_arrays, test_arrays, names, f"diff_best_{name}", best_oof - pred, best_test - test_pred)

    required = ["Race", "Year", "LapNumber"]
    extras = ["Position", "Stint", "TyreLife", "Compound", "PitStop", "Driver", "RaceProgress", "LapTime (s)"]
    key_count = 0
    for size in range(1, 7):
        for combo in itertools.combinations(extras, size):
            if sum(col in combo for col in ["RaceProgress", "LapTime (s)"]) > 1:
                continue
            keys = required + list(combo)
            key_name = "_".join(col.replace(" (s)", "s") for col in combo)
            stats = external.groupby(keys, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
            train_prior = train.merge(stats, on=keys, how="left")
            test_prior = test.merge(stats, on=keys, how="left")
            prior = train_prior["mean"].to_numpy()
            test_prior_values = test_prior["mean"].to_numpy()
            mask = ~np.isnan(prior)
            if float(mask.mean()) < min_match:
                continue
            fill_value = float(np.nanmean(prior))
            count = train_prior["count"].fillna(0).to_numpy()
            test_count = test_prior["count"].fillna(0).to_numpy()
            add_feature(train_arrays, test_arrays, names, f"prior_{key_name}", prior, test_prior_values)
            add_feature(
                train_arrays,
                test_arrays,
                names,
                f"mask_{key_name}",
                mask.astype(float),
                (~np.isnan(test_prior_values)).astype(float),
            )
            add_feature(train_arrays, test_arrays, names, f"count_{key_name}", count, test_count)
            add_feature(
                train_arrays,
                test_arrays,
                names,
                f"diff_{key_name}",
                np.nan_to_num(prior, nan=fill_value) - best_oof,
                np.nan_to_num(test_prior_values, nan=fill_value) - best_test,
            )
            key_count += 1

    return np.vstack(train_arrays).T, np.vstack(test_arrays).T, names, best_sub, best_test_col, key_count


def crossfit_lgb(
    x: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    seed: int,
    leaves: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    best_iterations: list[int] = []
    for fold, (trn_idx, val_idx) in enumerate(cv.split(x, y), start=1):
        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1600,
            learning_rate=0.02,
            num_leaves=leaves,
            min_child_samples=1000,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=2.0,
            reg_lambda=12.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(
                x[trn_idx],
                y[trn_idx],
                eval_set=[(x[val_idx], y[val_idx])],
                callbacks=[early_stopping(90, verbose=False), log_evaluation(0)],
            )
            oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
            test_pred += model.predict_proba(x_test)[:, 1] / n_splits
        best_iterations.append(int(model.best_iteration_ or model.n_estimators))
    return oof, test_pred, best_iterations


def best_blend(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, max_alpha: float) -> tuple[float, float, np.ndarray]:
    best_auc = float(roc_auc_score(y, base))
    best_alpha = 0.0
    best_pred = base.copy()
    for alpha in np.linspace(0.0, max_alpha, 501):
        pred = (1.0 - alpha) * base + alpha * candidate
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)
            best_pred = pred
    return best_alpha, best_auc, best_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide LightGBM stack over generated external-key priors.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_external_prior_lgb_stack_v1_alpha0767.csv", type=Path)
    parser.add_argument("--base-submission", default="submissions/submission_external_prior_lgb_stack_v1_alpha0767.csv", type=Path)
    parser.add_argument("--suffix", default="external_prior_lgb_wide_stack")
    parser.add_argument("--seeds", default="17,29,41,53,67")
    parser.add_argument("--leaves", default="31")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--max-alpha", default=1.0, type=float)
    parser.add_argument("--min-match", default=0.01, type=float)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    y = train[TARGET].astype(int).to_numpy()
    x, x_test, feature_names, fallback_submission, fallback_col, key_count = build_wide_features(
        train, test, external, args.output_dir, args.min_match
    )
    base_oof_df = pd.read_csv(args.base_oof)
    base_sub_df = pd.read_csv(args.base_submission)
    base_oof = base_oof_df[prediction_column(base_oof_df)].astype(float).to_numpy()
    base_test_col = prediction_column(base_sub_df)
    base_test = base_sub_df[base_test_col].astype(float).to_numpy()
    base_auc = float(roc_auc_score(y, base_oof))
    print(f"base {base_auc} features {x.shape} keys {key_count}", flush=True)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    leaves_values = [int(value) for value in args.leaves.split(",") if value.strip()]
    records: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for seed in seeds:
        for leaves in leaves_values:
            meta_oof, meta_test, best_iterations = crossfit_lgb(x, x_test, y, seed, leaves, args.n_splits)
            blend_alpha, blend_auc, blend_oof = best_blend(y, base_oof, meta_oof, args.max_alpha)
            blend_test = (1.0 - blend_alpha) * base_test + blend_alpha * meta_test
            row = {
                "seed": seed,
                "leaves": leaves,
                "base_auc": base_auc,
                "meta_auc": float(roc_auc_score(y, meta_oof)),
                "best_alpha_meta": blend_alpha,
                "blend_auc": blend_auc,
                "delta_vs_base": blend_auc - base_auc,
                "best_iterations": best_iterations,
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
            "leaves": args.leaves,
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

    submission = base_sub_df.copy() if base_sub_df is not None else fallback_submission.copy()
    submission[base_test_col or fallback_col] = np.clip(best["blend_test"], 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": best["blend_oof"]}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": best["meta_oof"]}).to_csv(meta_oof_path, index=False)

    summary = {
        "base_auc": base_auc,
        "best": {key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "results": results,
        "features": feature_names,
        "n_features": len(feature_names),
        "prior_key_count": key_count,
        "output": str(submission_path),
        "oof_output": str(oof_path),
        "meta_oof_output": str(meta_oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
