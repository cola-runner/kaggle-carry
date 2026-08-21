from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from external_prior_meta_stack import ID_COL, TARGET, build_features, prediction_column


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
            n_estimators=1400,
            learning_rate=0.022,
            num_leaves=leaves,
            min_child_samples=900,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=8.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1,
        )
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
    parser = argparse.ArgumentParser(description="LightGBM meta stack over external-key priors.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_external_prior_meta_stack_v1.csv", type=Path)
    parser.add_argument("--base-submission", default="submissions/submission_external_prior_meta_stack_v1.csv", type=Path)
    parser.add_argument("--suffix", default="external_prior_lgb_stack")
    parser.add_argument("--seeds", default="17,29,41,53,67")
    parser.add_argument("--leaves", default="7,15")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--max-alpha", default=0.7, type=float)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    y = train[TARGET].astype(int).to_numpy()
    x, x_test, feature_names, fallback_submission, _ = build_features(train, test, external, args.output_dir)

    base_oof_df = pd.read_csv(args.base_oof)
    base_sub_df = pd.read_csv(args.base_submission)
    base_oof = base_oof_df[prediction_column(base_oof_df)].astype(float).to_numpy()
    base_test_col = prediction_column(base_sub_df)
    base_test = base_sub_df[base_test_col].astype(float).to_numpy()
    base_auc = float(roc_auc_score(y, base_oof))
    print(f"base {base_auc} features {x.shape}", flush=True)

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
    submission[base_test_col] = np.clip(best["blend_test"], 0.0, 1.0)
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
