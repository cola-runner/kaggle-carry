from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from external_prior_meta_stack import ID_COL, TARGET, build_features, logit, prediction_column, rank_pct


def load_oof(path: Path, train: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path)
    col = prediction_column(frame)
    if ID_COL in frame.columns:
        indexed = frame.set_index(ID_COL)
        return indexed.loc[train[ID_COL].to_numpy(), col].astype(float).to_numpy()
    if len(frame) != len(train):
        raise ValueError(f"{path} has no id column and length does not match train.csv")
    return frame[col].astype(float).to_numpy()


def load_submission(path: Path) -> tuple[pd.DataFrame, str, np.ndarray]:
    frame = pd.read_csv(path)
    col = prediction_column(frame)
    return frame, col, frame[col].astype(float).to_numpy()


def parse_extra_predictions(raw: str, output_dir: Path) -> list[tuple[str, Path, Path]]:
    specs: list[tuple[str, Path, Path]] = []
    if not raw.strip():
        return specs
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError("--extra-predictions must use name:oof_path:submission_path entries")
        name, oof_raw, sub_raw = parts
        oof_path = Path(oof_raw)
        sub_path = Path(sub_raw)
        if not oof_path.is_absolute():
            oof_path = output_dir / oof_path
        if not sub_path.is_absolute():
            sub_path = output_dir / sub_path
        specs.append((name, oof_path, sub_path))
    return specs


def append_prediction_features(
    x: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    train: pd.DataFrame,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    extras: list[tuple[str, Path, Path]],
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame | None, str | None]:
    train_arrays: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    names: list[str] = []
    sample_submission: pd.DataFrame | None = None
    submission_col: str | None = None

    def add(name: str, train_values: np.ndarray, test_values: np.ndarray) -> None:
        train_arrays.append(train_values.astype(float))
        test_arrays.append(test_values.astype(float))
        names.append(name)

    add("pred_base_current", base_oof, base_test)
    add("logit_base_current", logit(base_oof), logit(base_test))
    add("rank_base_current", rank_pct(base_oof), rank_pct(base_test))

    for name, oof_path, sub_path in extras:
        if not oof_path.exists() or not sub_path.exists():
            raise FileNotFoundError(f"Missing extra prediction pair for {name}: {oof_path}, {sub_path}")
        pred = load_oof(oof_path, train)
        sub_frame, col, test_pred = load_submission(sub_path)
        sample_submission = sub_frame
        submission_col = col

        diff = pred - base_oof
        test_diff = test_pred - base_test
        add(f"pred_{name}", pred, test_pred)
        add(f"logit_{name}", logit(pred), logit(test_pred))
        add(f"rank_{name}", rank_pct(pred), rank_pct(test_pred))
        add(f"diff_base_{name}", diff, test_diff)
        add(f"absdiff_base_{name}", np.abs(diff), np.abs(test_diff))
        add(f"avg_base_{name}", 0.5 * (base_oof + pred), 0.5 * (base_test + test_pred))

    if train_arrays:
        x = np.column_stack([x, *train_arrays])
        x_test = np.column_stack([x_test, *test_arrays])
        feature_names = [*feature_names, *names]
    return x, x_test, feature_names, sample_submission, submission_col


def crossfit_lgb(
    x: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    seed: int,
    leaves: int,
    n_splits: int,
    n_estimators: int,
    learning_rate: float,
    min_child_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    best_iterations: list[int] = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(x, y), start=1):
        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=leaves,
            min_child_samples=min_child_samples,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=1.5,
            reg_lambda=10.0,
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
    parser = argparse.ArgumentParser(description="LightGBM meta stack with external nearest-neighbor predictions.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_external_nn_greedy_v1.csv", type=Path)
    parser.add_argument("--base-submission", default="submissions/submission_external_nn_greedy_v1.csv", type=Path)
    parser.add_argument(
        "--extra-predictions",
        default="nn_greedy:oof_external_nn_greedy_v1.csv:submission_external_nn_greedy_v1.csv",
    )
    parser.add_argument("--suffix", default="external_nn_meta_lgb_stack")
    parser.add_argument("--seeds", default="101,202")
    parser.add_argument("--leaves", default="7,15")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--n-estimators", default=1400, type=int)
    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--min-child-samples", default=900, type=int)
    parser.add_argument("--max-alpha", default=0.8, type=float)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    external = pd.read_csv(args.external_csv)
    y = train[TARGET].astype(int).to_numpy()

    x, x_test, feature_names, fallback_submission, fallback_col = build_features(train, test, external, args.output_dir)
    base_oof = load_oof(args.base_oof, train)
    base_sub_df, base_sub_col, base_test = load_submission(args.base_submission)
    base_auc = float(roc_auc_score(y, base_oof))
    extras = parse_extra_predictions(args.extra_predictions, args.output_dir)
    x, x_test, feature_names, extra_submission, extra_col = append_prediction_features(
        x=x,
        x_test=x_test,
        feature_names=feature_names,
        train=train,
        base_oof=base_oof,
        base_test=base_test,
        extras=extras,
    )
    print(f"base {base_auc} features {x.shape} extras {[name for name, _, _ in extras]}", flush=True)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    leaves_values = [int(value) for value in args.leaves.split(",") if value.strip()]
    records: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for seed in seeds:
        for leaves in leaves_values:
            meta_oof, meta_test, best_iterations = crossfit_lgb(
                x=x,
                x_test=x_test,
                y=y,
                seed=seed,
                leaves=leaves,
                n_splits=args.n_splits,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                min_child_samples=args.min_child_samples,
            )
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

    submission = extra_submission.copy() if extra_submission is not None else base_sub_df.copy()
    sub_col = extra_col or base_sub_col or fallback_col
    submission[sub_col] = np.clip(best["blend_test"], 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": np.clip(best["blend_oof"], 0.0, 1.0)}).to_csv(
        oof_path, index=False
    )
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "pred": np.clip(best["meta_oof"], 0.0, 1.0)}).to_csv(
        meta_oof_path, index=False
    )

    summary = {
        "base_oof": str(args.base_oof),
        "base_submission": str(args.base_submission),
        "base_auc": base_auc,
        "best": {key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "results": results,
        "features": feature_names,
        "n_features": len(feature_names),
        "extra_predictions": [name for name, _, _ in extras],
        "output": str(submission_path),
        "oof_output": str(oof_path),
        "meta_oof_output": str(meta_oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAVED", json.dumps(summary["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
