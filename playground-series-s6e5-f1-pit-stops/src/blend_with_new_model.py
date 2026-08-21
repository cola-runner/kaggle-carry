from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET = "PitNextLap"


def prediction_column(df: pd.DataFrame) -> str:
    preferred = ["pred", "blend", "catboost_realmlp_external", TARGET]
    for col in preferred:
        if col in df.columns and col != ID_COL:
            return col
    candidates = [col for col in df.columns if col not in {ID_COL, TARGET}]
    if not candidates:
        raise ValueError(f"Could not infer prediction column from {list(df.columns)}")
    return candidates[-1]


def load_oof(path: Path) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(path)
    pred_col = prediction_column(df)
    if df[pred_col].isna().any():
        missing = int(df[pred_col].isna().sum())
        raise ValueError(f"{path} has {missing} missing OOF predictions; use a full-CV OOF file.")
    return df[TARGET].astype(int), df[pred_col].astype(float).to_numpy()


def load_submission(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(path)
    pred_col = prediction_column(df)
    return df, df[pred_col].astype(float).to_numpy(), pred_col


def pct_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def optimize_two_way(
    y: np.ndarray,
    first_oof: np.ndarray,
    second_oof: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
    mode: str,
) -> dict[str, object]:
    best_score = -1.0
    best_alpha = 0.0
    best_oof = None
    best_test = None

    for alpha in np.linspace(0, 1, 1001):
        pred = alpha * first_oof + (1 - alpha) * second_oof
        score = roc_auc_score(y, pred)
        if score > best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_oof = pred
            best_test = alpha * first_test + (1 - alpha) * second_test

    return {
        "mode": mode,
        "score": best_score,
        "alpha_current": best_alpha,
        "alpha_new": float(1 - best_alpha),
        "oof": best_oof,
        "test": best_test,
    }


def logistic_stack(
    y: np.ndarray,
    current_oof: np.ndarray,
    new_oof: np.ndarray,
    current_test: np.ndarray,
    new_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    X = np.column_stack([current_oof, new_oof, pct_rank(current_oof), pct_rank(new_oof)])
    X_test = np.column_stack([current_test, new_test, pct_rank(current_test), pct_rank(new_test)])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    for train_idx, valid_idx in folds.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_valid = scaler.transform(X[valid_idx])
        X_test_fold = scaler.transform(X_test)
        model = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed)
        model.fit(X_train, y[train_idx])
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_pred += model.predict_proba(X_test_fold)[:, 1] / folds.n_splits

    return {
        "mode": "logistic_stack",
        "score": float(roc_auc_score(y, oof)),
        "alpha_current": None,
        "alpha_new": None,
        "oof": oof,
        "test": test_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend current best submission with a new full-CV model.")
    parser.add_argument("--current-oof", default="submissions/oof_logstack_sweep.csv", type=Path)
    parser.add_argument("--current-submission", default="submissions/submission_logstack_sweep.csv", type=Path)
    parser.add_argument("--new-oof", required=True, type=Path)
    parser.add_argument("--new-submission", required=True, type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="new_model_blend")
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    y_current, current_oof = load_oof(args.current_oof)
    y_new, new_oof = load_oof(args.new_oof)
    if not np.array_equal(y_current.to_numpy(), y_new.to_numpy()):
        raise ValueError("OOF target columns do not match.")
    y = y_current.to_numpy()

    current_submission, current_test, current_pred_col = load_submission(args.current_submission)
    new_submission, new_test, _ = load_submission(args.new_submission)
    if len(current_test) != len(new_test):
        raise ValueError("Submission files have different row counts.")

    current_score = float(roc_auc_score(y, current_oof))
    new_score = float(roc_auc_score(y, new_oof))
    raw = optimize_two_way(y, current_oof, new_oof, current_test, new_test, mode="raw_blend")
    rank = optimize_two_way(
        y,
        pct_rank(current_oof),
        pct_rank(new_oof),
        pct_rank(current_test),
        pct_rank(new_test),
        mode="rank_blend",
    )
    stack = logistic_stack(y, current_oof, new_oof, current_test, new_test, seed=args.seed)
    candidates = [raw, rank, stack]
    best = max(candidates, key=lambda item: item["score"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = current_submission.copy()
    submission[current_pred_col] = np.clip(best["test"], 0, 1)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: pd.read_csv(args.new_oof)[ID_COL].values, TARGET: y, "pred": best["oof"]}).to_csv(
        oof_path, index=False
    )

    summary = {
        "current_oof_auc": current_score,
        "new_oof_auc": new_score,
        "candidates": [
            {
                "mode": candidate["mode"],
                "score": candidate["score"],
                "alpha_current": candidate["alpha_current"],
                "alpha_new": candidate["alpha_new"],
            }
            for candidate in candidates
        ],
        "best_mode": best["mode"],
        "best_score": best["score"],
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
