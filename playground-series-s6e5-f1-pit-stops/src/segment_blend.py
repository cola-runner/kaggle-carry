from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def load_oof(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    col = prediction_column(df)
    if df[col].isna().any():
        raise ValueError(f"{path} contains missing OOF predictions.")
    return df[TARGET].astype(int).to_numpy(), df[col].astype(float).to_numpy()


def load_submission(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(path)
    col = prediction_column(df)
    return df, df[col].astype(float).to_numpy(), col


def optimize_segment_weights(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    segment: pd.Series,
    grid: np.ndarray,
    min_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    out = base.copy()
    weights: dict[str, float] = {}

    for value, index in segment.groupby(segment, dropna=False).groups.items():
        idx = np.asarray(index)
        if len(idx) < min_rows or np.unique(y[idx]).size < 2:
            continue

        best_score = roc_auc_score(y[idx], base[idx])
        best_alpha = 1.0
        for alpha in grid:
            pred = alpha * base[idx] + (1.0 - alpha) * candidate[idx]
            score = roc_auc_score(y[idx], pred)
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)

        out[idx] = best_alpha * base[idx] + (1.0 - best_alpha) * candidate[idx]
        weights[str(value)] = best_alpha

    return out, weights


def apply_segment_weights(
    base: np.ndarray,
    candidate: np.ndarray,
    segment: pd.Series,
    weights: dict[str, float],
) -> np.ndarray:
    out = base.copy()
    for value, alpha in weights.items():
        mask = segment.astype(str).eq(value).to_numpy()
        out[mask] = alpha * base[mask] + (1.0 - alpha) * candidate[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-degree segment blend for F1 pit stop submissions.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_blend_logstack_realmlp_catboost_seed7070.csv", type=Path)
    parser.add_argument(
        "--base-submission",
        default="submissions/submission_blend_logstack_realmlp_catboost_seed7070.csv",
        type=Path,
    )
    parser.add_argument("--candidate-oof", default="submissions/oof_lgbm_normproxy_external.csv", type=Path)
    parser.add_argument("--candidate-submission", default="submissions/submission_lgbm_normproxy_external.csv", type=Path)
    parser.add_argument("--segment", default="Year")
    parser.add_argument("--min-rows", default=5000, type=int)
    parser.add_argument("--grid-size", default=1001, type=int)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="segment_year_normproxy")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y_base, base_oof = load_oof(args.base_oof)
    y_candidate, candidate_oof = load_oof(args.candidate_oof)
    if not np.array_equal(y_base, y_candidate):
        raise ValueError("OOF target columns do not match.")

    base_submission, base_test, pred_col = load_submission(args.base_submission)
    _, candidate_test, _ = load_submission(args.candidate_submission)
    if len(base_test) != len(candidate_test):
        raise ValueError("Submission files have different row counts.")

    grid = np.linspace(0.0, 1.0, args.grid_size)
    blended_oof, weights = optimize_segment_weights(
        y=y_base,
        base=base_oof,
        candidate=candidate_oof,
        segment=train[args.segment],
        grid=grid,
        min_rows=args.min_rows,
    )
    blended_test = apply_segment_weights(
        base=base_test,
        candidate=candidate_test,
        segment=test[args.segment],
        weights=weights,
    )

    base_auc = float(roc_auc_score(y_base, base_oof))
    candidate_auc = float(roc_auc_score(y_base, candidate_oof))
    blend_auc = float(roc_auc_score(y_base, blended_oof))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = base_submission.copy()
    submission[pred_col] = np.clip(blended_test, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)

    pd.DataFrame({ID_COL: train[ID_COL].values, TARGET: y_base, "pred": blended_oof}).to_csv(oof_path, index=False)

    summary = {
        "segment": args.segment,
        "base_oof_auc": base_auc,
        "candidate_oof_auc": candidate_auc,
        "blend_oof_auc": blend_auc,
        "weights_alpha_base": weights,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
