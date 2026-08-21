from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from external_prior_meta_stack import ID_COL, TARGET, prediction_column


def read_prediction(path: Path) -> tuple[pd.DataFrame, str, np.ndarray]:
    frame = pd.read_csv(path)
    pred_col = prediction_column(frame)
    return frame, pred_col, frame[pred_col].astype(float).to_numpy()


def blend(base: np.ndarray, meta: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * base + alpha * meta, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend two OOF/submission prediction pairs with a fixed alpha.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_external_prior_meta_stack_v1.csv", type=Path)
    parser.add_argument("--base-submission", default="submissions/submission_external_prior_meta_stack_v1.csv", type=Path)
    parser.add_argument("--meta-oof", default="submissions/oof_meta_external_prior_lgb_stack_v1.csv", type=Path)
    parser.add_argument("--meta-submission", default="submissions/submission_external_prior_lgb_stack_v1.csv", type=Path)
    parser.add_argument("--meta-submission-blend-base", default=None, type=Path)
    parser.add_argument("--meta-submission-blend-alpha", default=None, type=float)
    parser.add_argument("--alpha", default=0.767, type=float)
    parser.add_argument("--suffix", default="external_prior_lgb_stack_v1_alpha0767")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    y = train[TARGET].astype(int).to_numpy()

    base_oof_df, _, base_oof = read_prediction(args.base_oof)
    _, _, meta_oof = read_prediction(args.meta_oof)
    base_sub_df, sub_col, base_test = read_prediction(args.base_submission)
    _, _, meta_test = read_prediction(args.meta_submission)
    if args.meta_submission_blend_base is not None:
        if args.meta_submission_blend_alpha is None or args.meta_submission_blend_alpha <= 0:
            raise ValueError("--meta-submission-blend-alpha must be positive when recovering a blended meta submission")
        _, _, meta_base_test = read_prediction(args.meta_submission_blend_base)
        if len(meta_base_test) != len(meta_test):
            raise ValueError("Blended meta submission base length does not match meta submission")
        beta = args.meta_submission_blend_alpha
        meta_test = (meta_test - (1.0 - beta) * meta_base_test) / beta

    if len(base_oof) != len(meta_oof) or len(base_oof) != len(y):
        raise ValueError("OOF prediction lengths do not match train.csv")
    if len(base_test) != len(meta_test):
        raise ValueError("Submission prediction lengths do not match")

    blended_oof = blend(base_oof, meta_oof, args.alpha)
    blended_test = blend(base_test, meta_test, args.alpha)
    base_auc = float(roc_auc_score(y, base_oof))
    meta_auc = float(roc_auc_score(y, meta_oof))
    blend_auc = float(roc_auc_score(y, blended_oof))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = base_sub_df.copy()
    submission[sub_col] = blended_test
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: base_oof_df[ID_COL], TARGET: y, "pred": blended_oof}).to_csv(oof_path, index=False)

    summary = {
        "base_oof": str(args.base_oof),
        "meta_oof": str(args.meta_oof),
        "alpha_meta": args.alpha,
        "base_auc": base_auc,
        "meta_auc": meta_auc,
        "blend_auc": blend_auc,
        "meta_submission_blend_base": str(args.meta_submission_blend_base)
        if args.meta_submission_blend_base is not None
        else None,
        "meta_submission_blend_alpha": args.meta_submission_blend_alpha,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
