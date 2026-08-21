"""Blend leveraging top-10 team public notebook outputs as new diverse members.

Public-notebook submissions from top-10 LB teams (Ak rank 6 LB 0.95464,
Cyril rank 7 LB 0.95461) are valid blend components — their notebook outputs
are public artifacts. Cyril's AutoGluon submission is notably less correlated
with the rest (0.972 vs 0.985+), so it should contribute real diversity.

We mix in test space (no OOF available for these notebook outputs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET = "PitNextLap"


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load(path):
    return pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-cyril", default=0.25, type=float)
    parser.add_argument("--w-ak", default=0.20, type=float)
    parser.add_argument("--w-nina-top", default=0.30, type=float)
    parser.add_argument("--w-v6", default=0.15, type=float)
    parser.add_argument("--w-nina-2", default=0.05, type=float)
    parser.add_argument("--w-masaya-cat", default=0.05, type=float,
                        help="Test-space file (the masayakawamata cat test sub) for added GBDT diversity.")
    parser.add_argument("--suffix", default="top10_blend_v1")
    args = parser.parse_args()

    cyril = load("/tmp/public_oof/cyrilbourgeois_s6-e05-autogluon-initial-test/submission.csv")
    ak    = load("/tmp/public_oof/abisheksrivastav_lightgbm-xgboost-catboost-ensemble/submission.csv")
    nina_top = load("/tmp/public_oof/nina_ps-s6e5-01/0.95452.csv")
    nina_2   = load("/tmp/public_oof/nina_ps-s6e5-01/0.95450.csv")
    v6 = load("submissions/submission_public_blend_v6.csv")
    masaya_cat_test = load("/tmp/public_oof/masayakawamata_s6e5-cat-with-fe/test_cat_0.9527003594478181.csv")

    ids = cyril[ID_COL].to_numpy()
    for d in [ak, nina_top, nina_2, v6, masaya_cat_test]:
        assert (d[ID_COL].to_numpy() == ids).all(), "id mismatch"

    w_total = (args.w_cyril + args.w_ak + args.w_nina_top + args.w_nina_2 + args.w_v6 + args.w_masaya_cat)
    if abs(w_total - 1.0) > 1e-6:
        raise SystemExit(f"weights must sum to 1.0, got {w_total}")

    z = (
        args.w_cyril * logit(cyril[TARGET].to_numpy())
        + args.w_ak * logit(ak[TARGET].to_numpy())
        + args.w_nina_top * logit(nina_top[TARGET].to_numpy())
        + args.w_nina_2 * logit(nina_2[TARGET].to_numpy())
        + args.w_v6 * logit(v6[TARGET].to_numpy())
        + args.w_masaya_cat * logit(masaya_cat_test[TARGET].to_numpy())
    )
    final = np.clip(sig(z), 0.0, 1.0)

    out = Path(f"submissions/submission_{args.suffix}.csv")
    pd.DataFrame({ID_COL: ids, TARGET: final}).to_csv(out, index=False)
    print(f"weights: cyril={args.w_cyril}  ak={args.w_ak}  nina_top={args.w_nina_top}"
          f"  nina_2={args.w_nina_2}  v6={args.w_v6}  masaya_cat={args.w_masaya_cat}  sum={w_total:.4f}")
    print(f"pred stats: mean={final.mean():.4f}  std={final.std():.4f}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
