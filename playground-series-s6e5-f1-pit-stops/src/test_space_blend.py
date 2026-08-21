"""Test-space-only blend: combine our OOF-validated blend (v4) with top public
submission CSVs that have no OOF (Nina ps-s6e5-01 dataset).

Strategy:
  - Anchor on our `submission_public_blend_v4.csv` (OOF AUC 0.954698, expected LB ~0.9542).
  - Layer top public CSVs (e.g. 0.95452, 0.95450, 0.95419 from Nina dataset)
    via logit-space weighted average. Without OOF we use prior-LB scores as
    pseudo-weights and submit a small grid to find the best practical mix.

Produces a single submission CSV. Pick the alpha grid carefully (we only have
a handful of daily Kaggle submission slots).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET = "PitNextLap"

NINA_DIR = Path("/tmp/public_oof/nina_ps-s6e5-01")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(ID_COL).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="submissions/submission_public_blend_v4.csv", type=Path)
    parser.add_argument("--nina-files", default="0.95452.csv,0.95450.csv,0.95419.csv,0.95411.csv,0.95405.csv,0.95403.csv,0.95397.csv",
                        help="Comma-separated Nina files to mix in (relative to nina dir).")
    parser.add_argument("--nina-weights", default="0.30,0.18,0.12,0.10,0.07,0.06,0.05",
                        help="Relative weights for nina files (sum normalised).")
    parser.add_argument("--anchor-weight", default=0.60, type=float)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="test_blend_v1")
    args = parser.parse_args()

    anchor = load_sub(args.anchor)
    pred_col = [c for c in anchor.columns if c != ID_COL][0]
    ids = anchor[ID_COL].to_numpy()

    nina_files = [f.strip() for f in args.nina_files.split(",") if f.strip()]
    nina_weights = np.array([float(w) for w in args.nina_weights.split(",") if w.strip()])
    assert len(nina_weights) == len(nina_files), "nina-files and nina-weights must match"

    nina_logits = []
    for fn in nina_files:
        d = load_sub(NINA_DIR / fn)
        assert (d[ID_COL].to_numpy() == ids).all(), f"id mismatch in {fn}"
        nina_logits.append(logit(d[pred_col].to_numpy()))

    nina_weights = nina_weights / nina_weights.sum() * (1 - args.anchor_weight)
    final_logit = args.anchor_weight * logit(anchor[pred_col].to_numpy())
    for w, l in zip(nina_weights, nina_logits):
        final_logit = final_logit + w * l
    final = np.clip(sig(final_logit), 0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"submission_{args.suffix}.csv"
    pd.DataFrame({ID_COL: ids, TARGET: final}).to_csv(out_path, index=False)

    print(f"anchor: {args.anchor.name}  weight {args.anchor_weight:.2f}")
    for fn, w in zip(nina_files, nina_weights):
        print(f"  + {fn}  effective weight {w:.4f}")
    print(f"wrote: {out_path}")
    print(f"prediction stats: mean={final.mean():.4f}  std={final.std():.4f}  min={final.min():.4f}  max={final.max():.4f}")


if __name__ == "__main__":
    main()
