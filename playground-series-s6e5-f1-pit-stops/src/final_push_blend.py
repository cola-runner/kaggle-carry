"""Final push: anchor on the strongest public CSV (Nina 0.95452.csv), inject our
v4 anchor for diversity. Tries to break the 0.95452 public ceiling.
"""
from __future__ import annotations

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


def load(path):
    return pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)


def main():
    nina_top = load(NINA_DIR / "0.95452.csv")
    nina_2 = load(NINA_DIR / "0.95450.csv")
    nina_3 = load(NINA_DIR / "0.95419.csv")
    nina_4 = load(NINA_DIR / "0.95411.csv")
    our_v4 = load("submissions/submission_public_blend_v4.csv")

    pc = TARGET
    ids = nina_top[ID_COL].to_numpy()
    for d in [nina_2, nina_3, nina_4, our_v4]:
        assert (d[ID_COL].to_numpy() == ids).all(), "id mismatch"

    # Weights (sum=1):  nina_top 0.60, nina_2 0.12, nina_3 0.05, nina_4 0.03, our_v4 0.20
    weights = {"nina_top": 0.60, "nina_2": 0.12, "nina_3": 0.05, "nina_4": 0.03, "our_v4": 0.20}
    z = (
        weights["nina_top"] * logit(nina_top[pc].to_numpy())
        + weights["nina_2"]  * logit(nina_2[pc].to_numpy())
        + weights["nina_3"]  * logit(nina_3[pc].to_numpy())
        + weights["nina_4"]  * logit(nina_4[pc].to_numpy())
        + weights["our_v4"]  * logit(our_v4[pc].to_numpy())
    )
    final = np.clip(sig(z), 0, 1)

    out = Path("submissions/submission_final_push_v1.csv")
    pd.DataFrame({ID_COL: ids, TARGET: final}).to_csv(out, index=False)
    print(f"weights: {weights}  (sum={sum(weights.values())})")
    print(f"pred stats: mean={final.mean():.4f} std={final.std():.4f}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
