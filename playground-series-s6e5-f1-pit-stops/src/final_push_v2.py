"""Last shot at breaking into top 10. Drop weaker Nina files; only blend the
top-2 public CSVs with our diversity injection."""
from pathlib import Path
import numpy as np
import pandas as pd

ID_COL = "id"; TARGET = "PitNextLap"
NINA_DIR = Path("/tmp/public_oof/nina_ps-s6e5-01")

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))
def load(p): return pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)

nina_top = load(NINA_DIR / "0.95452.csv")
nina_2   = load(NINA_DIR / "0.95450.csv")
v4       = load("submissions/submission_public_blend_v4.csv")

ids = nina_top[ID_COL].to_numpy()
for d in [nina_2, v4]:
    assert (d[ID_COL].to_numpy() == ids).all()

# Skinny weights: lean heavily on best public, only small diversity injection
weights = {"nina_top": 0.75, "nina_2": 0.05, "v4": 0.20}
z = (weights["nina_top"] * logit(nina_top[TARGET].to_numpy())
   + weights["nina_2"]   * logit(nina_2[TARGET].to_numpy())
   + weights["v4"]       * logit(v4[TARGET].to_numpy()))
final = np.clip(sig(z), 0, 1)
pd.DataFrame({ID_COL: ids, TARGET: final}).to_csv("submissions/submission_final_push_v2.csv", index=False)
print(f"weights: {weights}  pred stats: mean={final.mean():.4f}")
