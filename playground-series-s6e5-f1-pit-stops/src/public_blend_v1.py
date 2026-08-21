"""Build a final submission by logit-blending public OOF/submission CSVs + own CatBoost.

Inputs (downloaded with `kaggle kernels output`):
  /tmp/public_oof/gkanamoto_s6e5-tabm/{oof_pred.csv, submission.csv}
  /tmp/public_oof/donmarch14_s6e5-tabm/{oof_preds.csv, submission.csv}
  /tmp/public_oof/stpeteishii_f1-pit-stops-mlp-baseline-w-oof/{oof_mlp_tpu.csv, submission_mlp_tpu.csv}
  /tmp/public_oof/mhamza0810_ps-s6e5-multi-seed-xgb-with-meta-feature/{weighted_oof_xgb_20_seed.csv, XGB_weighted_submission_20_seeds.csv}
  submissions/oof_catboost_baseline_v1.csv, submission_catboost_baseline_v1.csv

Procedure:
  1. Re-fit greedy logit-space weights on train OOFs against true labels.
  2. Apply identical weights to test submission CSVs to produce a final submission.
  3. Save both OOF (for later stacking with the user's stack) and submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID_COL = "id"
TARGET = "PitNextLap"

MEMBERS = [
    # name, oof_path, oof_pred_col, sub_path, sub_pred_col, oof_has_id
    ("tabm_gka",            "/tmp/public_oof/gkanamoto_s6e5-tabm/oof_pred.csv",                                                    "oof_pred",   "/tmp/public_oof/gkanamoto_s6e5-tabm/submission.csv",                                                  "PitNextLap", True),
    ("tabm_don",            "/tmp/public_oof/donmarch14_s6e5-tabm/oof_preds.csv",                                                  "PitNextLap", "/tmp/public_oof/donmarch14_s6e5-tabm/submission.csv",                                                 "PitNextLap", True),
    ("mlp_stp",             "/tmp/public_oof/stpeteishii_f1-pit-stops-mlp-baseline-w-oof/oof_mlp_tpu.csv",                         "oof_pred",   "/tmp/public_oof/stpeteishii_f1-pit-stops-mlp-baseline-w-oof/submission_mlp_tpu.csv",                  "PitNextLap", True),
    ("xgb_hamza_raw",       "/tmp/public_oof/mhamza0810_ps-s6e5-multi-seed-xgb-with-meta-feature/oof_xgb_20_seed.csv",             "oof_xgb_20_seed",     "/tmp/public_oof/mhamza0810_ps-s6e5-multi-seed-xgb-with-meta-feature/XGB_submission_20_seeds.csv",           "PitNextLap", False),
    ("xgb_hamza_weighted",  "/tmp/public_oof/mhamza0810_ps-s6e5-multi-seed-xgb-with-meta-feature/weighted_oof_xgb_20_seed.csv",    "weighted_oof_20_seeds", "/tmp/public_oof/mhamza0810_ps-s6e5-multi-seed-xgb-with-meta-feature/XGB_weighted_submission_20_seeds.csv", "PitNextLap", False),
    ("my_catboost",         "submissions/oof_catboost_baseline_v1.csv",                                                            "pred",       "submissions/submission_catboost_baseline_v1.csv",                                                     "PitNextLap", True),
    ("cb_trknbr",           "submissions/oof_catboost_trknbr_v1.csv",                                                              "pred",       "submissions/submission_catboost_trknbr_v1.csv",                                                       "PitNextLap", True),
    ("masaya_cat",          "/tmp/public_oof/masayakawamata_s6e5-cat-with-fe/oof_cat_0.9527003594478181.csv",                       "PitNextLap", "/tmp/public_oof/masayakawamata_s6e5-cat-with-fe/test_cat_0.9527003594478181.csv",                       "PitNextLap", True),
    ("masaya_lgbmgoss",     "/tmp/public_oof/masayakawamata_s6e5-lgbmgoss-with-fe/oof_lgbmgoss_0.9518478303570366.csv",             "PitNextLap", "/tmp/public_oof/masayakawamata_s6e5-lgbmgoss-with-fe/test_lgbmgoss_0.9518478303570366.csv",             "PitNextLap", True),
    ("masaya_lgbmxt",       "/tmp/public_oof/masayakawamata_s6e5-lgbmxt-with-fe/oof_lgbmxt_0.9521400678382188.csv",                 "PitNextLap", "/tmp/public_oof/masayakawamata_s6e5-lgbmxt-with-fe/test_lgbmxt_0.9521400678382188.csv",                 "PitNextLap", True),
    ("masaya_xgb",          "/tmp/public_oof/masayakawamata_s6e5-xgb-with-fe/oof_xgb_0.9520300874188106.csv",                       "PitNextLap", "/tmp/public_oof/masayakawamata_s6e5-xgb-with-fe/test_xgb_0.9520300874188106.csv",                       "PitNextLap", True),
    ("yekenot_realmlp",     "/tmp/public_oof/yekenot_ps-s6-e5-realmlp-pytabkit/oof_preds.csv",                                      "PitNextLap", "/tmp/public_oof/yekenot_ps-s6-e5-realmlp-pytabkit/submission.csv",                                      "PitNextLap", True),
    ("arunklenin_fe",       "/tmp/public_oof/arunklenin_fe-ensemble/oof_predictions.csv",                                          "PitNextLap", "/tmp/public_oof/arunklenin_fe-ensemble/submission.csv",                                                 "PitNextLap", True),
    ("pseudo_v2",           "submissions/oof_catboost_pseudo_v2_from_strong_blend.csv",                                            "pred",       "submissions/submission_catboost_pseudo_v2_from_strong_blend.csv",                                       "PitNextLap", True),
    ("tabm_gka_v2",         "/tmp/public_oof/gkanamoto_s65e-tabm-ver2-0/oof_pred.csv",                                              "oof_pred",   "/tmp/public_oof/gkanamoto_s65e-tabm-ver2-0/submission.csv",                                             "PitNextLap", True),
    ("simarbir_realmlp",    "/tmp/public_oof/simarbirsinghsandhu_realmlp-encoding-fe/oof_preds.csv",                                "PitNextLap", "/tmp/public_oof/simarbirsinghsandhu_realmlp-encoding-fe/submission.csv",                                "PitNextLap", True),
    ("masaya_realmlp_blend","/tmp/public_oof/masayakawamata_s6e5-realmlp-with-fe/oof_strong_realmlp_goodparts_blend_equal_silu_plr_512x256x128_ens12__silu_plr_768x384x192_ens8_0.951215.csv", "PitNextLap", "/tmp/public_oof/masayakawamata_s6e5-realmlp-with-fe/test_strong_realmlp_goodparts_blend_equal_silu_plr_512x256x128_ens12__silu_plr_768x384x192_ens8_0.951215.csv", "PitNextLap", True),
]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--suffix", default="public_blend_v1")
    parser.add_argument("--max-steps", default=8, type=int)
    parser.add_argument("--alpha-grid", default=50, type=int)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COL].to_numpy()
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    y = train[TARGET].astype(int).to_numpy()
    train_ids = train[ID_COL].to_numpy()
    train_id_set = set(train_ids)

    test_id_set = set(test_ids)

    def load_oof(path: str, pred_col: str, has_id: bool) -> np.ndarray:
        df = pd.read_csv(path)
        if has_id:
            df = df[df[ID_COL].isin(train_id_set)].drop_duplicates(ID_COL).set_index(ID_COL).reindex(train_ids)
            arr = df[pred_col].to_numpy()
        else:
            arr = df[pred_col].to_numpy()
            assert len(arr) == len(train_ids), f"positional length {len(arr)} != {len(train_ids)} for {path}"
        return arr

    def load_sub(path: str, pred_col: str) -> np.ndarray:
        df = pd.read_csv(path)
        # Most subs have id + PitNextLap. Align to test_ids order.
        id_col = ID_COL if ID_COL in df.columns else df.columns[0]
        df = df.drop_duplicates(id_col).set_index(id_col)
        arr = df[pred_col].reindex(test_ids).to_numpy()
        assert not np.isnan(arr).any(), f"NaN in test sub from {path}"
        return arr

    oofs: dict[str, np.ndarray] = {}
    subs: dict[str, np.ndarray] = {}
    for name, oof_path, oof_col, sub_path, sub_col, has_id in MEMBERS:
        oofs[name] = load_oof(oof_path, oof_col, has_id)
        subs[name] = load_sub(sub_path, sub_col)
    print(f"loaded {len(oofs)} members")

    aucs = {n: float(roc_auc_score(y, p)) for n, p in oofs.items()}
    for n, a in aucs.items():
        print(f"  {n:25s}  OOF AUC: {a:.6f}")

    anchor = max(aucs, key=aucs.get)
    print(f"\nanchor: {anchor}  AUC={aucs[anchor]:.6f}")

    weights = {n: 0.0 for n in oofs}
    weights[anchor] = 1.0
    current_oof_logit = logit(oofs[anchor]).copy()
    current_sub_logit = logit(subs[anchor]).copy()
    best_auc = aucs[anchor]
    history = [{"step": 0, "added": anchor, "alpha": 1.0, "auc": best_auc, "weights": dict(weights)}]

    for step in range(args.max_steps):
        best_cand = None
        for n in oofs:
            if weights[n] >= 0.65:
                continue
            for a in np.linspace(0.005, 0.5, args.alpha_grid):
                au = float(roc_auc_score(y, sig((1 - a) * current_oof_logit + a * logit(oofs[n]))))
                if au > best_auc + 1e-7 and (best_cand is None or au > best_cand[0]):
                    best_cand = (au, a, n)
        if best_cand is None:
            break
        au, a, n = best_cand
        current_oof_logit = (1 - a) * current_oof_logit + a * logit(oofs[n])
        current_sub_logit = (1 - a) * current_sub_logit + a * logit(subs[n])
        for k in weights:
            weights[k] *= (1 - a)
        weights[n] += a
        best_auc = au
        nz = {k: round(v, 4) for k, v in weights.items() if v > 0.005}
        history.append({"step": step + 1, "added": n, "alpha": float(a), "auc": best_auc, "weights": dict(weights)})
        print(f"  step {step+1}: + {n} alpha={a:.4f}  AUC={au:.6f}  weights={nz}")

    final_oof = sig(current_oof_logit)
    final_test = np.clip(sig(current_sub_logit), 0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    pd.DataFrame({ID_COL: train_ids, TARGET: y, "pred": final_oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub[pred_col] = final_test
    sub.to_csv(sub_path, index=False)

    summary = {
        "individual_oof_aucs": aucs,
        "final_blend_oof_auc": best_auc,
        "anchor": anchor,
        "final_weights": {k: float(v) for k, v in weights.items() if v > 1e-4},
        "history": [{**h, "weights": {k: float(v) for k, v in h["weights"].items() if v > 1e-4}} for h in history],
        "oof_path": str(oof_path),
        "submission_path": str(sub_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinal OOF AUC: {best_auc:.6f}")
    print(f"Wrote: {sub_path}")
    print(f"Wrote: {oof_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
