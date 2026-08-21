"""Stronger NN member: + OOF-safe target encoding + external F1 data + multi-seed.

Goals over `mlp_nn_cv.py`:
  - Add target-encoded versions of high-cardinality categoricals as numeric
    features (this is what gives CatBoost its tabular edge).
  - Concatenate the external F1 strategy dataset into each training fold
    (the validation slice stays purely competition train, so OOF stays valid).
  - Average over multiple seeds to reduce single-network variance.

Outputs:
  submissions/oof_nn_mlp_v2_<suffix>.csv         (id, PitNextLap, pred)
  submissions/submission_nn_mlp_v2_<suffix>.csv
  submissions/summary_nn_mlp_v2_<suffix>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn


ID_COL = "id"
TARGET = "PitNextLap"

CAT_COLS = ["Driver", "Compound", "Race", "Race_Compound", "Race_Year", "Compound_Stint"]
NUM_COLS_BASE = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change",
    "TotalLaps_est", "LapsRemaining_est", "TyreLife_frac_race", "TyreLife_frac_lap",
    "TyreLife_x_Progress", "Degradation_per_TyreLife", "Delta_per_TyreLife",
    "Abs_Position_Change",
]
TE_KEYS = ["Driver", "Race", "Race_Year", "Race_Compound", "Compound_Stint",
           "Driver_Race", "Driver_Compound", "Driver_Year"]


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {p.name for p in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return
    for zip_path in sorted(data_dir.glob("*.zip")):
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(columns={"LapTime (s)": "LapTime_s"})
    total_laps = out["LapNumber"] / out["RaceProgress"].replace(0, np.nan)
    out["TotalLaps_est"] = total_laps.clip(1, 120)
    out["LapsRemaining_est"] = out["TotalLaps_est"] - out["LapNumber"]
    out["TyreLife_frac_race"] = out["TyreLife"] / out["TotalLaps_est"].replace(0, np.nan)
    out["TyreLife_frac_lap"] = out["TyreLife"] / out["LapNumber"].replace(0, np.nan)
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1)
    out["Delta_per_TyreLife"] = out["LapTime_Delta"] / (out["TyreLife"] + 1)
    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["Compound_Stint"] = out["Compound"].astype(str) + "_" + out["Stint"].astype(str)
    out["Driver_Race"] = out["Driver"].astype(str) + "_" + out["Race"].astype(str)
    out["Driver_Compound"] = out["Driver"].astype(str) + "_" + out["Compound"].astype(str)
    out["Driver_Year"] = out["Driver"].astype(str) + "_" + out["Year"].astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def build_vocabs(*frames: pd.DataFrame) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for col in CAT_COLS:
        values = pd.concat([f[col].astype(str) for f in frames], ignore_index=True)
        uniq = pd.Index(values.unique())
        vocabs[col] = {v: i + 1 for i, v in enumerate(uniq)}
    return vocabs


def encode_cat(frame: pd.DataFrame, vocabs: dict[str, dict[str, int]]) -> np.ndarray:
    out = np.zeros((len(frame), len(CAT_COLS)), dtype=np.int64)
    for j, col in enumerate(CAT_COLS):
        out[:, j] = frame[col].astype(str).map(vocabs[col]).fillna(0).astype(np.int64).to_numpy()
    return out


def target_encode_columns(
    train_keys: pd.DataFrame,
    y_train: np.ndarray,
    valid_keys: pd.DataFrame,
    test_keys: pd.DataFrame,
    external_keys: pd.DataFrame,
    y_external: np.ndarray,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed target encoding using train-fold + external; safe for validation/test."""
    n_train = len(train_keys)
    n_valid = len(valid_keys)
    n_test = len(test_keys)
    cols = TE_KEYS
    out_train = np.zeros((n_train, len(cols)), dtype=np.float32)
    out_valid = np.zeros((n_valid, len(cols)), dtype=np.float32)
    out_test = np.zeros((n_test, len(cols)), dtype=np.float32)

    src_keys = pd.concat([train_keys, external_keys], ignore_index=True)
    src_y = np.concatenate([y_train, y_external])
    global_mean = float(src_y.mean())

    src_df = src_keys.assign(_y=src_y)
    for j, col in enumerate(cols):
        grp = src_df.groupby(col)["_y"].agg(["mean", "count"])
        encoded = (grp["mean"] * grp["count"] + global_mean * smooth) / (grp["count"] + smooth)
        mp = encoded.to_dict()
        out_train[:, j] = train_keys[col].map(mp).fillna(global_mean).astype(np.float32).to_numpy()
        out_valid[:, j] = valid_keys[col].map(mp).fillna(global_mean).astype(np.float32).to_numpy()
        out_test[:, j] = test_keys[col].map(mp).fillna(global_mean).astype(np.float32).to_numpy()
    return out_train, out_valid, out_test


def standardize_block(
    src: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    mean = src.mean(axis=0, keepdims=True)
    std = src.std(axis=0, keepdims=True) + 1e-6
    src_z = ((src - mean) / std).astype(np.float32)
    return src_z, [((o - mean) / std).astype(np.float32) for o in others]


def emb_dim(card: int) -> int:
    return int(min(48, max(4, round(card ** 0.5) + 1)))


class TabMLP(nn.Module):
    def __init__(self, cardinalities: dict[str, int], num_dim: int, hidden: tuple[int, ...] = (512, 256, 128), dropout: float = 0.30) -> None:
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(cardinalities[c], emb_dim(cardinalities[c])) for c in CAT_COLS]
        )
        emb_total = sum(emb_dim(cardinalities[c]) for c in CAT_COLS)
        self.bn_num = nn.BatchNorm1d(num_dim)
        layers: list[nn.Module] = []
        prev = emb_total + num_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.embs[i](x_cat[:, i]) for i in range(x_cat.shape[1])], dim=1)
        num = self.bn_num(x_num)
        return self.mlp(torch.cat([emb, num], dim=1)).squeeze(-1)


def train_seed(
    seed: int,
    cat_tr: np.ndarray,
    num_tr: np.ndarray,
    y_tr: np.ndarray,
    cat_va: np.ndarray,
    num_va: np.ndarray,
    y_va: np.ndarray,
    cat_te: np.ndarray,
    num_te: np.ndarray,
    cardinalities: dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cat_tr_t = torch.from_numpy(cat_tr).to(device)
    num_tr_t = torch.from_numpy(num_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    cat_va_t = torch.from_numpy(cat_va).to(device)
    num_va_t = torch.from_numpy(num_va).to(device)
    cat_te_t = torch.from_numpy(cat_te).to(device)
    num_te_t = torch.from_numpy(num_te).to(device)

    model = TabMLP(cardinalities, num_dim=num_tr.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    bad = 0
    n = len(cat_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(cat_tr_t[idx], num_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, len(cat_va), batch_size):
                preds.append(model(cat_va_t[start:start + batch_size], num_va_t[start:start + batch_size]))
            valid_pred = torch.sigmoid(torch.cat(preds)).cpu().numpy()
        auc = float(roc_auc_score(y_va, valid_pred))
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"   seed={seed}  ep {epoch:>2}  loss={np.mean(losses):.4f}  auc={auc:.6f}  best={best_auc:.6f}", flush=True)
        if bad >= patience:
            print(f"   seed={seed}  early stop @ ep {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for start in range(0, len(cat_va), batch_size):
            preds.append(model(cat_va_t[start:start + batch_size], num_va_t[start:start + batch_size]))
        valid_pred = torch.sigmoid(torch.cat(preds)).cpu().numpy()
        preds_t = []
        for start in range(0, len(cat_te), batch_size):
            preds_t.append(model(cat_te_t[start:start + batch_size], num_te_t[start:start + batch_size]))
        test_pred = torch.sigmoid(torch.cat(preds_t)).cpu().numpy()
    return valid_pred, test_pred, best_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--cv-seed", default=2026, type=int)
    parser.add_argument("--seeds", default="11,23,47", help="Comma-separated NN init seeds to average.")
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.30, type=float)
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument("--te-smooth", default=50.0, type=float)
    parser.add_argument("--suffix", default="v2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("mps") if (args.device == "auto" and torch.backends.mps.is_available()) else torch.device(args.device if args.device != "auto" else "cpu")
    print(f"Using device: {device}", flush=True)

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    external_raw = pd.read_csv(args.external_csv)
    print(f"train: {len(train_raw)}  test: {len(test_raw)}  external: {len(external_raw)}", flush=True)

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    external = add_simple_features(external_raw)

    y = train[TARGET].astype(int).to_numpy()
    y_ext = external[TARGET].astype(int).to_numpy()

    vocabs = build_vocabs(train, test, external)
    cardinalities = {c: len(v) + 1 for c, v in vocabs.items()}
    cat_train = encode_cat(train, vocabs)
    cat_test = encode_cat(test, vocabs)
    cat_external = encode_cat(external, vocabs)
    print(f"Cardinalities: {cardinalities}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"NN seeds: {seeds}", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.cv_seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_summaries = []
    n_runs = len(seeds) * args.n_splits
    runs_done = 0

    for fold, (tri, vai) in enumerate(skf.split(cat_train, y), start=1):
        t0 = time.time()
        print(f"\n=========== fold {fold} ===========", flush=True)

        train_keys = train.iloc[tri][TE_KEYS].astype(str).reset_index(drop=True)
        valid_keys = train.iloc[vai][TE_KEYS].astype(str).reset_index(drop=True)
        test_keys = test[TE_KEYS].astype(str).reset_index(drop=True)
        external_keys = external[TE_KEYS].astype(str).reset_index(drop=True)

        te_tr, te_va, te_te = target_encode_columns(
            train_keys, y[tri], valid_keys, test_keys, external_keys, y_ext, smooth=args.te_smooth,
        )
        te_ext = np.zeros((len(external), len(TE_KEYS)), dtype=np.float32)
        for j, col in enumerate(TE_KEYS):
            src = pd.concat([train_keys[col], external_keys[col]], ignore_index=True)
            src_y = np.concatenate([y[tri], y_ext])
            grp = pd.DataFrame({col: src, "_y": src_y}).groupby(col)["_y"].agg(["mean", "count"])
            global_mean = float(src_y.mean())
            encoded = (grp["mean"] * grp["count"] + global_mean * args.te_smooth) / (grp["count"] + args.te_smooth)
            te_ext[:, j] = external_keys[col].map(encoded.to_dict()).fillna(global_mean).astype(np.float32).to_numpy()

        num_tr_base = train.iloc[tri][NUM_COLS_BASE].astype(float).fillna(0).to_numpy(dtype=np.float32)
        num_va_base = train.iloc[vai][NUM_COLS_BASE].astype(float).fillna(0).to_numpy(dtype=np.float32)
        num_te_base = test[NUM_COLS_BASE].astype(float).fillna(0).to_numpy(dtype=np.float32)
        num_ext_base = external[NUM_COLS_BASE].astype(float).fillna(0).to_numpy(dtype=np.float32)

        all_num_tr = np.concatenate([num_tr_base, te_tr], axis=1)
        all_num_va = np.concatenate([num_va_base, te_va], axis=1)
        all_num_te = np.concatenate([num_te_base, te_te], axis=1)
        all_num_ext = np.concatenate([num_ext_base, te_ext], axis=1)

        cat_tr_fold = np.concatenate([cat_train[tri], cat_external], axis=0)
        num_tr_fold = np.concatenate([all_num_tr, all_num_ext], axis=0)
        y_tr_fold = np.concatenate([y[tri], y_ext])

        # standardize numeric using the training fold (train+external) stats
        num_tr_fold_z, (num_va_z, num_te_z) = standardize_block(num_tr_fold, all_num_va, all_num_te)

        fold_seed_aucs = []
        seed_valid = []
        seed_test = []
        for sd in seeds:
            runs_done += 1
            print(f" -- run {runs_done}/{n_runs}: fold {fold}  seed {sd}", flush=True)
            valid_pred, fold_test_pred, best_auc = train_seed(
                seed=sd + 1000 * fold,
                cat_tr=cat_tr_fold,
                num_tr=num_tr_fold_z,
                y_tr=y_tr_fold,
                cat_va=cat_train[vai],
                num_va=num_va_z,
                y_va=y[vai],
                cat_te=cat_test,
                num_te=num_te_z,
                cardinalities=cardinalities,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                patience=args.patience,
            )
            fold_seed_aucs.append(best_auc)
            seed_valid.append(valid_pred)
            seed_test.append(fold_test_pred)

        avg_valid = np.mean(seed_valid, axis=0)
        avg_test = np.mean(seed_test, axis=0)
        oof[vai] = avg_valid
        test_pred += avg_test / args.n_splits
        fold_auc = float(roc_auc_score(y[vai], avg_valid))
        print(f"[fold {fold}] avg AUC across {len(seeds)} seeds = {fold_auc:.6f}  (per-seed: {fold_seed_aucs})  t={time.time()-t0:.1f}s", flush=True)
        fold_summaries.append({"fold": fold, "avg_auc": fold_auc, "per_seed_auc": fold_seed_aucs, "seconds": time.time() - t0})

    oof_auc = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {oof_auc:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_nn_mlp_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_nn_mlp_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_nn_mlp_{args.suffix}.json"

    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub[pred_col] = np.clip(test_pred, 0.0, 1.0)
    sub.to_csv(sub_path, index=False)

    summary = {
        "device": str(device), "n_splits": args.n_splits, "cv_seed": args.cv_seed, "seeds": seeds,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout, "te_smooth": args.te_smooth,
        "cardinalities": cardinalities, "oof_auc": oof_auc,
        "fold_summaries": fold_summaries,
        "oof_path": str(oof_path), "submission_path": str(sub_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": oof_auc, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
