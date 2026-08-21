"""PyTorch MLP with categorical embeddings as a decorrelated stack member.

Uses the same simple-feature recipe as `simple_external_cv.py` and the same
StratifiedKFold(seed=2026, n_splits=5) so the resulting OOF can be blended
directly with the existing CatBoost / LightGBM OOF.

Outputs:
  submissions/oof_nn_mlp_<suffix>.csv         (id, PitNextLap, pred)
  submissions/submission_nn_mlp_<suffix>.csv  (id, PitNextLap)
  submissions/summary_nn_mlp_<suffix>.json
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
NUM_COLS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change",
    "TotalLaps_est", "LapsRemaining_est", "TyreLife_frac_race", "TyreLife_frac_lap",
    "TyreLife_x_Progress", "Degradation_per_TyreLife", "Delta_per_TyreLife",
    "Abs_Position_Change",
]


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
    return out.replace([np.inf, -np.inf], np.nan)


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame) -> tuple[
    dict[str, dict[str, int]], dict[str, int], np.ndarray, np.ndarray
]:
    """Map categorical strings to integer codes shared across train and test.

    The 0 code is reserved for unknown / missing.
    """
    vocabs: dict[str, dict[str, int]] = {}
    cardinalities: dict[str, int] = {}
    cat_train = np.zeros((len(train), len(CAT_COLS)), dtype=np.int64)
    cat_test = np.zeros((len(test), len(CAT_COLS)), dtype=np.int64)

    for j, col in enumerate(CAT_COLS):
        values = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
        uniq = pd.Index(values.unique())
        vocab = {v: i + 1 for i, v in enumerate(uniq)}
        vocabs[col] = vocab
        cardinalities[col] = len(vocab) + 1  # +1 for the unknown bucket
        cat_train[:, j] = train[col].astype(str).map(vocab).fillna(0).astype(np.int64).to_numpy()
        cat_test[:, j] = test[col].astype(str).map(vocab).fillna(0).astype(np.int64).to_numpy()

    return vocabs, cardinalities, cat_train, cat_test


def standardize_numeric(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    stats = {}
    num_train = np.zeros((len(train), len(NUM_COLS)), dtype=np.float32)
    num_test = np.zeros((len(test), len(NUM_COLS)), dtype=np.float32)
    for j, col in enumerate(NUM_COLS):
        tr_vals = train[col].astype(float)
        te_vals = test[col].astype(float)
        median = float(tr_vals.median())
        # Fill na with median, then standardize using train mean/std.
        tr_filled = tr_vals.fillna(median).to_numpy(dtype=np.float64)
        te_filled = te_vals.fillna(median).to_numpy(dtype=np.float64)
        mean = float(tr_filled.mean())
        std = float(tr_filled.std() + 1e-6)
        num_train[:, j] = ((tr_filled - mean) / std).astype(np.float32)
        num_test[:, j] = ((te_filled - mean) / std).astype(np.float32)
        stats[col] = {"median": median, "mean": mean, "std": std}
    return num_train, num_test, stats


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
        in_dim = emb_total + num_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.embs[i](x_cat[:, i]) for i in range(x_cat.shape[1])], dim=1)
        num = self.bn_num(x_num)
        z = torch.cat([emb, num], dim=1)
        return self.mlp(z).squeeze(-1)


def train_fold(
    fold: int,
    tri: np.ndarray,
    vai: np.ndarray,
    cat_all: np.ndarray,
    num_all: np.ndarray,
    y_all: np.ndarray,
    cat_test: np.ndarray,
    num_test: np.ndarray,
    cardinalities: dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed + fold)

    cat_tr = torch.from_numpy(cat_all[tri]).to(device)
    num_tr = torch.from_numpy(num_all[tri]).to(device)
    y_tr = torch.from_numpy(y_all[tri].astype(np.float32)).to(device)

    cat_va = torch.from_numpy(cat_all[vai]).to(device)
    num_va = torch.from_numpy(num_all[vai]).to(device)
    y_va = y_all[vai]

    cat_te = torch.from_numpy(cat_test).to(device)
    num_te = torch.from_numpy(num_test).to(device)

    model = TabMLP(cardinalities, num_dim=num_all.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(tri)
    rng = np.random.default_rng(seed + fold)
    best_auc = -np.inf
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xc = cat_tr[idx]
            xn = num_tr[idx]
            yb = y_tr[idx]
            logits = model(xc, xn)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        model.eval()
        with torch.no_grad():
            valid_logits = []
            for start in range(0, len(vai), batch_size):
                vl = model(cat_va[start : start + batch_size], num_va[start : start + batch_size])
                valid_logits.append(vl)
            valid_pred = torch.sigmoid(torch.cat(valid_logits)).cpu().numpy()
        auc = float(roc_auc_score(y_va, valid_pred))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_auc": auc})
        improved = auc > best_auc + 1e-6
        if improved:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"[fold {fold}] epoch {epoch:>2}  train_loss={np.mean(losses):.4f}  valid_auc={auc:.6f}"
              f"  best={best_auc:.6f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if bad >= patience:
            print(f"[fold {fold}] early stop after epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_logits = []
        for start in range(0, len(vai), batch_size):
            vl = model(cat_va[start : start + batch_size], num_va[start : start + batch_size])
            valid_logits.append(vl)
        valid_pred = torch.sigmoid(torch.cat(valid_logits)).cpu().numpy()

        test_logits = []
        for start in range(0, len(cat_te), batch_size):
            tl = model(cat_te[start : start + batch_size], num_te[start : start + batch_size])
            test_logits.append(tl)
        test_pred = torch.sigmoid(torch.cat(test_logits)).cpu().numpy()

    return valid_pred, test_pred, {"best_valid_auc": best_auc, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP NN member with cat embeddings.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int,
                        help="Match the StratifiedKFold seed used by the GBDT pipeline.")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.30, type=float)
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    extract_zip_if_needed(args.data_dir)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    y = train[TARGET].astype(int).to_numpy()

    vocabs, cardinalities, cat_train, cat_test = encode_categoricals(train, test)
    num_train, num_test, stats = standardize_numeric(train, test)
    print(f"Cardinalities: {cardinalities}", flush=True)
    print(f"Numeric features: {len(NUM_COLS)}", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_summaries = []

    for fold, (tri, vai) in enumerate(skf.split(cat_train, y), start=1):
        t0 = time.time()
        print(f"\n========== fold {fold} ==========", flush=True)
        valid_pred, fold_test_pred, info = train_fold(
            fold=fold,
            tri=tri,
            vai=vai,
            cat_all=cat_train,
            num_all=num_train,
            y_all=y,
            cat_test=cat_test,
            num_test=num_test,
            cardinalities=cardinalities,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            seed=args.seed,
        )
        oof[vai] = valid_pred
        test_pred += fold_test_pred / args.n_splits
        info["fold_seconds"] = time.time() - t0
        info["fold"] = fold
        fold_summaries.append(info)
        print(f"[fold {fold}] best valid AUC: {info['best_valid_auc']:.6f}  t={info['fold_seconds']:.1f}s",
              flush=True)

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
        "device": str(device),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "cardinalities": cardinalities,
        "numeric_features": NUM_COLS,
        "oof_auc": oof_auc,
        "fold_summaries": [
            {k: v for k, v in s.items() if k != "history"} | {"final_epoch": s["history"][-1]["epoch"]}
            for s in fold_summaries
        ],
        "oof_path": str(oof_path),
        "submission_path": str(sub_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": oof_auc, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
