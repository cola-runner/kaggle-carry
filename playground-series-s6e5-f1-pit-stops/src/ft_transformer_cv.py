"""FT-Transformer (Feature Tokenizer + Transformer) for tabular pit-stop prediction.

Each numerical feature gets its own learned token (scale * 1d-token + bias), each
categorical feature gets a standard embedding. A learned CLS token is prepended.
Stack of transformer encoder layers attend over the resulting token sequence,
and the CLS output projects to the logit.

Compared to a plain MLP this gives the model an explicit *per-feature* attention
mechanism, which on tabular tasks usually closes the AUC gap to GBDTs and yields
a much less correlated prediction.
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


def encode_num(train: pd.DataFrame, test: pd.DataFrame, external: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Standardize numerical features using all known rows (train+test+external)."""
    pool = pd.concat([train[NUM_COLS], test[NUM_COLS], external[NUM_COLS]], ignore_index=True)
    medians = pool.median().to_dict()
    pool = pool.fillna(medians)
    mean = pool.mean().to_dict()
    std = (pool.std() + 1e-6).to_dict()

    def arr(df):
        z = df[NUM_COLS].copy()
        for c in NUM_COLS:
            z[c] = (z[c].fillna(medians[c]) - mean[c]) / std[c]
        return z.to_numpy(dtype=np.float32)

    return arr(train), arr(test), arr(external), {"mean": mean, "std": std, "medians": medians}


class NumericalFeatureTokenizer(nn.Module):
    """For each of n numerical features produces a token of dim `d`.

    Token = `w_i * x_i + b_i` (per-feature affine projection to a d-dim vector).
    """
    def __init__(self, n_features: int, d: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_features, d))
        self.b = nn.Parameter(torch.zeros(n_features, d))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_features); output: (B, n_features, d)
        return x_num.unsqueeze(-1) * self.w + self.b


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities: list[int], d: int) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c in cardinalities])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, n_features); output: (B, n_features, d)
        return torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class FTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ff_mult, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln1(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cardinalities: list[int],
        d: int = 96,
        n_blocks: int = 3,
        n_heads: int = 8,
        dropout: float = 0.10,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.num_tok = NumericalFeatureTokenizer(n_num, d)
        self.cat_tok = CategoricalFeatureTokenizer(cardinalities, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([FTBlock(d, n_heads, dropout, ff_mult) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        num = self.num_tok(x_num)
        cat = self.cat_tok(x_cat)
        cls = self.cls.expand(x_num.shape[0], -1, -1)
        h = torch.cat([cls, num, cat], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, 0]).squeeze(-1)


def train_seed(
    seed: int,
    cat_tr: np.ndarray, num_tr: np.ndarray, y_tr: np.ndarray,
    cat_va: np.ndarray, num_va: np.ndarray, y_va: np.ndarray,
    cat_te: np.ndarray, num_te: np.ndarray,
    cardinalities: list[int],
    device: torch.device,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    dropout: float, patience: int,
    d: int, n_blocks: int, n_heads: int, ff_mult: int,
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

    model = FTTransformer(
        n_num=num_tr.shape[1], cardinalities=cardinalities,
        d=d, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout, ff_mult=ff_mult,
    ).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    parser.add_argument("--seeds", default="11,23", help="Comma-separated NN init seeds to average.")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.10, type=float)
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument("--d", default=96, type=int)
    parser.add_argument("--n-blocks", default=3, type=int)
    parser.add_argument("--n-heads", default=8, type=int)
    parser.add_argument("--ff-mult", default=2, type=int)
    parser.add_argument("--use-external", action="store_true",
                        help="Concat external rows into each training fold.")
    parser.add_argument("--suffix", default="v1")
    parser.add_argument("--device", default="auto")
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
    external_raw = pd.read_csv(args.external_csv) if args.external_csv.exists() else None
    print(f"train: {len(train_raw)}  test: {len(test_raw)}  external: {0 if external_raw is None else len(external_raw)}", flush=True)

    train = add_simple_features(train_raw)
    test = add_simple_features(test_raw)
    external = add_simple_features(external_raw) if external_raw is not None else train.iloc[:0].copy()

    y = train[TARGET].astype(int).to_numpy()
    y_ext = external[TARGET].astype(int).to_numpy() if len(external) else np.zeros(0, dtype=int)

    vocabs = build_vocabs(train, test, external)
    cardinalities = [len(vocabs[c]) + 1 for c in CAT_COLS]
    cat_train = encode_cat(train, vocabs)
    cat_test = encode_cat(test, vocabs)
    cat_external = encode_cat(external, vocabs)
    num_train, num_test, num_external, _ = encode_num(train, test, external)
    print(f"Cardinalities: {dict(zip(CAT_COLS, cardinalities))}", flush=True)
    print(f"Numeric features: {len(NUM_COLS)}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.cv_seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_summaries = []

    for fold, (tri, vai) in enumerate(skf.split(cat_train, y), start=1):
        t0 = time.time()
        print(f"\n=========== fold {fold} ===========", flush=True)
        cat_tr_fold = cat_train[tri]
        num_tr_fold = num_train[tri]
        y_tr_fold = y[tri]
        if args.use_external and len(external):
            cat_tr_fold = np.concatenate([cat_tr_fold, cat_external], axis=0)
            num_tr_fold = np.concatenate([num_tr_fold, num_external], axis=0)
            y_tr_fold = np.concatenate([y_tr_fold, y_ext])

        seed_valid = []
        seed_test = []
        per_seed = []
        for sd in seeds:
            print(f" -- fold {fold}  seed {sd}", flush=True)
            valid_pred, fold_test_pred, best_auc = train_seed(
                seed=sd + 1000 * fold,
                cat_tr=cat_tr_fold, num_tr=num_tr_fold, y_tr=y_tr_fold,
                cat_va=cat_train[vai], num_va=num_train[vai], y_va=y[vai],
                cat_te=cat_test, num_te=num_test,
                cardinalities=cardinalities, device=device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, dropout=args.dropout, patience=args.patience,
                d=args.d, n_blocks=args.n_blocks, n_heads=args.n_heads, ff_mult=args.ff_mult,
            )
            seed_valid.append(valid_pred)
            seed_test.append(fold_test_pred)
            per_seed.append(best_auc)

        avg_valid = np.mean(seed_valid, axis=0)
        avg_test = np.mean(seed_test, axis=0)
        oof[vai] = avg_valid
        test_pred += avg_test / args.n_splits
        fold_auc = float(roc_auc_score(y[vai], avg_valid))
        print(f"[fold {fold}] avg AUC = {fold_auc:.6f}  (per-seed: {per_seed})  t={time.time()-t0:.1f}s", flush=True)
        fold_summaries.append({"fold": fold, "avg_auc": fold_auc, "per_seed": per_seed, "seconds": time.time() - t0})

    oof_auc = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC: {oof_auc:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / f"oof_ft_transformer_{args.suffix}.csv"
    sub_path = args.output_dir / f"submission_ft_transformer_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_ft_transformer_{args.suffix}.json"
    pd.DataFrame({ID_COL: train_raw[ID_COL].to_numpy(), TARGET: y, "pred": oof}).to_csv(oof_path, index=False)
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COL][0]
    sub[pred_col] = np.clip(test_pred, 0.0, 1.0)
    sub.to_csv(sub_path, index=False)
    summary = {
        "device": str(device), "n_splits": args.n_splits, "cv_seed": args.cv_seed, "seeds": seeds,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout, "d": args.d, "n_blocks": args.n_blocks,
        "n_heads": args.n_heads, "ff_mult": args.ff_mult, "use_external": args.use_external,
        "oof_auc": oof_auc, "fold_summaries": fold_summaries,
        "oof_path": str(oof_path), "submission_path": str(sub_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"oof_auc": oof_auc, "outputs": [str(oof_path), str(sub_path)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
