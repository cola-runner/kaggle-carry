from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ID_COL = "id"
TARGET = "PitNextLap"
LAPTIME_RAW = "LapTime (s)"
LAPTIME = "LapTime_s"
PRED_COLS = ["pred", "blend", "catboost_realmlp_external", TARGET]

NUMERIC_COLS = [
    "LapNumber",
    "PitStop",
    "Stint",
    "TyreLife",
    "Position",
    LAPTIME,
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]


@dataclass(frozen=True)
class KNNConfig:
    name: str
    bucket_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    cat_cols: tuple[str, ...] = ("Compound",)
    weights: dict[str, float] | None = None


CONFIGS = [
    KNNConfig(
        name="yr_race_knn",
        bucket_cols=("Year", "Race"),
        numeric_cols=tuple(NUMERIC_COLS),
        weights={
            "LapNumber": 4.0,
            "RaceProgress": 3.0,
            "TyreLife": 2.0,
            "Position": 2.0,
            "PitStop": 2.0,
            "Stint": 1.5,
        },
    ),
    KNNConfig(
        name="yr_race_lap_knn",
        bucket_cols=("Year", "Race", "LapNumber"),
        numeric_cols=(
            "PitStop",
            "Stint",
            "TyreLife",
            "Position",
            LAPTIME,
            "LapTime_Delta",
            "Cumulative_Degradation",
            "Position_Change",
        ),
        weights={"TyreLife": 2.0, "Position": 2.0, "PitStop": 2.0, "Stint": 1.5},
    ),
    KNNConfig(
        name="yr_race_lap_compound_knn",
        bucket_cols=("Year", "Race", "LapNumber", "Compound"),
        numeric_cols=(
            "PitStop",
            "Stint",
            "TyreLife",
            "Position",
            LAPTIME,
            "LapTime_Delta",
            "Cumulative_Degradation",
            "Position_Change",
        ),
        cat_cols=(),
        weights={"TyreLife": 2.0, "Position": 2.0, "PitStop": 2.0, "Stint": 1.5},
    ),
    KNNConfig(
        name="yr_race_lap_pos_knn",
        bucket_cols=("Year", "Race", "LapNumber", "Position"),
        numeric_cols=("PitStop", "Stint", "TyreLife", LAPTIME, "LapTime_Delta", "Cumulative_Degradation"),
        weights={"TyreLife": 2.0, "PitStop": 2.0, "Stint": 1.5},
    ),
    KNNConfig(
        name="yr_race_phase_knn",
        bucket_cols=("Year", "Race", "Stint", "Compound"),
        numeric_cols=("LapNumber", "PitStop", "TyreLife", "Position", LAPTIME, "LapTime_Delta", "Cumulative_Degradation"),
        cat_cols=(),
        weights={"LapNumber": 3.0, "TyreLife": 2.0, "Position": 2.0, "PitStop": 2.0},
    ),
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={LAPTIME_RAW: LAPTIME}).copy()


def prediction_column(df: pd.DataFrame) -> str:
    for col in PRED_COLS:
        if col in df.columns and col != ID_COL:
            return col
    candidates = [col for col in df.columns if col not in {ID_COL, TARGET}]
    if candidates:
        return candidates[-1]
    raise ValueError(f"Could not infer prediction column from {list(df.columns)}")


def parse_ks(raw: str) -> list[int]:
    ks = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not ks or ks[0] < 1:
        raise ValueError("--ks must contain positive integers")
    return ks


def robust_stats(frames: list[pd.DataFrame], cols: list[str]) -> tuple[pd.Series, pd.Series]:
    combined = pd.concat([frame[cols] for frame in frames], ignore_index=True)
    median = combined.median(numeric_only=True)
    q75 = combined.quantile(0.75, numeric_only=True)
    q25 = combined.quantile(0.25, numeric_only=True)
    scale = (q75 - q25).replace(0, np.nan).fillna(combined.std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
    return median, scale


def make_design(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    median: pd.Series,
    scale: pd.Series,
    cat_levels: dict[str, list[str]],
    weights: dict[str, float],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if numeric_cols:
        num = df[numeric_cols].copy()
        num = num.fillna(median[numeric_cols])
        arr = ((num - median[numeric_cols]) / scale[numeric_cols]).to_numpy(dtype=np.float32)
        for idx, col in enumerate(numeric_cols):
            arr[:, idx] *= weights.get(col, 1.0)
        parts.append(arr)

    for col in cat_cols:
        levels = cat_levels[col]
        values = df[col].astype(str).fillna("__NA__")
        encoded = np.zeros((len(df), len(levels)), dtype=np.float32)
        index = {value: idx for idx, value in enumerate(levels)}
        for row_idx, value in enumerate(values):
            col_idx = index.get(value)
            if col_idx is not None:
                encoded[row_idx, col_idx] = weights.get(col, 1.0)
        parts.append(encoded)

    if not parts:
        raise ValueError("No columns available for nearest-neighbor design")
    return np.hstack(parts)


def group_keys(df: pd.DataFrame, bucket_cols: tuple[str, ...]) -> pd.Series:
    if len(bucket_cols) == 1:
        return df[bucket_cols[0]].astype(str)
    return df[list(bucket_cols)].astype(str).agg("\x1f".join, axis=1)


def query_knn_predictions(
    external: pd.DataFrame,
    query: pd.DataFrame,
    config: KNNConfig,
    ks: list[int],
    global_mean: float,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    numeric_cols = [col for col in config.numeric_cols if col in external.columns and col in query.columns]
    cat_cols = [col for col in config.cat_cols if col in external.columns and col in query.columns]
    if not numeric_cols and not cat_cols:
        raise ValueError(f"{config.name}: no usable columns")

    median, scale = robust_stats([external, query], numeric_cols)
    cat_levels = {
        col: sorted(pd.concat([external[col], query[col]], ignore_index=True).astype(str).fillna("__NA__").unique())
        for col in cat_cols
    }
    weights = config.weights or {}
    ext_x = make_design(external, numeric_cols, cat_cols, median, scale, cat_levels, weights)
    qry_x = make_design(query, numeric_cols, cat_cols, median, scale, cat_levels, weights)
    ext_y = external[TARGET].astype(float).to_numpy()

    ext_keys = group_keys(external, config.bucket_cols)
    qry_keys = group_keys(query, config.bucket_cols)
    ext_groups = ext_keys.groupby(ext_keys).groups
    qry_groups = qry_keys.groupby(qry_keys).groups

    max_k = max(ks)
    preds = {k: np.full(len(query), global_mean, dtype=np.float64) for k in ks}
    min_dist = np.full(len(query), np.nan, dtype=np.float64)
    covered = np.zeros(len(query), dtype=bool)
    candidate_sizes: list[int] = []

    for key, qry_idx_obj in qry_groups.items():
        ext_idx_obj = ext_groups.get(key)
        if ext_idx_obj is None:
            continue

        ext_idx = np.asarray(ext_idx_obj, dtype=int)
        qry_idx = np.asarray(qry_idx_obj, dtype=int)
        n_neighbors = min(max_k, len(ext_idx))
        if n_neighbors < 1:
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(ext_x[ext_idx])
        distances, local_idx = nn.kneighbors(qry_x[qry_idx], return_distance=True)
        labels = ext_y[ext_idx[local_idx]]
        covered[qry_idx] = True
        min_dist[qry_idx] = distances[:, 0]
        candidate_sizes.extend([len(ext_idx)] * len(qry_idx))

        for k in ks:
            use_k = min(k, n_neighbors)
            dist_k = distances[:, :use_k]
            label_k = labels[:, :use_k]
            if use_k == 1:
                pred = label_k[:, 0]
            else:
                weight = 1.0 / np.maximum(dist_k, 1e-6)
                pred = (label_k * weight).sum(axis=1) / weight.sum(axis=1)
            preds[k][qry_idx] = pred

    stats = {
        "coverage": float(covered.mean()),
        "mean_min_distance": float(np.nanmean(min_dist)) if np.isfinite(min_dist).any() else None,
        "median_min_distance": float(np.nanmedian(min_dist)) if np.isfinite(min_dist).any() else None,
        "mean_candidate_size": float(np.mean(candidate_sizes)) if candidate_sizes else None,
        "n_query_groups": int(len(qry_groups)),
        "n_external_groups": int(len(ext_groups)),
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
    }
    return preds, stats


def best_blend(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    max_alpha: float,
    grid_size: int = 401,
) -> tuple[float, float]:
    best_auc = float(roc_auc_score(y, base))
    best_alpha = 0.0
    for alpha in np.linspace(0.0, max_alpha, grid_size):
        pred = (1.0 - alpha) * base + alpha * candidate
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)
    return best_alpha, best_auc


def greedy_blend(
    y: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    candidates: list[dict[str, object]],
    max_alpha: float,
    max_steps: int,
    grid_size: int,
) -> tuple[list[dict[str, object]], float, np.ndarray, np.ndarray]:
    current_oof = base_oof.copy()
    current_test = base_test.copy()
    current_auc = float(roc_auc_score(y, current_oof))
    remaining = candidates.copy()
    steps: list[dict[str, object]] = []

    for step_idx in range(1, max_steps + 1):
        best_idx = -1
        best_alpha = 0.0
        best_auc = current_auc

        for idx, candidate in enumerate(remaining):
            alpha, auc = best_blend(y, current_oof, candidate["oof"], max_alpha, grid_size=grid_size)
            if auc > best_auc:
                best_idx = idx
                best_alpha = alpha
                best_auc = auc

        if best_idx < 0 or best_alpha <= 0.0:
            break

        picked = remaining.pop(best_idx)
        previous_auc = current_auc
        current_oof = (1.0 - best_alpha) * current_oof + best_alpha * picked["oof"]
        current_test = (1.0 - best_alpha) * current_test + best_alpha * picked["test"]
        current_auc = best_auc

        record = picked["record"].copy()
        step = {
            "step": step_idx,
            "name": record["name"],
            "k": record["k"],
            "alpha": best_alpha,
            "previous_auc": previous_auc,
            "blend_auc": current_auc,
            "delta": current_auc - previous_auc,
            "individual_delta_vs_base": record["delta_vs_base"],
        }
        print("GREEDY_STEP", json.dumps(step, indent=2), flush=True)
        steps.append(step)

    return steps, current_auc, current_oof, current_test


def align_oof_to_train(oof_df: pd.DataFrame, train: pd.DataFrame, pred_col: str) -> np.ndarray:
    if ID_COL in oof_df.columns:
        indexed = oof_df.set_index(ID_COL)
        return indexed.loc[train[ID_COL].to_numpy(), pred_col].astype(float).to_numpy()
    if len(oof_df) != len(train):
        raise ValueError("OOF file has no id column and length does not match train frame")
    return oof_df[pred_col].astype(float).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external nearest-neighbor transfer signals.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--external-csv", default="data/external/f1_strategy/f1_strategy_dataset_v4.csv", type=Path)
    parser.add_argument("--output-dir", default="submissions", type=Path)
    parser.add_argument("--base-oof", default="submissions/oof_external_prior_lgb_stack_v1_alpha0767.csv", type=Path)
    parser.add_argument("--base-submission", default="submissions/submission_external_prior_lgb_stack_v1_alpha0767.csv", type=Path)
    parser.add_argument("--ks", default="1,3,5,10,25")
    parser.add_argument("--configs", default=",".join(config.name for config in CONFIGS))
    parser.add_argument("--max-alpha", type=float, default=0.2)
    parser.add_argument("--greedy-steps", type=int, default=0)
    parser.add_argument("--greedy-max-alpha", type=float, default=0.05)
    parser.add_argument("--greedy-grid-size", type=int, default=31)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--suffix", default="external_nn_probe")
    args = parser.parse_args()

    train = normalize_columns(pd.read_csv(args.data_dir / "train.csv"))
    test = normalize_columns(pd.read_csv(args.data_dir / "test.csv"))
    external = normalize_columns(pd.read_csv(args.external_csv))
    if args.max_rows:
        train = train.sample(args.max_rows, random_state=42).sort_index().reset_index(drop=True)

    selected = {name.strip() for name in args.configs.split(",") if name.strip()}
    configs = [config for config in CONFIGS if config.name in selected]
    if not configs:
        raise ValueError(f"No configs selected from {sorted(config.name for config in CONFIGS)}")
    ks = parse_ks(args.ks)
    global_mean = float(external[TARGET].mean())
    y = train[TARGET].astype(int).to_numpy()

    base_oof_df = pd.read_csv(args.base_oof)
    base_sub_df = pd.read_csv(args.base_submission)
    base_oof_col = prediction_column(base_oof_df)
    base_sub_col = prediction_column(base_sub_df)
    base_oof = align_oof_to_train(base_oof_df, train, base_oof_col)
    base_test = base_sub_df[base_sub_col].astype(float).to_numpy()
    base_auc = float(roc_auc_score(y, base_oof))

    records: list[dict[str, object]] = []
    best_record: dict[str, object] | None = None
    best_oof: np.ndarray | None = None
    best_test: np.ndarray | None = None
    greedy_candidates: list[dict[str, object]] = []

    for config in configs:
        print(f"[config] {config.name} bucket={config.bucket_cols}", flush=True)
        train_preds, train_stats = query_knn_predictions(external, train, config, ks, global_mean)
        test_preds, test_stats = query_knn_predictions(external, test, config, ks, global_mean)

        for k in ks:
            nn_oof = train_preds[k]
            nn_test = test_preds[k]
            nn_auc = float(roc_auc_score(y, nn_oof))
            alpha, blend_auc = best_blend(y, base_oof, nn_oof, args.max_alpha)
            record = {
                "name": config.name,
                "bucket_cols": list(config.bucket_cols),
                "k": k,
                "nn_auc": nn_auc,
                "base_auc": base_auc,
                "best_alpha_nn": alpha,
                "blend_auc": blend_auc,
                "delta_vs_base": blend_auc - base_auc,
                "train_stats": train_stats,
                "test_stats": test_stats,
            }
            print(json.dumps(record, indent=2), flush=True)
            records.append(record)
            if best_record is None or blend_auc > best_record["blend_auc"]:
                best_record = record
                best_oof = (1.0 - alpha) * base_oof + alpha * nn_oof
                best_test = (1.0 - alpha) * base_test + alpha * nn_test
            if blend_auc > base_auc and alpha > 0.0:
                greedy_candidates.append({"record": record, "oof": nn_oof.copy(), "test": nn_test.copy()})

    if best_record is None or best_oof is None or best_test is None:
        raise RuntimeError("No nearest-neighbor records were produced")

    greedy_summary = None
    if args.greedy_steps > 0 and greedy_candidates:
        steps, greedy_auc, greedy_oof, greedy_test = greedy_blend(
            y=y,
            base_oof=base_oof,
            base_test=base_test,
            candidates=greedy_candidates,
            max_alpha=args.greedy_max_alpha,
            max_steps=args.greedy_steps,
            grid_size=args.greedy_grid_size,
        )
        greedy_summary = {
            "max_steps": args.greedy_steps,
            "max_alpha": args.greedy_max_alpha,
            "grid_size": args.greedy_grid_size,
            "candidate_count": len(greedy_candidates),
            "steps": steps,
            "blend_auc": greedy_auc,
            "delta_vs_base": greedy_auc - base_auc,
        }
        if greedy_auc > best_record["blend_auc"]:
            best_record = {
                "name": "greedy_nn_blend",
                "base_auc": base_auc,
                "blend_auc": greedy_auc,
                "delta_vs_base": greedy_auc - base_auc,
                "steps": steps,
            }
            best_oof = greedy_oof
            best_test = greedy_test

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"submission_{args.suffix}.csv"
    oof_path = args.output_dir / f"oof_{args.suffix}.csv"
    summary_path = args.output_dir / f"summary_{args.suffix}.json"

    submission = base_sub_df.copy()
    submission[base_sub_col] = np.clip(best_test, 0.0, 1.0)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), TARGET: y, "pred": np.clip(best_oof, 0.0, 1.0)}).to_csv(
        oof_path, index=False
    )

    summary = {
        "base_oof": str(args.base_oof),
        "base_submission": str(args.base_submission),
        "base_auc": base_auc,
        "best": best_record,
        "greedy": greedy_summary,
        "records": records,
        "output": str(submission_path),
        "oof_output": str(oof_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BEST", json.dumps(best_record, indent=2), flush=True)


if __name__ == "__main__":
    main()
