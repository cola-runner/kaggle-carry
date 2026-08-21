"""Yekenot's feature engineering recipe (replicated from his public S6E5 notebook).

Reference: `kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit` — uses this FE
together with a tuned pytabkit RealMLP_TD_Classifier to reach OOF 0.9540.

Produces:
  - 5 arithmetic interaction features
  - For every numerical column (including 2 new interactions): a floor-factorized
    categorical version (`<col>_cat_`)
  - Count-encoded numeric features for categorical + Year_cat_ + PitStop_cat_
  - KBinsDiscretizer-quantile bin features: RaceProgress (200 bins), LapTime (7 bins)
  - Interaction-key categoricals: Race_Compound_, Race_Year_
    (these are the keys passed to OOF-safe TargetEncoder inside each fold)

Usage:
    fe = YekenotFE()
    X = fe.fit_transform(X_raw)
    X_test = fe.transform(X_test_raw)
    X_external = fe.transform(X_external_raw)
    print('combo_names for target encoding:', fe.combo_names)

The TargetEncoder step itself is intentionally NOT inside this module — it must
run inside the fold loop with the train-fold labels (no leakage to val/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


LAPTIME_COL = "LapTime (s)"

INTERACTION_NUM_NEW = [
    "_LapNumber_/_RaceProgress",
    "_TyreLife_/_LapNumber",
]
INTERACTION_NUM_ALL = INTERACTION_NUM_NEW + [
    "_LapTime (s)_*_Cumulative_Degradation",
    "_LapTime (s)_*_Cumulative_Degradation_abs",
    "_LapTime (s)_/_Cumulative_Degradation_abs",
]
IMPORTANT_COMBOS = [
    ("Race", "Compound"),
    ("Race", "Year"),
]


class YekenotFE:
    """Stateful FE producer; call fit() on train then transform() on test/external."""

    def __init__(self) -> None:
        self.category_map: dict = {}
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.combo_names: list[str] = []
        self.new_cat_cols: list[str] = []
        self.new_num_cols: list[str] = []
        self._fitted = False

    def _init_columns(self, df: pd.DataFrame) -> None:
        # raw columns
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    def _apply(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        # 1. Arithmetic interactions ----------------------------------------
        df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
        df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
        df["_LapTime (s)_*_Cumulative_Degradation"] = (df[LAPTIME_COL] * df["Cumulative_Degradation"]).astype("float32")
        df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (df[LAPTIME_COL] * df["Cumulative_Degradation"].abs()).astype("float32")
        df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (df[LAPTIME_COL] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")

        # 2. Floor-factorize all numericals (+ 2 of the new interactions) ----
        for col in self.num_cols + INTERACTION_NUM_NEW:
            cat_name = f"{col}_cat_" if col in self.num_cols else f"{col[1:]}_cat_"
            if fit:
                codes, uniques = np.floor(df[col]).factorize()
                self.category_map[col] = uniques
            else:
                uniques = self.category_map[col]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
            df[cat_name] = codes.astype("int32") if fit else codes
            df[cat_name] = df[cat_name].astype(str)

        # 3. Count encoding for cat cols + Year_cat_ + PitStop_cat_ ----------
        for col in self.cat_cols + ["Year_cat_", "PitStop_cat_"]:
            count_name = f"_{col}_count" if col in self.cat_cols else f"_{col[:-1]}_count"
            if fit:
                count_map = df[col].value_counts()
                self.category_map[count_name] = count_map
            else:
                count_map = self.category_map[count_name]
            df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

        # 4. KBinsDiscretizer quantile bins ---------------------------------
        bin_config = {"RaceProgress": [200], LAPTIME_COL: [7]}
        for col, bins_list in bin_config.items():
            for n_bins in bins_list:
                strategy = "quantile"
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                    self.category_map[bin_name] = kb
                else:
                    kb = self.category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype("int32")
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

        # 5. Interaction-key combos for downstream TargetEncoder -----------
        combo_names = []
        for cols in IMPORTANT_COMBOS:
            combo_name = "_".join(cols) + "_"
            combo_names.append(combo_name)
            combo_series = df[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + "_" + df[col].astype(str)
            if fit:
                codes, uniques = pd.factorize(combo_series, sort=False)
                self.category_map[combo_name] = uniques
            else:
                uniques = self.category_map[combo_name]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = combo_series.map(code_map).fillna(-1).astype("int32")
            df[combo_name] = codes
            df[combo_name] = df[combo_name].astype(str)

        if fit:
            self.combo_names = combo_names
            self.new_cat_cols = [c for c in df.columns if c.endswith("_")]
            self.new_num_cols = [c for c in df.columns if c.startswith("_")]
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_columns(df)
        out = self._apply(df, fit=True)
        self._fitted = True
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on the training data first.")
        return self._apply(df, fit=False)
