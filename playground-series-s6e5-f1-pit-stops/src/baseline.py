from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def extract_zip_if_needed(data_dir: Path) -> None:
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    existing = {path.name for path in data_dir.glob("*.csv")}
    if required.issubset(existing):
        return

    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        missing = ", ".join(sorted(required - existing))
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Download the Kaggle data zip first."
        )

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} ...")
        with ZipFile(zip_path) as zf:
            zf.extractall(data_dir)


def read_competition_files(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extract_zip_if_needed(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def infer_id_and_target(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[str, str]:
    id_col = sample.columns[0]
    sample_target_cols = [col for col in sample.columns if col != id_col]
    if sample_target_cols and sample_target_cols[0] in train.columns:
        return id_col, sample_target_cols[0]

    train_only_cols = [col for col in train.columns if col not in test.columns and col != id_col]
    if len(train_only_cols) != 1:
        raise ValueError(
            "Could not infer the target column. "
            f"Train-only columns were: {train_only_cols}"
        )
    return id_col, train_only_cols[0]


def prepare_target(y: pd.Series) -> tuple[pd.Series, str]:
    if y.nunique(dropna=False) != 2:
        raise ValueError(f"This baseline expects a binary target, got {y.nunique()} classes.")

    if pd.api.types.is_numeric_dtype(y):
        return y, "numeric"

    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
    print(f"Encoded target classes: {dict(enumerate(encoder.classes_))}")
    return encoded, "label_encoded"


def build_model(numeric_cols: list[str], categorical_cols: list[str], random_state: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Beginner baseline for Kaggle S6E5.")
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--output", default="submissions/submission_baseline.csv", type=Path)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--valid-size", default=0.2, type=float)
    args = parser.parse_args()

    train, test, sample = read_competition_files(args.data_dir)
    id_col, target_col = infer_id_and_target(train, test, sample)

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"ID column:   {id_col}")
    print(f"Target:      {target_col}")

    feature_cols = [col for col in test.columns if col in train.columns and col != id_col]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y, target_mode = prepare_target(train[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    print(f"Features:    {len(feature_cols)}")
    print(f"Numeric:     {len(numeric_cols)}")
    print(f"Categorical: {len(categorical_cols)}")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    pipeline = build_model(numeric_cols, categorical_cols, args.random_state)
    pipeline.fit(X_train, y_train)

    valid_pred = pipeline.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_pred)
    print(f"Validation ROC AUC: {valid_auc:.6f}")

    print("Refitting on all training data ...")
    pipeline.fit(X, y)
    test_pred = pipeline.predict_proba(X_test)[:, 1]
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = sample.copy()
    prediction_col = [col for col in submission.columns if col != id_col][0]
    submission[prediction_col] = test_pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "competition": "playground-series-s6e5",
        "target": target_col,
        "target_mode": target_mode,
        "id_column": id_col,
        "rows_train": int(train.shape[0]),
        "rows_test": int(test.shape[0]),
        "features": len(feature_cols),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "validation_roc_auc": float(valid_auc),
        "output": str(args.output),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote submission: {args.output}")
    print(f"Wrote summary:    {summary_path}")


if __name__ == "__main__":
    main()
