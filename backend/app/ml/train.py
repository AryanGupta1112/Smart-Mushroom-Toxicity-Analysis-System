from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.config import settings
from app.ml.feature_catalog import CATEGORICAL_FEATURES, EXPECTED_FEATURES, FEATURE_DEFINITIONS, NUMERIC_FEATURES

RAW_TO_CANONICAL = {
    "cap-shape": "cap_shape",
    "cap-surface": "cap_surface",
    "cap-color": "cap_color",
    "bruises%3F": "bruises",
    "odor": "odor",
    "gill-attachment": "gill_attachment",
    "gill-spacing": "gill_spacing",
    "gill-size": "gill_size",
    "gill-color": "gill_color",
    "stalk-shape": "stalk_shape",
    "stalk-root": "stalk_root",
    "stalk-surface-above-ring": "stalk_surface_above_ring",
    "stalk-surface-below-ring": "stalk_surface_below_ring",
    "stalk-color-above-ring": "stalk_color_above_ring",
    "stalk-color-below-ring": "stalk_color_below_ring",
    "veil-type": "veil_type",
    "veil-color": "veil_color",
    "ring-number": "ring_number",
    "ring-type": "ring_type",
    "spore-print-color": "spore_print_color",
    "population": "population",
    "habitat": "habitat",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors (KNN)",
    "svm": "Support Vector Machine (SVM)",
}

MODEL_PLAIN_EXPLANATIONS = {
    "logistic_regression": "Learns weighted importance for each feature value.",
    "decision_tree": "Builds if-then style decision rules.",
    "random_forest": "Combines many trees to make a stronger final decision.",
    "knn": "Compares each mushroom to similar examples from training data.",
    "svm": "Finds the best boundary between safe-looking and risky patterns.",
}


@dataclass
class DatasetBundle:
    x: pd.DataFrame
    y: pd.Series
    source_name: str
    source_rows: int


def _safe_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_mushroom_dataset() -> DatasetBundle:
    bundle = fetch_openml(name="mushroom", version=1, as_frame=True, parser="auto")
    frame = bundle.frame.copy()
    if frame is None or frame.empty:
        raise RuntimeError("OpenML mushroom dataset is empty")

    frame = frame.rename(columns=RAW_TO_CANONICAL)
    missing = [feature for feature in EXPECTED_FEATURES if feature not in frame.columns]
    if missing:
        raise RuntimeError("Mushroom dataset missing required features: " + ", ".join(missing))

    x = frame[EXPECTED_FEATURES].copy()
    for feature in EXPECTED_FEATURES:
        x[feature] = _safe_str(x[feature])

    target_raw = _safe_str(frame["class"])
    y = (target_raw == "p").astype(int)

    source_rows = int(x.shape[0])
    max_rows = int(settings.training_max_rows)
    if max_rows > 0 and source_rows > max_rows:
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=max_rows,
            random_state=42,
            stratify=y,
        )

    return DatasetBundle(x=x, y=y, source_name="OpenML Mushroom (UCI)", source_rows=source_rows)


def _build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    if NUMERIC_FEATURES:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_transformer, NUMERIC_FEATURES))

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers.append(("cat", categorical_transformer, CATEGORICAL_FEATURES))

    return ColumnTransformer(transformers=transformers)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def train_and_save_models(output_dir: Path | None = None) -> dict[str, Any]:
    output_path = output_dir or settings.model_artifacts_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78, flush=True)
    print("Smart Mushroom Toxicity Analysis - Training Started", flush=True)
    print("=" * 78, flush=True)
    print("This process learns patterns from labeled mushroom records.", flush=True)
    print("Goal: estimate how likely a mushroom pattern is poisonous.", flush=True)

    print("\nStep 1/4: Loading dataset from OpenML...", flush=True)
    dataset = load_mushroom_dataset()
    print(
        f"Loaded {dataset.x.shape[0]} records with {dataset.x.shape[1]} input features "
        f"from '{dataset.source_name}'.",
        flush=True,
    )

    print("\nStep 2/4: Splitting data into training and testing sets...", flush=True)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.x,
        dataset.y,
        test_size=0.2,
        random_state=42,
        stratify=dataset.y,
    )
    print(
        f"Training set: {x_train.shape[0]} records | Testing set: {x_test.shape[0]} records",
        flush=True,
    )

    model_configs: dict[str, tuple[Any, bool]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=300, C=2.0, solver="saga", verbose=1, random_state=42),
            True,
        ),
        "decision_tree": (DecisionTreeClassifier(max_depth=22, min_samples_leaf=1, random_state=42), False),
        "random_forest": (
            RandomForestClassifier(n_estimators=450, max_depth=40, min_samples_split=2, verbose=1, random_state=42),
            False,
        ),
        "knn": (KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
        "svm": (SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, verbose=True, random_state=42), True),
    }

    metrics_by_model: dict[str, dict[str, float]] = {}
    total_models = len(model_configs)
    print(f"\nStep 3/4: Training {total_models} machine-learning models...", flush=True)

    for idx, (model_name, (estimator, scale_numeric)) in enumerate(model_configs.items(), start=1):
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        simple_note = MODEL_PLAIN_EXPLANATIONS.get(model_name, "Model training in progress.")

        print("\n" + "-" * 78, flush=True)
        print(f"Model {idx}/{total_models}: {display_name}", flush=True)
        print(f"How this model works (simple): {simple_note}", flush=True)
        print("Status: Training started...", flush=True)

        started = time.perf_counter()
        preprocessor = _build_preprocessor(scale_numeric=scale_numeric)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1]

        metrics = _compute_metrics(y_test.to_numpy(), y_pred, y_prob)
        metrics_by_model[model_name] = metrics
        elapsed = time.perf_counter() - started
        print(f"Status: Finished in {elapsed:.2f} seconds.", flush=True)
        print("Result summary (simple):", flush=True)
        print(f"- Overall correctness (Accuracy): {_pct(metrics['accuracy'])}", flush=True)
        print(f"- When model says poisonous, how often right (Precision): {_pct(metrics['precision'])}", flush=True)
        print(f"- How many poisonous cases it catches (Recall): {_pct(metrics['recall'])}", flush=True)
        print(f"- Balance between precision and recall (F1): {_pct(metrics['f1_score'])}", flush=True)
        print(f"- Ranking quality across thresholds (ROC-AUC): {_pct(metrics['roc_auc'])}", flush=True)

        model_file = output_path / f"{model_name}.joblib"
        joblib.dump(pipeline, model_file)
        print(f"Saved model file: {model_file}", flush=True)

    best_model = max(metrics_by_model, key=lambda name: metrics_by_model[name]["accuracy"])
    positive_rate = float(dataset.y.mean())
    best_display_name = MODEL_DISPLAY_NAMES.get(best_model, best_model)

    feature_stats: dict[str, dict[str, Any]] = {}
    feature_value_target_rate: dict[str, dict[str, float]] = {}
    feature_value_frequencies: dict[str, dict[str, float]] = {}
    for feature in EXPECTED_FEATURES:
        options = FEATURE_DEFINITIONS[feature]["options"]
        feature_stats[feature] = {
            "min": 0,
            "max": len(options) - 1,
            "type": "categorical",
            "description": str(FEATURE_DEFINITIONS[feature]["description"]),
            "allowed_values": [{"code": code, "label": label} for code, label in options.items()],
        }

        rates: dict[str, float] = {}
        frequencies: dict[str, float] = {}
        grouped_target = dataset.y.groupby(dataset.x[feature]).mean()
        grouped_count = dataset.x[feature].value_counts(normalize=True)
        for code in options.keys():
            rates[code] = float(grouped_target.get(code, positive_rate))
            frequencies[code] = float(grouped_count.get(code, 0.0))
        feature_value_target_rate[feature] = rates
        feature_value_frequencies[feature] = frequencies

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset.source_name,
        "dataset_rows_raw": dataset.source_rows,
        "dataset_rows": int(dataset.x.shape[0]),
        "feature_columns": EXPECTED_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_model": best_model,
        "metrics": metrics_by_model,
        "target_positive_rate": positive_rate,
        "feature_stats": feature_stats,
        "feature_value_target_rate": feature_value_target_rate,
        "feature_value_frequencies": feature_value_frequencies,
    }

    with (output_path / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print("\nStep 4/4: Saving final training metadata...", flush=True)
    print(f"Metadata saved to: {output_path / 'metadata.json'}", flush=True)
    print("\n" + "=" * 78, flush=True)
    print("Training Complete", flush=True)
    print("=" * 78, flush=True)
    print(f"Best model on this training run: {best_display_name}", flush=True)
    print(f"Poisonous class ratio in dataset: {_pct(positive_rate)}", flush=True)
    print("You can now run the API and use this model in the dashboard.", flush=True)

    return metadata


if __name__ == "__main__":
    result = train_and_save_models()
    print(json.dumps(result, indent=2))
