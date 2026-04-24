from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse

from app.core.config import settings
from app.ml.train import train_and_save_models
from app.schemas.risk import FeatureContribution, ModelMetric, PatientInput, PredictionResponse
from app.services.recommendation_service import generate_recommendations
from app.utils.risk import probability_to_risk_score, risk_level_from_score


class ModelService:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.metadata: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        self.ready = False

    def ensure_ready(self) -> None:
        metadata_file = self.artifacts_dir / "metadata.json"
        if not metadata_file.exists():
            if settings.auto_train_models:
                train_and_save_models(output_dir=self.artifacts_dir)
            else:
                raise RuntimeError(
                    "Model artifacts are missing. Run model training separately with "
                    "'python -m app.ml.train' from the backend directory."
                )

        with metadata_file.open("r", encoding="utf-8") as fp:
            self.metadata = json.load(fp)

        self.models.clear()
        for model_name in self.metadata["metrics"].keys():
            model_path = self.artifacts_dir / f"{model_name}.joblib"
            if not model_path.exists():
                if settings.auto_train_models:
                    train_and_save_models(output_dir=self.artifacts_dir)
                    with metadata_file.open("r", encoding="utf-8") as fp:
                        self.metadata = json.load(fp)
                else:
                    raise RuntimeError(
                        f"Missing model artifact: {model_path.name}. Run training separately with "
                        "'python -m app.ml.train' from the backend directory."
                    )
            self.models[model_name] = joblib.load(model_path)

        self.ready = True

    def available_model_metrics(self) -> list[ModelMetric]:
        if not self.ready:
            self.ensure_ready()

        metrics: list[ModelMetric] = []
        for model_name, values in self.metadata["metrics"].items():
            metrics.append(ModelMetric(model_name=model_name, **values))
        return metrics

    def _parse_shap_output(self, shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                return np.asarray(shap_values[1])[0]
            return np.asarray(shap_values[0])[0]

        array = np.asarray(shap_values)
        if array.ndim == 3:
            return array[0, :, 1]
        if array.ndim == 2:
            return array[0]
        return array

    def _feature_origin(self, transformed_name: str) -> str:
        if transformed_name.startswith("num__"):
            return transformed_name.replace("num__", "", 1)
        if transformed_name.startswith("cat__"):
            raw = transformed_name.replace("cat__", "", 1)
            for candidate in self.metadata.get("categorical_features", []):
                if raw.startswith(f"{candidate}_") or raw == candidate:
                    return candidate
            return raw.split("_")[0]
        return transformed_name

    def _row_feature_value(self, row_df: pd.DataFrame, feature: str) -> str | float | int:
        value = row_df.iloc[0].to_dict().get(feature, "")
        if isinstance(value, (str, int, float)):
            return value
        return str(value)

    def _explain_with_shap(self, pipeline: Any, row_df: pd.DataFrame) -> list[FeatureContribution]:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]

        transformed = preprocessor.transform(row_df)
        if sparse.issparse(transformed):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        contrib_values = self._parse_shap_output(shap_values)

        grouped: dict[str, float] = {}
        for name, val in zip(feature_names, contrib_values):
            origin = self._feature_origin(str(name))
            grouped[origin] = grouped.get(origin, 0.0) + float(val)

        ordered = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        return [
            FeatureContribution(
                feature=feature,
                feature_value=self._row_feature_value(row_df, feature),
                impact_score=round(float(abs(score)), 6),
                direction="increases" if score >= 0 else "decreases",
            )
            for feature, score in ordered
        ]

    def _explain_with_coefficients(self, pipeline: Any, row_df: pd.DataFrame) -> list[FeatureContribution]:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]

        transformed = preprocessor.transform(row_df)
        if sparse.issparse(transformed):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "coef_"):
            coef = estimator.coef_[0]
            contrib = transformed[0] * coef
        elif hasattr(estimator, "feature_importances_"):
            contrib = transformed[0] * estimator.feature_importances_
        else:
            raise ValueError("Estimator has no interpretable coefficients or feature importances")

        grouped: dict[str, float] = {}
        for name, val in zip(feature_names, contrib):
            origin = self._feature_origin(str(name))
            grouped[origin] = grouped.get(origin, 0.0) + float(val)

        ordered = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        return [
            FeatureContribution(
                feature=feature,
                feature_value=self._row_feature_value(row_df, feature),
                impact_score=round(float(abs(score)), 6),
                direction="increases" if score >= 0 else "decreases",
            )
            for feature, score in ordered
        ]

    def _explain_from_value_rates(self, row_df: pd.DataFrame) -> list[FeatureContribution]:
        rates_by_feature: dict[str, dict[str, float]] = self.metadata.get("feature_value_target_rate", {})
        global_rate = float(self.metadata.get("target_positive_rate", 0.5))

        contributions: list[FeatureContribution] = []
        for feature in self.metadata.get("feature_columns", []):
            value = str(self._row_feature_value(row_df, feature))
            value_rate = float(rates_by_feature.get(feature, {}).get(value, global_rate))
            delta = value_rate - global_rate
            contributions.append(
                FeatureContribution(
                    feature=feature,
                    feature_value=value,
                    impact_score=round(abs(delta), 6),
                    direction="increases" if delta >= 0 else "decreases",
                )
            )
        return sorted(contributions, key=lambda item: item.impact_score, reverse=True)[:5]

    def explain(self, model_name: str, row_df: pd.DataFrame) -> list[FeatureContribution]:
        pipeline = self.models[model_name]

        if model_name in {"random_forest", "decision_tree"}:
            try:
                return self._explain_with_shap(pipeline, row_df)
            except Exception:
                pass

        try:
            return self._explain_with_coefficients(pipeline, row_df)
        except Exception:
            return self._explain_from_value_rates(row_df)

    def _resolve_model_name(self, requested_model_name: str | None) -> str:
        selected_model = (
            requested_model_name
            or (settings.inference_model_name.strip() if settings.inference_model_name else None)
            or self.metadata.get("best_model")
            or settings.default_model_name
        )
        if selected_model not in self.models:
            raise ValueError(f"Model '{selected_model}' is not available")
        return selected_model

    def _build_reliability_warnings(self, payload: PatientInput) -> list[str]:
        warnings: list[str] = []
        value_frequencies: dict[str, dict[str, float]] = self.metadata.get("feature_value_frequencies", {})

        for feature, value in payload.model_dump().items():
            observed_ratio = float(value_frequencies.get(feature, {}).get(str(value), 0.0))
            if observed_ratio <= 0.01:
                warnings.append(
                    f"Value '{value}' for {feature.replace('_', ' ')} appears in under 1% of training records. "
                    "Prediction confidence may be lower."
                )

        return warnings

    def predict(self, payload: PatientInput, model_name: str | None = None) -> PredictionResponse:
        if not self.ready:
            self.ensure_ready()

        selected_model = self._resolve_model_name(model_name)
        row = pd.DataFrame([payload.model_dump()])
        pipeline = self.models[selected_model]

        probability = float(pipeline.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        risk_score = probability_to_risk_score(probability)
        risk_level = risk_level_from_score(risk_score)
        top_factors = self.explain(selected_model, row)
        recommendations = generate_recommendations(payload, probability=probability)
        warnings = self._build_reliability_warnings(payload)

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 6),
            risk_score=risk_score,
            risk_level=risk_level,
            model_name=selected_model,
            top_factors=top_factors,
            recommendations=recommendations,
            warnings=warnings,
        )


model_service = ModelService(settings.model_artifacts_dir)
