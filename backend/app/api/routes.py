from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.schemas.risk import (
    FeaturesInfoResponse,
    HealthResponse,
    ModelsPerformanceResponse,
    PatientInput,
    PredictionResponse,
    TrainingSummaryResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from app.services.history_service import create_history_record, list_history_records
from app.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    if not model_service.ready:
        model_service.ensure_ready()
    return HealthResponse(status="ok", model_loaded=model_service.ready, timestamp=datetime.now(timezone.utc))


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput, model_name: str | None = None, db: Session = Depends(get_db)) -> PredictionResponse:
    try:
        result = model_service.predict(payload, model_name=model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    create_history_record(db, payload.model_dump(), result)
    return result


@router.get("/models/performance", response_model=ModelsPerformanceResponse)
def model_performance() -> ModelsPerformanceResponse:
    if not model_service.ready:
        model_service.ensure_ready()

    return ModelsPerformanceResponse(
        best_model=model_service.metadata["best_model"],
        metrics=model_service.available_model_metrics(),
    )


@router.get("/models/training-summary", response_model=TrainingSummaryResponse)
def model_training_summary() -> TrainingSummaryResponse:
    if not model_service.ready:
        model_service.ensure_ready()

    metadata = model_service.metadata
    metrics = metadata.get("metrics", {})
    generated_at_raw = metadata.get("generated_at")
    generated_at = None
    if isinstance(generated_at_raw, str):
        try:
            generated_at = datetime.fromisoformat(generated_at_raw)
        except ValueError:
            generated_at = None

    return TrainingSummaryResponse(
        dataset_source=str(metadata.get("dataset_source", "OpenML Mushroom (UCI)")),
        dataset_rows=int(metadata.get("dataset_rows", 0)),
        feature_count=len(metadata.get("feature_columns", [])),
        numeric_feature_count=len(metadata.get("numeric_features", [])),
        categorical_feature_count=len(metadata.get("categorical_features", [])),
        generated_at=generated_at,
        best_model=str(metadata.get("best_model", settings.default_model_name)),
        active_inference_model=(
            settings.inference_model_name.strip()
            if settings.inference_model_name.strip()
            else str(metadata.get("best_model", settings.default_model_name))
        ),
        algorithms_trained=sorted(str(model_name) for model_name in metrics.keys()),
        train_test_split="80% training / 20% testing with class-balanced stratification",
        training_library="scikit-learn",
        training_script="backend/app/ml/train.py",
        artifacts_dir=str(settings.model_artifacts_dir),
    )


@router.post("/what-if", response_model=WhatIfResponse)
def what_if_analysis(payload: WhatIfRequest, model_name: str | None = None, db: Session = Depends(get_db)) -> WhatIfResponse:
    if not model_service.ready:
        model_service.ensure_ready()

    base_result = model_service.predict(payload.base_input, model_name=model_name)
    modified_result = model_service.predict(payload.modified_input, model_name=model_name)

    create_history_record(db, payload.base_input.model_dump(), base_result)
    create_history_record(db, payload.modified_input.model_dump(), modified_result)

    return WhatIfResponse(
        base_result=base_result,
        modified_result=modified_result,
        probability_delta=round(modified_result.probability - base_result.probability, 6),
        risk_score_delta=modified_result.risk_score - base_result.risk_score,
        risk_level_changed=modified_result.risk_level != base_result.risk_level,
    )


@router.get("/features/info", response_model=FeaturesInfoResponse)
def feature_info() -> FeaturesInfoResponse:
    if not model_service.ready:
        model_service.ensure_ready()

    stats = model_service.metadata["feature_stats"]
    features = [
        {
            "name": name,
            "type": stat["type"],
            "min": stat["min"],
            "max": stat["max"],
            "description": stat["description"],
            "allowed_values": stat.get("allowed_values", []),
        }
        for name, stat in stats.items()
    ]
    return FeaturesInfoResponse(features=features)


@router.get("/history")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    return list_history_records(db, limit=limit)
