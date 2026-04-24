from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.ml.feature_catalog import FEATURE_DEFINITIONS


ALLOWED_FEATURE_VALUES: dict[str, set[str]] = {
    feature: set(definition["options"].keys()) for feature, definition in FEATURE_DEFINITIONS.items()
}


class PatientInput(BaseModel):
    cap_shape: str = Field(..., description="Cap shape code")
    cap_surface: str = Field(..., description="Cap surface texture code")
    cap_color: str = Field(..., description="Cap color code")
    bruises: str = Field(..., description="Bruises present code")
    odor: str = Field(..., description="Odor code")
    gill_attachment: str = Field(..., description="Gill attachment code")
    gill_spacing: str = Field(..., description="Gill spacing code")
    gill_size: str = Field(..., description="Gill size code")
    gill_color: str = Field(..., description="Gill color code")
    stalk_shape: str = Field(..., description="Stalk shape code")
    stalk_root: str = Field(..., description="Stalk root code")
    stalk_surface_above_ring: str = Field(..., description="Stalk surface above ring code")
    stalk_surface_below_ring: str = Field(..., description="Stalk surface below ring code")
    stalk_color_above_ring: str = Field(..., description="Stalk color above ring code")
    stalk_color_below_ring: str = Field(..., description="Stalk color below ring code")
    veil_type: str = Field(..., description="Veil type code")
    veil_color: str = Field(..., description="Veil color code")
    ring_number: str = Field(..., description="Ring count code")
    ring_type: str = Field(..., description="Ring type code")
    spore_print_color: str = Field(..., description="Spore print color code")
    population: str = Field(..., description="Population code")
    habitat: str = Field(..., description="Habitat code")

    @field_validator("*")
    @classmethod
    def validate_category_codes(cls, value: str, info) -> str:
        if info.field_name is None:
            return value
        allowed = ALLOWED_FEATURE_VALUES.get(info.field_name)
        if allowed is None:
            return value
        if value not in allowed:
            raise ValueError(f"Invalid value for {info.field_name}: {value}")
        return value


class FeatureContribution(BaseModel):
    feature: str
    feature_value: str | float | int
    impact_score: float
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_score: int
    risk_level: str
    model_name: str
    top_factors: list[FeatureContribution]
    recommendations: list[str]
    warnings: list[str] = Field(default_factory=list)


class ModelMetric(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class ModelsPerformanceResponse(BaseModel):
    best_model: str
    metrics: list[ModelMetric]


class TrainingSummaryResponse(BaseModel):
    dataset_source: str
    dataset_rows: int
    feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    generated_at: datetime | None
    best_model: str
    active_inference_model: str
    algorithms_trained: list[str]
    train_test_split: str
    training_library: str
    training_script: str
    artifacts_dir: str


class WhatIfRequest(BaseModel):
    base_input: PatientInput
    modified_input: PatientInput


class WhatIfResponse(BaseModel):
    base_result: PredictionResponse
    modified_result: PredictionResponse
    probability_delta: float
    risk_score_delta: int
    risk_level_changed: bool


class FeatureValueOption(BaseModel):
    code: str
    label: str


class FeatureInfoItem(BaseModel):
    name: str
    type: Literal["numeric", "categorical"]
    min: float | int
    max: float | int
    description: str
    allowed_values: list[FeatureValueOption] = Field(default_factory=list)


class FeaturesInfoResponse(BaseModel):
    features: list[FeatureInfoItem]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    timestamp: datetime


class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    model_name: str
    prediction: int
    probability: float
    risk_score: int
    risk_level: str
    input_payload: dict
    top_factors: list[FeatureContribution]


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
