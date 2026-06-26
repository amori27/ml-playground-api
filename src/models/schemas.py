from pydantic import BaseModel
from typing import Any


class HealthResponse(BaseModel):
    status: str
    models_available: int
    trained_count: int


class DatasetInfo(BaseModel):
    name: str
    description: str
    n_samples: int
    n_features: int
    n_classes: int
    feature_names: list[str]
    target_names: list[str]


class TrainRequest(BaseModel):
    dataset_name: str
    model_type: str
    params: dict[str, Any] = {}


class TrainResponse(BaseModel):
    model_id: str
    dataset_name: str
    model_type: str
    accuracy: float
    params: dict[str, Any]


class ModelInfo(BaseModel):
    model_id: str
    dataset_name: str
    model_type: str
    accuracy: float
    params: dict[str, Any]
    created_at: str


class PredictRequest(BaseModel):
    model_id: str
    features: list[float]


class PredictResponse(BaseModel):
    model_id: str
    prediction: int
    prediction_label: str
    probabilities: list[float] | None = None
