from fastapi import FastAPI, HTTPException
import numpy as np
from src.core.datasets import DATASETS
from src.core.datasets import get_dataset_info
from src.core.trainer import train_model, MODELS
from src.models.schemas import (
    DatasetInfo,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from src.models.registry import register, get, list_models, count as model_count

app = FastAPI(title="ML Playground API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_available=len(DATASETS),
        trained_count=model_count(),
    )


@app.get("/datasets", response_model=list[DatasetInfo])
def datasets():
    return [get_dataset_info(name) for name in DATASETS]


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if req.dataset_name not in DATASETS:
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found")
    if req.model_type not in MODELS:
        raise HTTPException(404, f"Model type '{req.model_type}' not found")

    model, accuracy, target_names, feature_names, effective_params = train_model(
        req.dataset_name, req.model_type, req.params
    )
    model_id = register(
        model,
        req.dataset_name,
        req.model_type,
        accuracy,
        effective_params,
        target_names,
        feature_names,
    )

    return TrainResponse(
        model_id=model_id,
        dataset_name=req.dataset_name,
        model_type=req.model_type,
        accuracy=accuracy,
        params=effective_params,
    )


@app.get("/models", response_model=list[ModelInfo])
def models():
    return list_models()


@app.get("/models/{model_id}", response_model=ModelInfo)
def model_detail(model_id: str):
    try:
        entry = get(model_id)
    except KeyError:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return ModelInfo(
        model_id=model_id,
        dataset_name=entry["dataset_name"],
        model_type=entry["model_type"],
        accuracy=entry["accuracy"],
        params=entry["params"],
        created_at=entry["created_at"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        entry = get(req.model_id)
    except KeyError:
        raise HTTPException(404, f"Model '{req.model_id}' not found")

    feature_names = entry.get("feature_names")
    if feature_names and len(req.features) != len(feature_names):
        raise HTTPException(
            422,
            f"Expected {len(feature_names)} features, got {len(req.features)}",
        )

    model = entry["model"]
    features = np.array(req.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    target_names = entry.get("target_names", [])
    prediction_label = (
        target_names[prediction]
        if target_names and prediction < len(target_names)
        else str(prediction)
    )

    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        probabilities = [float(p) for p in probs]

    return PredictResponse(
        model_id=req.model_id,
        prediction=prediction,
        prediction_label=prediction_label,
        probabilities=probabilities,
    )
