import uuid
from datetime import datetime, timezone


_registry: dict[str, dict] = {}


def register(
    model,
    dataset_name: str,
    model_type: str,
    accuracy: float,
    params: dict,
    target_names: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> str:
    model_id = uuid.uuid4().hex[:8]
    _registry[model_id] = {
        "model": model,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "params": params,
        "target_names": target_names,
        "feature_names": feature_names,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return model_id


def get(model_id: str) -> dict:
    if model_id not in _registry:
        raise KeyError(f"Model '{model_id}' not found")
    return _registry[model_id]


def list_models() -> list[dict]:
    return [
        {
            "model_id": k,
            "dataset_name": v["dataset_name"],
            "model_type": v["model_type"],
            "accuracy": v["accuracy"],
            "params": v["params"],
            "created_at": v["created_at"],
        }
        for k, v in _registry.items()
    ]


def count() -> int:
    return len(_registry)
