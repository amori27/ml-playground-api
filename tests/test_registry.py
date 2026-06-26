import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.registry import register, get, list_models, count


def test_register_and_get():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    model_id = register(
        model,
        "iris",
        "random_forest",
        0.95,
        {"random_state": 42},
        target_names=["setosa", "versicolor", "virginica"],
        feature_names=["a", "b"],
    )
    assert model_id is not None
    entry = get(model_id)
    assert entry["dataset_name"] == "iris"
    assert entry["model_type"] == "random_forest"
    assert entry["accuracy"] == 0.95
    assert entry["params"] == {"random_state": 42}
    assert entry["target_names"] == ["setosa", "versicolor", "virginica"]
    assert entry["feature_names"] == ["a", "b"]


def test_get_unknown():
    with pytest.raises(KeyError):
        get("nonexistent")


def test_list_and_count():
    before = count()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    register(model, "iris", "random_forest", 0.9, {})
    assert count() == before + 1
    models = list_models()
    assert len(models) == count()
    last = models[0]
    assert "model_id" in last
    assert last["dataset_name"] == "iris"
    assert last["model_type"] == "random_forest"
    assert "created_at" in last
