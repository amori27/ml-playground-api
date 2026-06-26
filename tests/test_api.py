from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_available"] == 4
    assert isinstance(data["trained_count"], int)


def test_datasets():
    resp = client.get("/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    names = [d["name"] for d in data]
    assert "iris" in names
    assert "wine" in names
    assert "breast_cancer" in names
    assert "digits" in names
    for ds in data:
        assert ds["n_samples"] > 0
        assert ds["n_features"] > 0


def test_train_and_predict():
    train_resp = client.post(
        "/train",
        json={
            "dataset_name": "iris",
            "model_type": "random_forest",
            "params": {"random_state": 42},
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert "model_id" in train_data
    assert train_data["accuracy"] > 0.5
    assert train_data["dataset_name"] == "iris"
    assert train_data["model_type"] == "random_forest"

    model_id = train_data["model_id"]

    predict_resp = client.post(
        "/predict",
        json={
            "model_id": model_id,
            "features": [5.1, 3.5, 1.4, 0.2],
        },
    )
    assert predict_resp.status_code == 200
    pred_data = predict_resp.json()
    assert pred_data["prediction"] in (0, 1, 2)
    assert isinstance(pred_data["prediction_label"], str)
    assert isinstance(pred_data["probabilities"], list)


def test_models_list():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_model_detail():
    train_resp = client.post(
        "/train",
        json={
            "dataset_name": "iris",
            "model_type": "logistic_regression",
            "params": {"max_iter": 2000},
        },
    )
    model_id = train_resp.json()["model_id"]
    resp = client.get(f"/models/{model_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == model_id
    assert data["model_type"] == "logistic_regression"


def test_model_detail_not_found():
    resp = client.get("/models/nonexistent")
    assert resp.status_code == 404


def test_train_unknown_dataset():
    resp = client.post(
        "/train",
        json={
            "dataset_name": "unknown_dataset",
            "model_type": "random_forest",
        },
    )
    assert resp.status_code == 404


def test_train_unknown_model_type():
    resp = client.post(
        "/train",
        json={
            "dataset_name": "iris",
            "model_type": "unknown_model",
        },
    )
    assert resp.status_code == 404


def test_predict_unknown_model():
    resp = client.post(
        "/predict",
        json={
            "model_id": "nonexistent",
            "features": [1.0, 2.0, 3.0, 4.0],
        },
    )
    assert resp.status_code == 404


def test_predict_wrong_feature_count():
    train_resp = client.post(
        "/train",
        json={
            "dataset_name": "iris",
            "model_type": "random_forest",
            "params": {"random_state": 42},
        },
    )
    model_id = train_resp.json()["model_id"]
    resp = client.post(
        "/predict",
        json={
            "model_id": model_id,
            "features": [1.0, 2.0],
        },
    )
    assert resp.status_code == 422
