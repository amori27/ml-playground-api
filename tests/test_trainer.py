import pytest
from src.core.trainer import train_model, MODELS


def test_all_model_types_present():
    assert set(MODELS.keys()) == {"random_forest", "svm", "logistic_regression"}


def test_train_random_forest():
    model, accuracy, target_names, feature_names, params = train_model(
        "iris", "random_forest", {"random_state": 42}
    )
    assert accuracy > 0.5
    assert len(target_names) == 3
    assert len(feature_names) == 4
    assert params == {"random_state": 42}


def test_train_svm():
    model, accuracy, target_names, feature_names, params = train_model(
        "iris", "svm", {"kernel": "linear"}
    )
    assert accuracy > 0.5
    assert hasattr(model, "predict_proba")


def test_train_logistic_regression():
    model, accuracy, target_names, feature_names, params = train_model(
        "wine", "logistic_regression", {"max_iter": 2000, "C": 1.0}
    )
    assert accuracy > 0.5


def test_unknown_model_type():
    with pytest.raises(ValueError):
        train_model("iris", "unknown")


def test_unknown_dataset():
    with pytest.raises(KeyError):
        train_model("unknown", "random_forest")


def test_train_on_all_datasets():
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        model, accuracy, target_names, feature_names, params = train_model(
            dataset_name, "random_forest", {"random_state": 42}
        )
        assert accuracy > 0.5, f"Failed on {dataset_name}"
