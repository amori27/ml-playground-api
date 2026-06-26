from src.core.datasets import DATASETS, get_dataset_info, load_dataset


def test_all_datasets_present():
    assert set(DATASETS.keys()) == {"iris", "wine", "breast_cancer", "digits"}


def test_get_dataset_info():
    info = get_dataset_info("iris")
    assert info["name"] == "iris"
    assert info["n_samples"] > 0
    assert info["n_features"] > 0
    assert info["n_classes"] > 0
    assert len(info["feature_names"]) == info["n_features"]
    assert len(info["target_names"]) == info["n_classes"]


def test_get_dataset_info_all():
    for name in DATASETS:
        info = get_dataset_info(name)
        assert info["name"] == name
        assert info["n_samples"] > 0


def test_load_dataset():
    X, y, target_names, feature_names = load_dataset("iris")
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert len(target_names) == 3
    assert len(feature_names) == 4


def test_load_dataset_wine():
    X, y, target_names, feature_names = load_dataset("wine")
    assert X.shape[0] == 178
    assert X.shape[1] == 13


def test_load_dataset_breast_cancer():
    X, y, target_names, feature_names = load_dataset("breast_cancer")
    assert X.shape[0] == 569
    assert X.shape[1] == 30


def test_load_dataset_digits():
    X, y, target_names, feature_names = load_dataset("digits")
    assert X.shape[0] == 1797
    assert X.shape[1] == 64
    assert len(target_names) == 10
