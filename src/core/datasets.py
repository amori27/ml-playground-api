from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def get_dataset_info(name: str) -> dict:
    loader = DATASETS[name]
    data = loader()
    return {
        "name": name,
        "description": data.DESCR.split("\n")[0] if hasattr(data, "DESCR") else "",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(data.target_names),
        "feature_names": list(data.feature_names),
        "target_names": [str(t) for t in data.target_names],
    }


def load_dataset(name: str):
    loader = DATASETS[name]
    data = loader()
    return data.data, data.target, [str(t) for t in data.target_names], list(data.feature_names)
