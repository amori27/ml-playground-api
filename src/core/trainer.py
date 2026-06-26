from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.core.datasets import load_dataset

MODELS = {
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "logistic_regression": LogisticRegression,
}


def train_model(dataset_name: str, model_type: str, params: dict | None = None):
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    X, y, target_names, feature_names = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    effective_params = dict(params) if params else {}

    if model_type == "svm":
        svm_params = {k: v for k, v in effective_params.items()}
        base = SVC(**svm_params)
        base.fit(X_train, y_train)
        model = CalibratedClassifierCV(base, cv=3, ensemble=False)
        model.fit(X_train, y_train)
    else:
        model_class = MODELS[model_type]
        model = model_class(**effective_params)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    return model, accuracy, target_names, feature_names, effective_params
