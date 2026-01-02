import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ⚠️ IMPORTANT :
# adapte ces imports si les fonctions dans src/models.py ont des noms différents
from src.models import (
    train_logistic_regression,
    train_random_forest,
)


def _make_dummy_data():
    """
    Small synthetic binary dataset just for testing the training pipeline.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_logistic_regression_trains_and_predicts():
    X_train, X_test, y_train, y_test = _make_dummy_data()

    model = train_logistic_regression(X_train, y_train)

    # Model object is not None and has predict()
    assert model is not None
    assert hasattr(model, "predict")

    # Can predict on test set
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)

    # Sanity check: better than random guessing
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.5


def test_random_forest_trains_and_predicts():
    X_train, X_test, y_train, y_test = _make_dummy_data()

    model = train_random_forest(X_train, y_train)

    # Model object is not None and has predict()
    assert model is not None
    assert hasattr(model, "predict")

    # Can predict on test set
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)

    # Sanity check: better than random guessing
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.5