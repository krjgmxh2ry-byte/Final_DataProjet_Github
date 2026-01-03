import pandas as pd
import numpy as np

from src.models import train_random_forest
from src.evaluation import evaluate_random_forest


def test_evaluate_random_forest_returns_predictions():
    # Fake dataset
    X = pd.DataFrame({
        "a": np.random.randn(120),
        "b": np.random.randn(120)
    })
    y = (X["a"] + X["b"] > 0).astype(int)

    model = train_random_forest(X, y)

    y_pred, y_prob = evaluate_random_forest(model, X, y)

    # Same length as input
    assert len(y_pred) == len(y)
    assert len(y_prob) == len(y)

    # Probabilities should be between 0 and 1
    assert (y_prob >= 0).all()
    assert (y_prob <= 1).all()
    