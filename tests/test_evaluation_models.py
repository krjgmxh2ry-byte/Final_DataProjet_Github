import pandas as pd
import numpy as np

from src.models import train_logistic_regression
from src.evaluation import evaluate_logistic_regression


def test_evaluate_logistic_regression_returns_predictions():
    # Fake dataset
    X = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
    })
    y = (X["a"] + X["b"] > 0).astype(int)

    model = train_logistic_regression(X, y)

    y_pred, y_prob = evaluate_logistic_regression(model, X, y)

    # shapes must match input length
    assert len(y_pred) == len(y)
    assert len(y_prob) == len(y)

    # probabilities must be between 0 and 1
    assert (y_prob >= 0).all()
    assert (y_prob <= 1).all()