import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models import train_logistic_regression
from src.evaluation import plot_feature_importance


def test_plot_feature_importance_runs_without_error():
    """
    Check that feature importance plotting works for logistic regression.
    """

    # Simple synthetic dataset
    X = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
    })
    y = (X["a"] + X["b"] > 0).astype(int)

    model = train_logistic_regression(X, y)

    # Should run without crashing
    plot_feature_importance(model, ["a", "b"])

    # Close plot windows
    plt.close("all")