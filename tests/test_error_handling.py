import numpy as np
import pytest

from src.models import train_logistic_regression


def test_logistic_regression_raises_on_shape_mismatch():
    """
    Edge-case / error-handling test.

    If X and y have incompatible shapes, the training function
    should raise a ValueError (coming from scikit-learn).
    """
    # 10 samples, 3 features
    X = np.random.randn(10, 3)
    # Only 9 labels instead of 10
    y = np.random.randint(0, 2, size=9)

    with pytest.raises(ValueError):
        train_logistic_regression(X, y)