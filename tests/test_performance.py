import time
import numpy as np

from src.models import train_random_forest


def _make_large_dummy_data():
    """
    Create a synthetic but reasonably sized dataset
    to test training performance.
    """
    n_samples = 5000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    return X, y


def test_random_forest_training_is_fast_enough():
    """
    Performance test for a critical path: model training.

    The goal is not micro-optimisation, just to ensure
    training stays within a reasonable time budget.
    """
    X, y = _make_large_dummy_data()

    start = time.perf_counter()
    model = train_random_forest(X, y)
    elapsed = time.perf_counter() - start

    # Sanity checks
    assert model is not None

    # Adjust threshold if needed depending on your machine.
    assert elapsed < 5.0, f"Training took too long: {elapsed:.2f} seconds"