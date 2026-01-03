import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import (
    plot_precision_recall_curve,
    probabilities_histogram,
)


def test_precision_recall_and_histogram_run_without_error():
    """
    Smoke test for evaluation plots based on probabilities.

    Goal:
    - Use simple synthetic probabilities and labels.
    - Verify that precision–recall curve and probability histogram
      run without raising exceptions.
    """

    # Fake binary ground-truth labels
    y_test = np.array([0, 1, 0, 1, 1, 0, 0, 1])

    # Fake model probabilities between 0 and 1
    y_prob = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2, 0.4, 0.85])

    # Call plotting functions (should run without crashing)
    plot_precision_recall_curve(y_test, y_prob)
    probabilities_histogram(y_test, y_prob)

    # Close figures so tests do not hang
    plt.close("all")