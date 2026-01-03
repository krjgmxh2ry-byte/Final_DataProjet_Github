import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import plot_confusion_matrix


def test_plot_confusion_matrix_runs_without_error():
    """
    Basic smoke test for the confusion matrix plotting function.

    Goal:
    - Call plot_confusion_matrix with simple binary labels.
    - Check that the function runs without raising any exceptions.
    """

    # Fake binary ground-truth and predictions
    y_test = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    # Call the plotting function
    plot_confusion_matrix(y_test, y_pred)

    # Close any open figures so tests do not hang
    plt.close("all")