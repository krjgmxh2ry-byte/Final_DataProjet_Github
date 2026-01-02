"""Model evaluation and visualization."""

import numpy as np
import pandas as pd

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Plotting tools
import matplotlib.pyplot as plt

# Confusion matrix utilities
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def evaluate_logistic_regression(
    model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model performance: print accuracy and classification report."""
    y_pred = model.predict(X_test)
    # Probability of belonging to class 1 (used for curves ROC/PR).
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return y_pred, y_prob


def evaluate_random_forest(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate RF performance, returns predictions and probabilities.
    """
    y_pred = model.predict(X_test)
    # Probability of belonging to class 1 (used for ROC/PR curves).
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_pred, y_prob


def plot_roc_curve(
    y_test: pd.Series, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Plot ROC curve for binary classification."""
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running
    return fpr, tpr


def plot_feature_importance(model: LogisticRegression, feature_names: list) -> None:
    """Plot the coefficients of the logistic regression model."""
    # Coefficients > 0 favour class 1, < 0 favour class 0.
    coefs = pd.Series(model.coef_[0], index=feature_names)
    coefs = coefs.sort_values()

    plt.figure(figsize=(8, 6))
    coefs.plot(kind="barh", color="skyblue")
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.grid(alpha=0.3)
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running


def plot_precision_recall_curve(y_test, y_prob) -> None:
    """Plot the precision-recall curve to evaluate classifier performance across thresholds."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running


def probabilities_histogram(y_test, y_prob) -> None:
    """Plot a histogram of predicted probabilities for each true class to visualize model confidence."""
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.5, label="True 1")
    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.5, label="True 0")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.legend()
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running


def plot_roc_curve_comparison(fpr1, tpr1, fpr2, tpr2) -> None:
    """Visually compare two ROCs (AUC shown in the legend)."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f"Logistic Regression (AUC={auc(fpr1, tpr1):.2f})")
    plt.plot(fpr2, tpr2, label=f"Random Forest (AUC={auc(fpr2, tpr2):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()
    #plt.close()   # Close the plot window after showing it so the script can continue running



  
