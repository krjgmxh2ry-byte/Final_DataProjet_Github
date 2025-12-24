"""Main script to compare ML models on Yahoo Finance dataset."""

from src.data_loader import (
    load_and_split,
)
from src.evaluation import (
    evaluate_logistic_regression,
    evaluate_random_forest,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_roc_curve_comparison,
    probabilities_histogram,
)
from src.models import train_logistic_regression, train_random_forest


def main() -> None:
    """Main workflow for training, evaluating, and comparing Logistic Regression and Random Forest
    classifiers on the final portfolio dataset. Produces multiple evaluation plots."""

    # Logistic Regression
    X_train, X_test, y_train, y_test, df = load_and_split()
    lr_model = train_logistic_regression(X_train, y_train)
    y_pred_lr, y_prob_lr = evaluate_logistic_regression(lr_model, X_test, y_test)
    fpr_lr, tpr_lr = plot_roc_curve(y_test, y_prob_lr)
    plot_feature_importance(lr_model, X_train.columns.tolist())
    plot_confusion_matrix(y_test, y_pred_lr)
    plot_precision_recall_curve(y_test, y_prob_lr)
    probabilities_histogram(y_test, y_prob_lr)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf, y_prob_rf = evaluate_random_forest(rf_model, X_test, y_test)
    fpr_rf, tpr_rf = plot_roc_curve(y_test, y_prob_rf)
    plot_confusion_matrix(y_test, y_pred_rf)
    plot_precision_recall_curve(y_test, y_prob_rf)
    probabilities_histogram(y_test, y_prob_rf)

    # comaprison
    plot_roc_curve_comparison(fpr_lr, tpr_lr, fpr_rf, tpr_rf)


if __name__ == "__main__":
    main()
