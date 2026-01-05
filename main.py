"""Main script to compare ML models on Yahoo Finance dataset."""

from os import mkdir
from pathlib import Path
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

RESULTS_PATH = Path("results")


def main() -> None:
    """Main workflow for training, evaluating, and comparing Logistic Regression and Random Forest
    classifiers on the final portfolio dataset. Produces multiple evaluation plots."""

    # 1) Data loading + split train/test
    X_train, X_test, y_train, y_test = load_and_split()
    # 2) Training + evaluation Logistic Regression
    (
        mkdir(RESULTS_PATH / "logistic_regression")
        if not (RESULTS_PATH / "logistic_regression").exists()
        else None
    )
    lr_out_path = Path(RESULTS_PATH / "logistic_regression")
    lr_model = train_logistic_regression(X_train, y_train)
    y_pred_lr, y_prob_lr = evaluate_logistic_regression(lr_model, X_test, y_test)
    fpr_lr, tpr_lr = plot_roc_curve(y_test, y_prob_lr, lr_out_path)
    plot_feature_importance(lr_model, X_train.columns.tolist(), lr_out_path)
    plot_confusion_matrix(y_test, y_pred_lr, lr_out_path)
    plot_precision_recall_curve(y_test, y_prob_lr, lr_out_path)
    probabilities_histogram(y_test, y_prob_lr, lr_out_path)

    # 3) Training + Random Forest Evaluation
    (
        mkdir(RESULTS_PATH / "random_forest")
        if not (RESULTS_PATH / "random_forest").exists()
        else None
    )
    rf_out_path = Path(RESULTS_PATH / "random_forest")
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf, y_prob_rf = evaluate_random_forest(rf_model, X_test, y_test)
    fpr_rf, tpr_rf = plot_roc_curve(y_test, y_prob_rf, rf_out_path)
    plot_confusion_matrix(y_test, y_pred_rf, rf_out_path)
    plot_precision_recall_curve(y_test, y_prob_rf, rf_out_path)
    probabilities_histogram(y_test, y_prob_rf, rf_out_path)

    # 4) Visual comparison of the two models
    (
        mkdir(RESULTS_PATH / "model_comparison")
        if not (RESULTS_PATH / "model_comparison").exists()
        else None
    )
    outpath = Path(RESULTS_PATH / "model_comparison")
    plot_roc_curve_comparison(fpr_lr, tpr_lr, fpr_rf, tpr_rf, outpath)


if __name__ == "__main__":
    main()
