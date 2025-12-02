"""
Main script to compare ML models on Iris dataset.
"""

from src.data_loader import load_and_split
from src.models import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
)
from src.evaluation import evaluate_model


def main():
    print("=" * 60)
    print("Iris Classification: Model Comparison")
    print("=" * 60)

    # Load data
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"    Train size: {X_train.shape}")
    print(f"    Test size: {X_test.shape}")



    # Train models
    print("\n2. Training models...")
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    print("    ✓ All models trained")

    # Evaluate
    print("\n3. Evaluating models...")
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_acc = evaluate_model(knn_model, X_test, y_test, "KNN")
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Conclusion
    results = {
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Logistic Regression": lr_acc,
    }
    winner = max(results, key=results.get)
    print("\n" + "=" * 60)
    print(f"Winner: {winner} ({results[winner]:.3f} accuracy)")
    print("=" * 60)


if __name__ == "__main__":
    main()
