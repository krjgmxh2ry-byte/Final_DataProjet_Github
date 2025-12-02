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
