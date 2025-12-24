"""Model definitions and training."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, **kwargs):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, **kwargs
) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model
