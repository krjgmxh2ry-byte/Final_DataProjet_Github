"""Model definitions and training."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    """Train K-Nearest Neighbors model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        max_iter=200,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
