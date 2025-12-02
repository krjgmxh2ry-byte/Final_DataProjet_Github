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
