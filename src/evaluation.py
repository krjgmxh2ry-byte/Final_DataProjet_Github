"""Model evaluation and visualization."""
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print results."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy
