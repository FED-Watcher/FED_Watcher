from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def train_model(X_train, y_train, sample_weights, params):
    """Trains Gradient Boosting Classifier."""

    # Ensure params are valid for GBC
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate_model(model, X_test, y_test):
    """Calculates metrics for multi-class classification."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        # Weighted F1 is better for multi-class imbalance
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, y_pred
