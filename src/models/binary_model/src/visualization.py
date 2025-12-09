import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import pandas as pd
import numpy as np
from pathlib import Path


def generate_plots(model, X_train, y_train, X_test, y_test, importance_df, output_dir):
    """Generates and saves all model visualizations to the output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 1. Feature Importance
    plt.figure(figsize=(10, 8))
    importance_sorted = importance_df.sort_values("importance", ascending=True).tail(15)
    plt.barh(importance_sorted["feature"], importance_sorted["importance"], color="steelblue")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    # 2. Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cm_train = confusion_matrix(y_train, y_train_pred)
    ConfusionMatrixDisplay(cm_train, display_labels=["Down", "Up"]).plot(
        cmap="Blues", ax=ax1, values_format="d"
    )
    ax1.set_title("Train Confusion Matrix")

    cm_test = confusion_matrix(y_test, y_test_pred)
    ConfusionMatrixDisplay(cm_test, display_labels=["Down", "Up"]).plot(
        cmap="Greens", ax=ax2, values_format="d"
    )
    ax2.set_title("Test Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png")
    plt.close()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()
