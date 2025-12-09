import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


def generate_plots(model, X_test, y_test, feature_names, output_dir):
    """Generates Confusion Matrix and Feature Importance plots."""
    output_dir = Path(output_dir)
    y_pred = model.predict(X_test)

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    labels = [-2, -1, 0, 1, 2]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Classes: -2 to +2)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = importances.argsort()

    plt.barh(range(len(indices)), importances[indices], color="darkgreen", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()
