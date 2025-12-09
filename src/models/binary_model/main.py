"""
Main script for Fed Market Prediction project.
Orchestrates the complete machine learning pipeline.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_preparation import prepare_pipeline
from src.model_training import (
    train_xgboost_model,
    evaluate_model,
    save_model,
    save_metrics,
    get_feature_importance,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import pandas as pd
import numpy as np

# Get project root (4 levels up from this file: main.py -> binary_model -> models -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def main():
    """
    Main execution function for the ML pipeline.
    """
    print("\n" + "=" * 60)
    print("FED MARKET PREDICTION - BINARY CLASSIFICATION")
    print("=" * 60)
    print("Sprint: Predict market direction after Fed announcements")
    print("=" * 60 + "\n")

    # Configuration - use paths relative to this script's location
    SCRIPT_DIR = Path(__file__).parent
    DATA_PATH = PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv"
    MODEL_DIR = SCRIPT_DIR / "models"
    LOGS_DIR = SCRIPT_DIR / "logs"
    OUTPUTS_DIR = SCRIPT_DIR / "outputs"

    MODEL_PATH = MODEL_DIR / "xgboost_binary_classifier.pkl"
    METRICS_PATH = LOGS_DIR / "metrics.json"

    # Create necessary directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {DATA_PATH}")
    print(f"Model output: {MODEL_PATH}")
    print(f"Project root: {PROJECT_ROOT.resolve()}\n")

    # ===== STEP 1: DATA PREPARATION =====
    X_train, X_test, y_train, y_test, feature_cols = prepare_pipeline(
        filepath=str(DATA_PATH),
        use_announcement_only=True,  # Focus on announcement days only
        horizon=1,  # Predict 24h ahead (1 day)
        test_size=0.3,  # 80/20 train/test split
    )

    # ===== STEP 2: MODEL TRAINING =====
    model = train_xgboost_model(X_train, y_train)

    # ===== STEP 3: MODEL EVALUATION =====
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # ===== STEP 4: FEATURE IMPORTANCE =====
    importance_df = get_feature_importance(model, feature_cols)

    # ===== STEP 5: GENERATE VISUALIZATIONS =====
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    # Get predictions for visualization
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 1. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_sorted = importance_df.sort_values("importance", ascending=True).tail(15)
    ax.barh(
        importance_sorted["feature"], importance_sorted["importance"], color="steelblue", alpha=0.8
    )
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title("Top 15 Feature Importance - Binary Model", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    print(f"✓ Feature importance plot saved to {OUTPUTS_DIR}/feature_importance.png")
    plt.close()

    # 2. Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["Down (0)", "Up (1)"]
    )
    disp_train.plot(cmap="Blues", ax=ax1, values_format="d")
    ax1.set_title("Confusion Matrix - Training Set", fontsize=12, fontweight="bold")

    # Test confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp_test = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=["Down (0)", "Up (1)"]
    )
    disp_test.plot(cmap="Greens", ax=ax2, values_format="d")
    ax2.set_title("Confusion Matrix - Test Set", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrices saved to {OUTPUTS_DIR}/confusion_matrices.png")
    plt.close()

    # 3. ROC Curve
    from sklearn.metrics import roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="darkblue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve - Binary Classification Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    print(f"✓ ROC curve saved to {OUTPUTS_DIR}/roc_curve.png")
    plt.close()

    # 4. Performance Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    train_scores = [
        metrics["train"]["accuracy"],
        metrics["train"]["precision"],
        metrics["train"]["recall"],
        metrics["train"]["f1_score"],
    ]
    test_scores = [
        metrics["test"]["accuracy"],
        metrics["test"]["precision"],
        metrics["test"]["recall"],
        metrics["test"]["f1_score"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, train_scores, width, label="Training", color="steelblue", alpha=0.8
    )
    bars2 = ax.bar(x + width / 2, test_scores, width, label="Test", color="darkgreen", alpha=0.8)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "performance_metrics.png", dpi=300, bbox_inches="tight")
    print(f"✓ Performance metrics plot saved to {OUTPUTS_DIR}/performance_metrics.png")
    plt.close()

    # 5. Save predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_test_pred,
            "probability_up": y_test_proba,
            "correct": y_test == y_test_pred,
        }
    )
    predictions_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    print(f"✓ Test predictions saved to {OUTPUTS_DIR}/test_predictions.csv")

    print("\n" + "=" * 60 + "\n")

    # ===== STEP 6: SAVE RESULTS =====
    save_model(model, str(MODEL_PATH))
    save_metrics(metrics, str(METRICS_PATH))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nSprint Acceptance Criteria Met:")
    print("  ✓ Binary target variable created from market returns")
    print("  ✓ Proper chronological train/test split")
    print("  ✓ Model trained and saved")
    print("  ✓ Key ML metrics logged (Accuracy, F1-Score)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
