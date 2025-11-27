"""
Main script for Fed Market Prediction project.
Orchestrates the complete machine learning pipeline.
"""

from src.data_preparation import prepare_pipeline
from src.model_training import (
    train_xgboost_model,
    evaluate_model,
    save_model,
    save_metrics,
    get_feature_importance,
)
from sklearn.model_selection import train_test_split


def main():
    """
    Main execution function for the ML pipeline.
    """
    print("\n" + "=" * 60)
    print("FED MARKET PREDICTION - BINARY CLASSIFICATION")
    print("=" * 60)
    print("Sprint: Predict market direction after Fed announcements")
    print("=" * 60 + "\n")

    # Configuration
    DATA_PATH = "MasterDataset_Enriched.csv"
    MODEL_PATH = "models/xgboost_binary_classifier.pkl"
    METRICS_PATH = "logs/metrics.json"

    # ===== STEP 1: DATA PREPARATION =====
    X_train, X_test, y_train, y_test, feature_cols = prepare_pipeline(
        filepath=DATA_PATH,
        use_announcement_only=True,  # Focus on announcement days only
        horizon=1,  # Predict 24h ahead (1 day)
        test_size=0.2,  # 80/20 train/test split
    )

    # ===== STEP 2: MODEL TRAINING =====
    model = train_xgboost_model(X_train, y_train)

    # ===== STEP 3: MODEL EVALUATION =====
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # ===== STEP 4: FEATURE IMPORTANCE =====
    importance_df = get_feature_importance(model, feature_cols)

    # ===== STEP 5: SAVE RESULTS =====
    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)

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
