"""
Model training module for Fed Market Prediction project.
Trains XGBoost binary classifier to predict market direction.
"""

import xgboost as xgb
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


def train_xgboost_model(X_train, y_train, params=None):
    """
    Train XGBoost binary classification model.

    Args:
        X_train: Training features
        y_train: Training target
        params (dict): XGBoost parameters (optional)

    Returns:
        xgb.XGBClassifier: Trained model
    """
    # Default parameters if none provided
    if params is None:
        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42,
            "eval_metric": "logloss",
        }

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)
    print(f"Parameters: {params}")

    # Initialize and train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Model training complete")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train and test sets.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        dict: Dictionary containing all metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        },
    }

    # Print results
    print("\nTRAIN SET METRICS:")
    print(f"  Accuracy:  {metrics['train']['accuracy']:.4f}")
    print(f"  Precision: {metrics['train']['precision']:.4f}")
    print(f"  Recall:    {metrics['train']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['train']['f1_score']:.4f}")

    print("\nTEST SET METRICS:")
    print(f"  Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"  Precision: {metrics['test']['precision']:.4f}")
    print(f"  Recall:    {metrics['test']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['test']['f1_score']:.4f}")

    # Confusion matrix
    print("\nCONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print("              Predicted")
    print("              Down  Up")
    print(f"Actual Down    {cm[0][0]:3d}  {cm[0][1]:3d}")
    print(f"       Up      {cm[1][0]:3d}  {cm[1][1]:3d}")

    # Classification report
    print("\nCLASSIFICATION REPORT (Test Set):")
    print(classification_report(
        y_test, y_test_pred,
        target_names=['Down', 'Up'],
        zero_division=0
    ))
    
    return metrics


def save_model(model, filepath):
    """
    Save trained model to disk using pickle.

    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {filepath}")


def save_metrics(metrics, filepath):
    """
    Save metrics to JSON file for logging.

    Args:
        metrics (dict): Dictionary of metrics
        filepath (str): Path to save the metrics
    """
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {filepath}")


def load_model(filepath):
    """
    Load a saved model from disk.

    Args:
        filepath (str): Path to the saved model

    Returns:
        Trained model
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {filepath}")
    return model


def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained model.

    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names

    Returns:
        pd.DataFrame: DataFrame with features and their importance scores
    """
    import pandas as pd

    importance_dict = {"feature": feature_names, "importance": model.feature_importances_}

    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values("importance", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))

    return importance_df
