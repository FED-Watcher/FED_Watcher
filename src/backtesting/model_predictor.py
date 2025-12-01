"""
Unified prediction interface for FED Watcher models.
Loads and runs predictions from both binary and multi-class models.
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path


class ModelPredictor:
    """
    Unified interface for making predictions with trained models.
    Supports both binary and multi-class classification models.
    """

    def __init__(self, model_path, model_type='binary', scaler_path=None, metadata_path=None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path (str): Path to the saved model (.pkl file)
            model_type (str): Type of model - 'binary' or 'multi_class'
            scaler_path (str): Path to feature scaler (required for multi-class)
            metadata_path (str): Path to model metadata JSON (optional)
        """
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None

        # Load model
        self.model = self._load_model()

        # Load scaler if provided
        self.scaler = self._load_scaler() if self.scaler_path else None

        # Load metadata if provided
        self.metadata = self._load_metadata() if self.metadata_path else None

        # Set up feature information
        self._setup_features()

        print(f"[OK] {self.model_type.capitalize()} model loaded successfully")
        print(f"  Model path: {self.model_path}")
        print(f"  Expected features: {len(self.feature_names)}")
        if self.scaler:
            print(f"  Feature scaler: Loaded")

    def _load_model(self):
        """Load the trained model from disk."""
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _load_scaler(self):
        """Load the feature scaler from disk."""
        if not self.scaler_path or not self.scaler_path.exists():
            return None
        with open(self.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def _load_metadata(self):
        """Load model metadata from JSON file."""
        if not self.metadata_path or not self.metadata_path.exists():
            return None
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def _setup_features(self):
        """Set up expected feature names based on model type and metadata."""
        if self.metadata and 'features' in self.metadata:
            self.feature_names = self.metadata['features']
        elif self.model_type == 'binary':
            # Default binary model features (from data_preparation.py)
            self.feature_names = [
                'Close', 'Volume', 'Volume_ratio_vs_5days', 'Announcement',
                'hawkish_dovish_ratio', 'net_sentiment_score',
                'negative_proportion', 'neutral_proportion', 'positive_proportion',
                'VIX_Close', 'DXY_Close', 'US02Y_Yield', 'US10Y_Yield',
                'Yield_Curve_10Y_2Y', 'sentiment_label'
            ]
        elif self.model_type == 'multi_class':
            # Default multi-class model features
            self.feature_names = [
                'hawkish_dovish_ratio', 'net_sentiment_score',
                'negative_proportion', 'neutral_proportion', 'positive_proportion',
                'VIX_Close', 'DXY_Close', 'US02Y_Yield', 'US10Y_Yield',
                'Yield_Curve_10Y_2Y', 'Volume_ratio_vs_5days'
            ]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def preprocess_data(self, data):
        """
        Preprocess input data to match training format.

        Args:
            data (pd.DataFrame): Input data with required features

        Returns:
            np.ndarray: Preprocessed feature matrix ready for prediction
        """
        # Select required features
        try:
            X = data[self.feature_names].copy()
        except KeyError as e:
            missing_features = set(self.feature_names) - set(data.columns)
            raise ValueError(f"Missing required features: {missing_features}") from e

        # Check for missing values
        if X.isnull().any().any():
            missing_info = X.isnull().sum()
            missing_info = missing_info[missing_info > 0]
            raise ValueError(f"Missing values detected in features:\n{missing_info}")

        # Apply scaling if scaler is available
        if self.scaler:
            X_scaled = self.scaler.transform(X.values)
            return X_scaled

        return X.values

    def predict(self, data):
        """
        Make predictions on new data.

        Args:
            data (pd.DataFrame): Input data with required features

        Returns:
            np.ndarray: Predictions
                - Binary model: 0 (Down) or 1 (Up)
                - Multi-class model: -2, -1, 0, 1, 2 (Strong Drop to Strong Rise)
        """
        X = self.preprocess_data(data)
        predictions = self.model.predict(X)

        # For multi-class models, convert from mapped classes back to original
        if self.model_type == 'multi_class' and self.metadata:
            inverse_mapping = self.metadata.get('inverse_mapping', {})
            if inverse_mapping:
                # Convert string keys to int
                inverse_mapping = {int(k): v for k, v in inverse_mapping.items()}
                predictions = np.array([inverse_mapping[p] for p in predictions])

        return predictions

    def predict_proba(self, data):
        """
        Get prediction probabilities.

        Args:
            data (pd.DataFrame): Input data with required features

        Returns:
            np.ndarray: Probability distributions for each class
        """
        X = self.preprocess_data(data)

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return probabilities
        else:
            raise AttributeError(f"Model does not support probability predictions")

    def get_feature_importance(self):
        """
        Get feature importance from the model.

        Returns:
            pd.DataFrame: DataFrame with features and their importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            raise AttributeError(f"Model does not provide feature importances")

    def get_prediction_label(self, prediction):
        """
        Convert numeric prediction to human-readable label.

        Args:
            prediction (int): Numeric prediction

        Returns:
            str: Human-readable label
        """
        if self.model_type == 'binary':
            return 'Up' if prediction == 1 else 'Down'
        elif self.model_type == 'multi_class':
            labels = {
                -2: 'Strong Drop',
                -1: 'Modest Drop',
                0: 'Neutral',
                1: 'Modest Rise',
                2: 'Strong Rise'
            }
            return labels.get(prediction, 'Unknown')
        else:
            return str(prediction)

    def get_signal_from_prediction(self, prediction):
        """
        Convert prediction to trading signal.

        Args:
            prediction (int): Model prediction

        Returns:
            str: Trading signal ('long', 'short', or 'neutral')
        """
        if self.model_type == 'binary':
            # Binary: 1 = Up (Long), 0 = Down (Short)
            return 'long' if prediction == 1 else 'short'
        elif self.model_type == 'multi_class':
            # Multi-class: positive = long, negative = short, 0 = neutral
            if prediction > 0:
                return 'long'
            elif prediction < 0:
                return 'short'
            else:
                return 'neutral'
        else:
            return 'neutral'