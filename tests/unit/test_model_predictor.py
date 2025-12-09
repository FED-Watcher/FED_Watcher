"""Comprehensive unit tests for ModelPredictor"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from src.backtesting.model_predictor import ModelPredictor


class SimpleModel:
    """Simple pickleable model class for testing"""

    def __init__(self, num_features=15):
        # Create feature importances matching the number of features
        importances = np.random.random(num_features)
        importances = importances / importances.sum()  # Normalize to sum to 1
        self.feature_importances_ = importances

    def predict(self, X):
        return np.array([1, 0, 1][: len(X)])

    def predict_proba(self, X):
        return np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]][: len(X)])


class SimpleModelNoProba:
    """Simple model without predict_proba for testing"""

    def __init__(self):
        self.feature_importances_ = np.random.random(15)

    def predict(self, X):
        return np.array([1, 0, 1][: len(X)])


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file"""
    model_path = tmp_path / "test_model.pkl"
    model = SimpleModel()

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


class SimpleScaler:
    """Simple pickleable scaler class for testing"""

    def transform(self, X):
        return X * 0.1


@pytest.fixture
def temp_scaler_file(tmp_path):
    """Create a temporary scaler file"""
    scaler_path = tmp_path / "test_scaler.pkl"
    scaler = SimpleScaler()

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler_path


@pytest.fixture
def temp_metadata_file(tmp_path):
    """Create a temporary metadata file"""
    metadata_path = tmp_path / "test_metadata.json"
    metadata = {
        "features": ["feature1", "feature2", "feature3"],
        "inverse_mapping": {"0": -1, "1": 0, "2": 1},
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return metadata_path


@pytest.fixture
def temp_model_no_proba_file(tmp_path):
    """Create a temporary model file without predict_proba"""
    model_path = tmp_path / "test_model_no_proba.pkl"
    model = SimpleModelNoProba()

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


@pytest.fixture
def test_data():
    """Create test data for predictions"""
    data = pd.DataFrame(
        {
            "Close": [5000, 4950, 5100],
            "Volume": [1000000, 1100000, 1200000],
            "Volume_ratio_vs_5days": [1.05, 1.1, 0.95],
            "Announcement": [1, 0, 1],
            "hawkish_dovish_ratio": [1.2, 0.8, 1.5],
            "net_sentiment_score": [0.1, -0.2, 0.3],
            "negative_proportion": [0.2, 0.4, 0.1],
            "neutral_proportion": [0.5, 0.4, 0.5],
            "positive_proportion": [0.3, 0.2, 0.4],
            "VIX_Close": [15.0, 18.0, 14.0],
            "DXY_Close": [102.0, 103.0, 101.0],
            "US02Y_Yield": [4.5, 4.6, 4.4],
            "US10Y_Yield": [4.2, 4.3, 4.1],
            "Yield_Curve_10Y_2Y": [-0.3, -0.3, -0.3],
            "sentiment_label": [1, 0, 1],
        }
    )
    return data


class TestModelPredictorInitialization:
    """Test ModelPredictor initialization"""

    def test_init_binary_model(self, temp_model_file, capsys):
        """Test initialization with binary model"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        assert predictor.model_type == "binary"
        assert predictor.model is not None
        assert len(predictor.feature_names) == 15  # Binary has 15 features

        captured = capsys.readouterr()
        assert "Binary model loaded successfully" in captured.out

    def test_init_multiclass_model(
        self, temp_model_file, temp_scaler_file, temp_metadata_file, capsys
    ):
        """Test initialization with multi-class model"""
        predictor = ModelPredictor(
            str(temp_model_file),
            model_type="multi_class",
            scaler_path=str(temp_scaler_file),
            metadata_path=str(temp_metadata_file),
        )

        assert predictor.model_type == "multi_class"
        assert predictor.scaler is not None
        assert predictor.metadata is not None
        assert predictor.feature_names == ["feature1", "feature2", "feature3"]

        captured = capsys.readouterr()
        assert "Multi_class model loaded successfully" in captured.out
        assert "Feature scaler: Loaded" in captured.out

    def test_init_multiclass_without_metadata(self, temp_model_file, temp_scaler_file):
        """Test multi-class initialization without metadata"""
        predictor = ModelPredictor(
            str(temp_model_file), model_type="multi_class", scaler_path=str(temp_scaler_file)
        )

        assert len(predictor.feature_names) == 11  # Default multi-class features

    def test_init_invalid_model_type(self, temp_model_file):
        """Test initialization with invalid model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            predictor = ModelPredictor(str(temp_model_file), model_type="invalid")
            # Force feature setup to trigger error
            predictor._setup_features()


class TestModelPredictorLoadMethods:
    """Test loading methods"""

    def test_load_model(self, temp_model_file):
        """Test model loading"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")
        assert predictor.model is not None

    def test_load_scaler_exists(self, temp_model_file, temp_scaler_file):
        """Test scaler loading when file exists"""
        predictor = ModelPredictor(
            str(temp_model_file), model_type="binary", scaler_path=str(temp_scaler_file)
        )
        assert predictor.scaler is not None

    def test_load_scaler_not_exists(self, temp_model_file, tmp_path):
        """Test scaler loading when file doesn't exist"""
        fake_scaler_path = tmp_path / "nonexistent_scaler.pkl"
        predictor = ModelPredictor(
            str(temp_model_file), model_type="binary", scaler_path=str(fake_scaler_path)
        )
        assert predictor.scaler is None

    def test_load_metadata_exists(self, temp_model_file, temp_metadata_file):
        """Test metadata loading when file exists"""
        predictor = ModelPredictor(
            str(temp_model_file), model_type="binary", metadata_path=str(temp_metadata_file)
        )
        assert predictor.metadata is not None
        assert "features" in predictor.metadata

    def test_load_metadata_not_exists(self, temp_model_file, tmp_path):
        """Test metadata loading when file doesn't exist"""
        fake_metadata_path = tmp_path / "nonexistent_metadata.json"
        predictor = ModelPredictor(
            str(temp_model_file), model_type="binary", metadata_path=str(fake_metadata_path)
        )
        assert predictor.metadata is None


class TestModelPredictorPreprocessing:
    """Test data preprocessing"""

    def test_preprocess_data_success(self, temp_model_file, test_data):
        """Test successful data preprocessing"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        X = predictor.preprocess_data(test_data)

        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(test_data)
        assert X.shape[1] == len(predictor.feature_names)

    def test_preprocess_data_missing_features(self, temp_model_file):
        """Test preprocessing with missing features"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        incomplete_data = pd.DataFrame({"Close": [5000], "Volume": [1000000]})

        with pytest.raises(ValueError, match="Missing required features"):
            predictor.preprocess_data(incomplete_data)

    def test_preprocess_data_missing_values(self, temp_model_file, test_data):
        """Test preprocessing with missing values"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        test_data_with_nan = test_data.copy()
        test_data_with_nan.loc[0, "Close"] = np.nan

        with pytest.raises(ValueError, match="Missing values detected"):
            predictor.preprocess_data(test_data_with_nan)

    def test_preprocess_data_with_scaler(
        self, temp_model_file, temp_scaler_file, temp_metadata_file
    ):
        """Test preprocessing with scaler"""
        predictor = ModelPredictor(
            str(temp_model_file),
            model_type="multi_class",
            scaler_path=str(temp_scaler_file),
            metadata_path=str(temp_metadata_file),
        )

        test_data = pd.DataFrame(
            {"feature1": [1.0, 2.0], "feature2": [3.0, 4.0], "feature3": [5.0, 6.0]}
        )

        X = predictor.preprocess_data(test_data)

        # Scaler was applied (values should be scaled)
        assert isinstance(X, np.ndarray)

    def test_preprocess_data_without_scaler(self, temp_model_file, test_data):
        """Test preprocessing without scaler"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        X = predictor.preprocess_data(test_data)

        assert isinstance(X, np.ndarray)


class TestModelPredictorPredictions:
    """Test prediction methods"""

    def test_predict_binary(self, temp_model_file, test_data):
        """Test binary predictions"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        predictions = predictor.predict(test_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(test_data)

    def test_predict_multiclass_with_mapping(
        self, temp_model_file, temp_scaler_file, temp_metadata_file
    ):
        """Test multi-class predictions with inverse mapping"""
        predictor = ModelPredictor(
            str(temp_model_file),
            model_type="multi_class",
            scaler_path=str(temp_scaler_file),
            metadata_path=str(temp_metadata_file),
        )

        test_data = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0], "feature2": [3.0, 4.0, 5.0], "feature3": [5.0, 6.0, 7.0]}
        )

        predictions = predictor.predict(test_data)

        # Predictions should be made
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

    def test_predict_multiclass_without_mapping(self, temp_model_file, temp_scaler_file):
        """Test multi-class predictions without inverse mapping"""
        predictor = ModelPredictor(
            str(temp_model_file), model_type="multi_class", scaler_path=str(temp_scaler_file)
        )

        test_data = pd.DataFrame(
            {
                "hawkish_dovish_ratio": [1.2, 0.8],
                "net_sentiment_score": [0.1, -0.2],
                "negative_proportion": [0.2, 0.4],
                "neutral_proportion": [0.5, 0.4],
                "positive_proportion": [0.3, 0.2],
                "VIX_Close": [15.0, 18.0],
                "DXY_Close": [102.0, 103.0],
                "US02Y_Yield": [4.5, 4.6],
                "US10Y_Yield": [4.2, 4.3],
                "Yield_Curve_10Y_2Y": [-0.3, -0.3],
                "Volume_ratio_vs_5days": [1.05, 1.1],
            }
        )

        predictions = predictor.predict(test_data)

        assert isinstance(predictions, np.ndarray)

    def test_predict_proba_success(self, temp_model_file, test_data):
        """Test probability predictions"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        probabilities = predictor.predict_proba(test_data)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(test_data)

    def test_predict_proba_not_supported(self, temp_model_no_proba_file, test_data):
        """Test probability predictions when not supported"""
        predictor = ModelPredictor(str(temp_model_no_proba_file), model_type="binary")

        with pytest.raises(AttributeError, match="does not support probability"):
            predictor.predict_proba(test_data)


class TestModelPredictorFeatureImportance:
    """Test feature importance methods"""

    def test_get_feature_importance_success(self, temp_model_file):
        """Test getting feature importance"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        importance_df = predictor.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 15  # Binary model has 15 features

    def test_get_feature_importance_not_supported(self, temp_model_file):
        """Test feature importance when not supported"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")
        delattr(predictor.model, "feature_importances_")

        with pytest.raises(AttributeError, match="does not provide feature importances"):
            predictor.get_feature_importance()


class TestModelPredictorLabeling:
    """Test label conversion methods"""

    def test_get_prediction_label_binary_up(self, temp_model_file):
        """Test binary label for up prediction"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        label = predictor.get_prediction_label(1)

        assert label == "Up"

    def test_get_prediction_label_binary_down(self, temp_model_file):
        """Test binary label for down prediction"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        label = predictor.get_prediction_label(0)

        assert label == "Down"

    def test_get_prediction_label_multiclass(self, temp_model_file):
        """Test multi-class labels"""
        predictor = ModelPredictor(str(temp_model_file), model_type="multi_class")

        assert predictor.get_prediction_label(-2) == "Strong Drop"
        assert predictor.get_prediction_label(-1) == "Modest Drop"
        assert predictor.get_prediction_label(0) == "Neutral"
        assert predictor.get_prediction_label(1) == "Modest Rise"
        assert predictor.get_prediction_label(2) == "Strong Rise"

    def test_get_prediction_label_unknown(self, temp_model_file):
        """Test unknown prediction label"""
        predictor = ModelPredictor(str(temp_model_file), model_type="multi_class")

        label = predictor.get_prediction_label(99)

        assert label == "Unknown"


class TestModelPredictorSignals:
    """Test trading signal conversion"""

    def test_get_signal_binary_long(self, temp_model_file):
        """Test binary signal for long"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        signal = predictor.get_signal_from_prediction(1)

        assert signal == "long"

    def test_get_signal_binary_short(self, temp_model_file):
        """Test binary signal for short"""
        predictor = ModelPredictor(str(temp_model_file), model_type="binary")

        signal = predictor.get_signal_from_prediction(0)

        assert signal == "short"

    def test_get_signal_multiclass_long(self, temp_model_file):
        """Test multi-class signal for long"""
        predictor = ModelPredictor(str(temp_model_file), model_type="multi_class")

        assert predictor.get_signal_from_prediction(2) == "long"
        assert predictor.get_signal_from_prediction(1) == "long"

    def test_get_signal_multiclass_short(self, temp_model_file):
        """Test multi-class signal for short"""
        predictor = ModelPredictor(str(temp_model_file), model_type="multi_class")

        assert predictor.get_signal_from_prediction(-2) == "short"
        assert predictor.get_signal_from_prediction(-1) == "short"

    def test_get_signal_multiclass_neutral(self, temp_model_file):
        """Test multi-class signal for neutral"""
        predictor = ModelPredictor(str(temp_model_file), model_type="multi_class")

        signal = predictor.get_signal_from_prediction(0)

        assert signal == "neutral"


pytestmark = pytest.mark.unit
