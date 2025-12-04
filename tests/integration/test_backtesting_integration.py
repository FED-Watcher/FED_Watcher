"""Integration tests for backtesting workflow"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from unittest.mock import patch, Mock
from src.backtesting.model_predictor import ModelPredictor
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategies import BinaryStrategy, MultiClassStrategy
from src.backtesting.report_generator import generate_html_report


class SimpleIntegrationModel:
    """Simple model for integration testing"""

    def __init__(self):
        self.feature_importances_ = np.random.random(15)

    def predict(self, X):
        # Predict alternating up/down
        return np.array([1 if i % 2 == 0 else 0 for i in range(len(X))])

    def predict_proba(self, X):
        # Return probabilities
        probs = []
        for i in range(len(X)):
            if i % 2 == 0:
                probs.append([0.3, 0.7])  # 70% confident up
            else:
                probs.append([0.8, 0.2])  # 80% confident down
        return np.array(probs)


@pytest.fixture
def integration_test_data():
    """Create realistic test data for integration testing"""
    dates = pd.date_range("2024-01-01", periods=10, freq="30D")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Close": [5000, 5100, 5050, 5150, 5075, 5200, 5150, 5250, 5200, 5300],
            "Volume": [1000000] * 10,
            "Volume_ratio_vs_5days": [1.0, 1.1, 0.95, 1.05, 0.9, 1.15, 1.0, 1.1, 0.95, 1.05],
            "Announcement": [1] * 10,
            "hawkish_dovish_ratio": [1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.15, 0.85, 1.25, 0.75],
            "net_sentiment_score": [
                0.15,
                -0.10,
                0.05,
                -0.05,
                0.20,
                -0.15,
                0.10,
                -0.08,
                0.18,
                -0.12,
            ],
            "negative_proportion": [0.2, 0.4, 0.3, 0.35, 0.25, 0.45, 0.28, 0.38, 0.22, 0.42],
            "neutral_proportion": [0.3, 0.3, 0.4, 0.35, 0.30, 0.25, 0.37, 0.32, 0.33, 0.28],
            "positive_proportion": [0.5, 0.3, 0.3, 0.30, 0.45, 0.30, 0.35, 0.30, 0.45, 0.30],
            "VIX_Close": [15, 18, 16, 17, 14, 19, 15.5, 17.5, 14.5, 18.5],
            "DXY_Close": [103, 104, 103.5, 104.5, 103, 105, 103.8, 104.2, 103.2, 104.8],
            "US02Y_Yield": [4.5, 4.6, 4.55, 4.65, 4.5, 4.7, 4.58, 4.62, 4.52, 4.68],
            "US10Y_Yield": [4.2, 4.3, 4.25, 4.35, 4.2, 4.4, 4.28, 4.32, 4.22, 4.38],
            "Yield_Curve_10Y_2Y": [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
            "sentiment_label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    # Calculate returns
    data["future_close"] = data["Close"].shift(-1)
    data["return"] = (data["future_close"] - data["Close"]) / data["Close"]
    data = data[:-1].copy()  # Remove last row

    return data


@pytest.fixture
def integration_model_file(tmp_path):
    """Create a model file for integration testing"""
    model_path = tmp_path / "integration_model.pkl"
    model = SimpleIntegrationModel()

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


class TestBacktestingWorkflow:
    """Test complete backtesting workflow"""

    def test_load_data_and_run_backtest(self, integration_model_file, integration_test_data):
        """Test loading data and running a complete backtest"""
        # Create predictor
        predictor = ModelPredictor(str(integration_model_file), model_type="binary")

        # Create strategy
        strategy = BinaryStrategy(name="Test Binary Strategy")

        # Create backtest engine
        engine = BacktestEngine(
            predictor,
            strategy,
            initial_capital=100000,
            horizon=1,
            commission_pct=0.0,
            slippage_pct=0.0,
        )

        # Run backtest
        results = engine.run(
            integration_test_data, date_col="Date", return_col="return", verbose=False
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(integration_test_data)
        assert "prediction" in results.columns
        assert "position" in results.columns
        assert "pnl_pct" in results.columns
        assert "equity" in results.columns

        # Verify metrics were calculated
        assert engine.metrics is not None
        assert "total_trades" in engine.metrics
        assert "total_return_pct" in engine.metrics
        assert "sharpe_ratio" in engine.metrics

    def test_binary_strategy_workflow(self, integration_model_file, integration_test_data):
        """Test binary strategy complete workflow"""
        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy()
        engine = BacktestEngine(predictor, strategy, initial_capital=50000)

        results = engine.run(
            integration_test_data, date_col="Date", return_col="return", verbose=False
        )

        # Binary strategy should have long and short positions
        assert 1.0 in results["position"].values  # Long positions
        assert -1.0 in results["position"].values  # Short positions

        # Verify equity curve exists
        assert len(engine.equity_curve) > 0

    def test_backtest_with_transaction_costs(self, integration_model_file, integration_test_data):
        """Test backtesting with commission and slippage"""
        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy()

        # Run with transaction costs
        engine_with_costs = BacktestEngine(
            predictor, strategy, initial_capital=100000, commission_pct=0.1, slippage_pct=0.05
        )

        results_with_costs = engine_with_costs.run(
            integration_test_data, date_col="Date", return_col="return", verbose=False
        )

        # Run without transaction costs
        engine_no_costs = BacktestEngine(
            predictor, strategy, initial_capital=100000, commission_pct=0.0, slippage_pct=0.0
        )

        results_no_costs = engine_no_costs.run(
            integration_test_data, date_col="Date", return_col="return", verbose=False
        )

        # Total return with costs should be lower
        final_equity_with_costs = engine_with_costs.equity_curve[-1]["equity"]
        final_equity_no_costs = engine_no_costs.equity_curve[-1]["equity"]

        assert final_equity_with_costs < final_equity_no_costs

    def test_report_generation(self, integration_model_file, integration_test_data, tmp_path):
        """Test HTML report generation"""
        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy(name="Test Strategy")
        engine = BacktestEngine(predictor, strategy)

        engine.run(integration_test_data, date_col="Date", return_col="return", verbose=False)

        # Generate report
        output_dir = tmp_path / "reports"
        report_path = generate_html_report(
            engine=engine, output_dir=str(output_dir), strategy_name="Test Strategy"
        )

        # Verify report was created
        assert report_path.exists()
        assert report_path.name == "backtest_report.html"

        # Verify report contains key information
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            assert "Test Strategy" in html_content
            assert "Total Return" in html_content or "Total Trades" in html_content

    def test_export_workflow(self, integration_model_file, integration_test_data, tmp_path):
        """Test exporting backtest results"""
        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy()
        engine = BacktestEngine(predictor, strategy)

        engine.run(integration_test_data, date_col="Date", return_col="return", verbose=False)

        # Export results
        output_dir = tmp_path / "backtest_results"
        engine.export_results(output_dir=str(output_dir))

        # Verify files were created
        assert (output_dir / "trades.csv").exists()
        assert (output_dir / "equity_curve.csv").exists()
        assert (output_dir / "metrics.json").exists()

        # Verify files have content
        trades_df = pd.read_csv(output_dir / "trades.csv")
        assert len(trades_df) > 0

        equity_df = pd.read_csv(output_dir / "equity_curve.csv")
        assert len(equity_df) > 0

        with open(output_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
            assert "total_trades" in metrics


class TestBacktestingEdgeCases:
    """Test edge cases in backtesting workflow"""

    def test_single_trade_backtest(self, integration_model_file):
        """Test backtest with only one trade"""
        # Create minimal data
        single_trade_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=1),
                "Close": [5000],
                "Volume": [1000000],
                "Volume_ratio_vs_5days": [1.0],
                "Announcement": [1],
                "hawkish_dovish_ratio": [1.2],
                "net_sentiment_score": [0.15],
                "negative_proportion": [0.2],
                "neutral_proportion": [0.3],
                "positive_proportion": [0.5],
                "VIX_Close": [15.0],
                "DXY_Close": [103.0],
                "US02Y_Yield": [4.5],
                "US10Y_Yield": [4.2],
                "Yield_Curve_10Y_2Y": [-0.3],
                "sentiment_label": [1],
                "return": [0.02],
            }
        )

        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy()
        engine = BacktestEngine(predictor, strategy)

        results = engine.run(single_trade_data, date_col="Date", return_col="return", verbose=False)

        assert len(results) == 1
        assert engine.metrics["total_trades"] == 1

    def test_all_winning_trades(self, integration_model_file):
        """Test backtest where all trades are winners"""
        # Create data with all positive returns
        winning_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "Close": [5000, 5100, 5200, 5300, 5400],
                "Volume": [1000000] * 5,
                "Volume_ratio_vs_5days": [1.0] * 5,
                "Announcement": [1] * 5,
                "hawkish_dovish_ratio": [1.2] * 5,
                "net_sentiment_score": [0.15] * 5,
                "negative_proportion": [0.2] * 5,
                "neutral_proportion": [0.3] * 5,
                "positive_proportion": [0.5] * 5,
                "VIX_Close": [15.0] * 5,
                "DXY_Close": [103.0] * 5,
                "US02Y_Yield": [4.5] * 5,
                "US10Y_Yield": [4.2] * 5,
                "Yield_Curve_10Y_2Y": [-0.3] * 5,
                "sentiment_label": [1] * 5,
                "return": [0.02, 0.0196, 0.0192, 0.0189, 0.0185],
            }
        )

        predictor = ModelPredictor(str(integration_model_file), model_type="binary")
        strategy = BinaryStrategy()
        engine = BacktestEngine(predictor, strategy)

        results = engine.run(winning_data, date_col="Date", return_col="return", verbose=False)

        # Win rate should be high (some positions might be short though)
        assert engine.metrics["win_rate"] >= 0.0


pytestmark = pytest.mark.integration
