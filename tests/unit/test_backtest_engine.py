"""Comprehensive unit tests for BacktestEngine"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategies import BinaryStrategy, MultiClassStrategy
from src.backtesting.model_predictor import ModelPredictor


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing"""
    predictor = Mock(spec=ModelPredictor)
    predictor.model_type = "binary"
    predictor.predict.return_value = np.array([1, 0, 1, 1, 0])
    predictor.predict_proba.return_value = np.array(
        [[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]]
    )
    predictor.get_prediction_label.side_effect = lambda x: "Up" if x == 1 else "Down"
    return predictor


@pytest.fixture
def test_data():
    """Create test data for backtesting"""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "Date": dates,
            "return": [0.02, -0.01, 0.03, 0.015, -0.02],
            "Close": [5000, 4950, 5100, 5176.5, 5073.0],
            "Volume": [1000000, 1100000, 1200000, 950000, 1050000],
        }
    )
    return data


@pytest.fixture
def binary_strategy():
    """Create binary strategy for testing"""
    return BinaryStrategy(name="Test Binary")


class TestBacktestEngineInitialization:
    """Test BacktestEngine initialization"""

    def test_initialization_default_params(self, mock_predictor, binary_strategy, capsys):
        """Test engine initializes with default parameters"""
        engine = BacktestEngine(mock_predictor, binary_strategy)

        assert engine.predictor == mock_predictor
        assert engine.strategy == binary_strategy
        assert engine.initial_capital == 100000
        assert engine.horizon == 1
        assert engine.commission_pct == 0.0
        assert engine.slippage_pct == 0.0
        assert engine.trades == []
        assert engine.equity_curve == []
        assert engine.metrics == {}

        # Check output
        captured = capsys.readouterr()
        assert "BACKTESTING ENGINE INITIALIZED" in captured.out
        assert "binary" in captured.out.lower()

    def test_initialization_custom_params(self, mock_predictor, binary_strategy):
        """Test engine initializes with custom parameters"""
        engine = BacktestEngine(
            mock_predictor,
            binary_strategy,
            initial_capital=50000,
            horizon=2,
            commission_pct=0.1,
            slippage_pct=0.05,
        )

        assert engine.initial_capital == 50000
        assert engine.horizon == 2
        assert engine.commission_pct == 0.1
        assert engine.slippage_pct == 0.05


class TestBacktestEngineRun:
    """Test BacktestEngine.run() method"""

    def test_run_with_date_column(self, mock_predictor, binary_strategy, test_data):
        """Test run with date as column"""
        engine = BacktestEngine(mock_predictor, binary_strategy, initial_capital=100000)

        results = engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        assert "date" in results.columns
        assert "prediction" in results.columns
        assert "position" in results.columns
        assert "pnl_pct" in results.columns
        assert "equity" in results.columns

    def test_run_with_date_index(self, mock_predictor, binary_strategy, test_data):
        """Test run with date as index"""
        test_data_indexed = test_data.set_index("Date")
        engine = BacktestEngine(mock_predictor, binary_strategy)

        results = engine.run(test_data_indexed, date_col="Date", return_col="return", verbose=False)

        assert len(results) == 5

    def test_run_missing_date_column(self, mock_predictor, binary_strategy, test_data):
        """Test run raises error when date column missing"""
        test_data_no_date = test_data.drop("Date", axis=1)
        engine = BacktestEngine(mock_predictor, binary_strategy)

        with pytest.raises(ValueError, match="Date column"):
            engine.run(test_data_no_date, date_col="Date", return_col="return")

    def test_run_with_probabilities(self, mock_predictor, binary_strategy, test_data):
        """Test run stores probabilities when available"""
        engine = BacktestEngine(mock_predictor, binary_strategy)

        results = engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "max_probability" in results.columns

    def test_run_without_probabilities(self, mock_predictor, binary_strategy, test_data):
        """Test run handles missing probabilities"""
        mock_predictor.predict_proba.side_effect = AttributeError("No probabilities")
        engine = BacktestEngine(mock_predictor, binary_strategy)

        results = engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert len(results) == 5
        assert "max_probability" not in results.columns

    def test_run_applies_commission(self, mock_predictor, binary_strategy, test_data):
        """Test commission is applied to trades"""
        engine = BacktestEngine(
            mock_predictor, binary_strategy, commission_pct=0.1, slippage_pct=0.05
        )

        results = engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        # First trade: Long position with 2% return
        # Expected: (1.0 * 2.0) - 0.1 - 0.05 = 1.85%
        assert results.iloc[0]["pnl_pct"] == pytest.approx(1.85, abs=0.01)

    def test_run_equity_tracking(self, mock_predictor, binary_strategy, test_data):
        """Test equity is tracked correctly"""
        engine = BacktestEngine(mock_predictor, binary_strategy, initial_capital=100000)

        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert len(engine.equity_curve) == 5
        assert engine.equity_df.iloc[0]["equity"] != 100000  # Changed from start

    def test_run_verbose_output(self, mock_predictor, binary_strategy, test_data, capsys):
        """Test verbose mode prints progress"""
        engine = BacktestEngine(mock_predictor, binary_strategy)

        engine.run(test_data, date_col="Date", return_col="return", verbose=True)

        captured = capsys.readouterr()
        assert "Starting backtest" in captured.out
        assert "BACKTEST COMPLETE" in captured.out


class TestBacktestEngineMetrics:
    """Test metric calculation methods"""

    def test_calculate_metrics(self, mock_predictor, binary_strategy, test_data):
        """Test metrics are calculated correctly"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        metrics = engine.metrics

        assert "total_trades" in metrics
        assert "total_return_pct" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert metrics["total_trades"] == 5

    def test_sharpe_ratio_calculation(self, mock_predictor, binary_strategy, test_data):
        """Test Sharpe ratio calculation"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "sharpe_ratio" in engine.metrics
        assert isinstance(engine.metrics["sharpe_ratio"], (int, float))

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Create engine without running
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        returns = pd.Series([1.0])  # Single return
        sharpe = engine._calculate_sharpe(returns)

        assert sharpe == 0.0

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        returns = pd.Series([2.0, 2.0, 2.0])  # No variance
        sharpe = engine._calculate_sharpe(returns)

        assert sharpe == 0.0

    def test_sortino_ratio_calculation(self, mock_predictor, binary_strategy, test_data):
        """Test Sortino ratio calculation"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "sortino_ratio" in engine.metrics

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside returns"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        returns = pd.Series([1.0, 2.0, 3.0])  # All positive
        sortino = engine._calculate_sortino(returns)

        assert sortino == 0.0

    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        calmar = engine._calculate_calmar(20.0, -10.0)
        assert calmar == 2.0

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        calmar = engine._calculate_calmar(20.0, 0.0)
        assert calmar == 0.0

    def test_max_drawdown_calculation(self, mock_predictor, binary_strategy, test_data):
        """Test max drawdown calculation"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "max_drawdown_pct" in engine.metrics
        assert engine.metrics["max_drawdown_pct"] <= 0  # Drawdown is negative

    def test_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        equity = pd.Series([100000])
        dd = engine._calculate_max_drawdown(equity)

        assert dd == 0.0

    def test_streaks_calculation(self, mock_predictor, binary_strategy, test_data):
        """Test win/loss streak calculation"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "max_win_streak" in engine.metrics
        assert "max_loss_streak" in engine.metrics

    def test_streaks_empty_trades(self):
        """Test streaks with empty trades"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        trades_df = pd.DataFrame()
        win_streak, loss_streak = engine._calculate_streaks(trades_df)

        assert win_streak == 0
        assert loss_streak == 0

    def test_annualize_return(self):
        """Test return annualization"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        annualized = engine._annualize_return(20.0, 12, trades_per_year=12)
        assert annualized == pytest.approx(20.0, abs=0.1)

    def test_annualize_return_zero_trades(self):
        """Test annualization with zero trades"""
        mock_pred = Mock()
        mock_strat = Mock()
        engine = BacktestEngine(mock_pred, mock_strat)

        annualized = engine._annualize_return(20.0, 0)
        assert annualized == 0.0

    def test_profit_factor_calculation(self, mock_predictor, binary_strategy, test_data):
        """Test profit factor calculation"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        assert "profit_factor" in engine.metrics


class TestBacktestEnginePrinting:
    """Test printing and output methods"""

    def test_print_summary(self, mock_predictor, binary_strategy, test_data, capsys):
        """Test print_summary outputs correctly"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.print_summary()

        captured = capsys.readouterr()
        assert "PERFORMANCE SUMMARY" in captured.out
        assert "Total Trades:" in captured.out
        assert "Sharpe Ratio:" in captured.out


class TestBacktestEnginePlotting:
    """Test plotting methods"""

    @patch("matplotlib.pyplot.show")
    def test_plot_equity_curve(self, mock_show, mock_predictor, binary_strategy, test_data):
        """Test equity curve plotting"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.plot_equity_curve(show_trades=True)

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_equity_curve_save(
        self, mock_savefig, mock_show, mock_predictor, binary_strategy, test_data, tmp_path
    ):
        """Test equity curve plotting with save"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        save_path = tmp_path / "equity.png"
        engine.plot_equity_curve(save_path=str(save_path), show_trades=False)

        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_returns_distribution(self, mock_show, mock_predictor, binary_strategy, test_data):
        """Test returns distribution plotting"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.plot_returns_distribution()

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_drawdown(self, mock_show, mock_predictor, binary_strategy, test_data):
        """Test drawdown plotting"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.plot_drawdown()

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_monthly_returns(self, mock_show, mock_predictor, binary_strategy, test_data):
        """Test monthly returns heatmap"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.plot_monthly_returns()

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_trade_analysis(self, mock_show, mock_predictor, binary_strategy, test_data):
        """Test comprehensive trade analysis plot"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        engine.plot_trade_analysis()

        mock_show.assert_called_once()


class TestBacktestEngineExport:
    """Test export functionality"""

    def test_export_results(self, mock_predictor, binary_strategy, test_data, tmp_path):
        """Test results export to files"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        output_dir = tmp_path / "results"
        engine.export_results(output_dir=str(output_dir))

        assert (output_dir / "trades.csv").exists()
        assert (output_dir / "equity_curve.csv").exists()
        assert (output_dir / "metrics.json").exists()

    def test_export_creates_directory(self, mock_predictor, binary_strategy, test_data, tmp_path):
        """Test export creates output directory if missing"""
        engine = BacktestEngine(mock_predictor, binary_strategy)
        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        output_dir = tmp_path / "new_results"
        assert not output_dir.exists()

        engine.export_results(output_dir=str(output_dir))

        assert output_dir.exists()


pytestmark = pytest.mark.unit
