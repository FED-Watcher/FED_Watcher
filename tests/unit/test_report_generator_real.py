"""Real integration tests for report generator"""

import pytest
import pandas as pd
from unittest.mock import Mock
from src.backtesting.report_generator import generate_html_report


@pytest.fixture
def mock_engine():
    engine = Mock()
    engine.strategy = Mock()
    engine.strategy.name = "Test Strategy"
    engine.initial_capital = 100000
    engine.metrics = {
        "total_return_pct": 15.5,
        "sharpe_ratio": 1.8,
        "win_rate": 0.62,
        "max_drawdown_pct": -8.3,
        "final_equity": 115500,
        "total_pnl": 15500,
        "total_trades": 20,
        "winning_trades": 13,
        "losing_trades": 7,
        "annualized_return_pct": 18.2,
        "sortino_ratio": 2.1,
        "calmar_ratio": 2.2,
        "profit_factor": 2.5,
        "expectancy": 1.2,
        "recovery_factor": 1.9,
        "max_win_streak": 5,
        "max_loss_streak": 3,
        "best_trade_pct": 8.5,
        "worst_trade_pct": -4.2,
    }
    engine.trades_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "prediction_label": ["Up", "Down", "Up"],
            "position": [1.0, -1.0, 1.0],
            "actual_return_pct": [2.5, -1.5, 3.0],
            "pnl_pct": [2.5, 1.5, 3.0],
            "pnl_dollars": [2500, 1500, 3000],
            "equity": [102500, 104000, 107000],
        }
    )
    engine.equity_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "equity": [102500, 104000, 107000],
        }
    )
    return engine


class TestReportGenerator:
    def test_report_created(self, mock_engine, tmp_path):
        report_path = generate_html_report(mock_engine, str(tmp_path))
        assert report_path.exists()

    def test_report_contains_metrics(self, mock_engine, tmp_path):
        report_path = generate_html_report(mock_engine, str(tmp_path))
        content = report_path.read_text()
        assert "15.50%" in content
        assert "1.8" in content

    def test_report_has_structure(self, mock_engine, tmp_path):
        report_path = generate_html_report(mock_engine, str(tmp_path))
        content = report_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<table" in content


pytestmark = pytest.mark.unit
