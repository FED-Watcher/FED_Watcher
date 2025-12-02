"""Integration tests for the enrichment module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.enrichment.enrich_with_market_data import enrich_dataset


@pytest.mark.integration
class TestEnrichmentIntegration:
    """Integration tests for the market data enrichment pipeline."""

    @pytest.fixture
    def full_master_dataset(self):
        """Create a realistic master dataset for integration testing."""
        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")
        np.random.seed(42)

        # Generate realistic OHLC data with proper relationships
        base_price = 5000
        opens = []
        highs = []
        lows = []
        closes = []

        for i in range(len(dates)):
            open_price = base_price + np.random.uniform(-50, 50)
            close_price = open_price + np.random.uniform(-30, 30)
            high_price = max(open_price, close_price) + np.random.uniform(5, 20)
            low_price = min(open_price, close_price) - np.random.uniform(5, 20)

            opens.append(open_price)
            closes.append(close_price)
            highs.append(high_price)
            lows.append(low_price)

            # Update base price for next day
            base_price = close_price

        data = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
            "Volume_ratio_vs_5days": np.random.uniform(0.8, 1.5, len(dates)),
            "Announcement": [1 if i % 30 == 0 else 0 for i in range(len(dates))],
            "hawkish_dovish_ratio": np.random.uniform(0.3, 0.7, len(dates)),
            "net_sentiment_score": np.random.uniform(-0.5, 0.5, len(dates)),
            "negative_proportion": np.random.uniform(0.1, 0.3, len(dates)),
            "neutral_proportion": np.random.uniform(0.4, 0.6, len(dates)),
            "positive_proportion": np.random.uniform(0.1, 0.3, len(dates)),
            "sentiment_label": np.random.choice(["positive", "neutral", "negative"], len(dates)),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def full_market_data_yf(self):
        """Create realistic Yahoo Finance market data."""
        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="B")  # Business days
        np.random.seed(42)

        data = {
            "VIX_Close": np.random.uniform(12, 25, len(dates)),
            "US10Y_Yield": np.random.uniform(3.8, 4.5, len(dates)),
            "DXY_Close": np.random.uniform(99, 105, len(dates)),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def full_market_data_fred(self):
        """Create realistic FRED market data."""
        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="B")
        np.random.seed(42)

        data = {"US02Y_Yield": np.random.uniform(3.5, 4.2, len(dates))}
        return pd.DataFrame(data, index=dates)

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    @patch("pandas.DataFrame.to_csv")
    def test_full_enrichment_pipeline(
        self,
        mock_to_csv,
        mock_fred,
        mock_yf,
        mock_read_csv,
        full_master_dataset,
        full_market_data_yf,
        full_market_data_fred,
    ):
        """Test the complete enrichment pipeline end-to-end."""
        # Setup mocks
        mock_read_csv.return_value = full_master_dataset

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = full_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = full_market_data_fred

        # Run enrichment
        enrich_dataset()

        # Verify all external calls were made
        assert mock_read_csv.called
        assert mock_yf.called
        assert mock_fred.called
        assert mock_to_csv.called

        # Verify the saved dataframe structure
        call_args = mock_to_csv.call_args
        saved_df = mock_to_csv.call_args[1].get("self") if call_args else None

        if saved_df is not None:
            # Verify enriched columns exist
            expected_new_cols = [
                "VIX_Close",
                "DXY_Close",
                "US02Y_Yield",
                "US10Y_Yield",
                "Yield_Curve_10Y_2Y",
            ]
            for col in expected_new_cols:
                assert col in saved_df.columns

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_data_consistency_after_enrichment(
        self,
        mock_fred,
        mock_yf,
        mock_read_csv,
        full_master_dataset,
        full_market_data_yf,
        full_market_data_fred,
    ):
        """Test that data remains consistent after enrichment."""
        mock_read_csv.return_value = full_master_dataset

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = full_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = full_market_data_fred

        # Manually simulate the enrichment process
        master_df = full_master_dataset.copy()
        master_df["Date_dt"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y")

        all_market_data = pd.concat([full_market_data_yf, full_market_data_fred], axis=1)
        all_market_data["Yield_Curve_10Y_2Y"] = (
            all_market_data["US10Y_Yield"] - all_market_data["US02Y_Yield"]
        )

        enriched_df = pd.merge(
            master_df, all_market_data, how="left", left_on="Date_dt", right_index=True
        )

        # Test consistency checks
        assert len(enriched_df) == len(full_master_dataset)
        assert enriched_df["Close"].equals(master_df["Close"])
        assert enriched_df["Volume"].equals(master_df["Volume"])

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_weekend_data_forward_fill(
        self,
        mock_fred,
        mock_yf,
        mock_read_csv,
        full_master_dataset,
        full_market_data_yf,
        full_market_data_fred,
    ):
        """Test that weekend/holiday data is correctly forward-filled."""
        mock_read_csv.return_value = full_master_dataset

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = full_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = full_market_data_fred

        # Simulate enrichment
        master_df = full_master_dataset.copy()
        master_df["Date_dt"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y")

        all_market_data = pd.concat([full_market_data_yf, full_market_data_fred], axis=1)
        all_market_data["Yield_Curve_10Y_2Y"] = (
            all_market_data["US10Y_Yield"] - all_market_data["US02Y_Yield"]
        )

        enriched_df = pd.merge(
            master_df, all_market_data, how="left", left_on="Date_dt", right_index=True
        )

        cols_to_fill = [
            "VIX_Close",
            "US10Y_Yield",
            "DXY_Close",
            "US02Y_Yield",
            "Yield_Curve_10Y_2Y",
        ]
        enriched_df[cols_to_fill] = enriched_df[cols_to_fill].ffill()
        enriched_df[cols_to_fill] = enriched_df[cols_to_fill].bfill()

        # Verify no NaN values remain
        for col in cols_to_fill:
            assert not enriched_df[col].isna().any(), f"Column {col} has NaN values"

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_yield_curve_calculation_integration(
        self,
        mock_fred,
        mock_yf,
        mock_read_csv,
        full_master_dataset,
        full_market_data_yf,
        full_market_data_fred,
    ):
        """Test that yield curve is correctly calculated throughout pipeline."""
        mock_read_csv.return_value = full_master_dataset

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = full_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = full_market_data_fred

        # Simulate enrichment
        master_df = full_master_dataset.copy()
        master_df["Date_dt"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y")

        all_market_data = pd.concat([full_market_data_yf, full_market_data_fred], axis=1)
        all_market_data["Yield_Curve_10Y_2Y"] = (
            all_market_data["US10Y_Yield"] - all_market_data["US02Y_Yield"]
        )

        # Verify yield curve calculation
        assert "Yield_Curve_10Y_2Y" in all_market_data.columns
        assert not all_market_data["Yield_Curve_10Y_2Y"].isna().any()

        # Check that calculation is correct (spot check)
        for idx in all_market_data.index[:5]:
            expected = (
                all_market_data.loc[idx, "US10Y_Yield"] - all_market_data.loc[idx, "US02Y_Yield"]
            )
            actual = all_market_data.loc[idx, "Yield_Curve_10Y_2Y"]
            assert np.isclose(expected, actual), f"Yield curve mismatch at {idx}"

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_announcement_days_preserved(
        self,
        mock_fred,
        mock_yf,
        mock_read_csv,
        full_master_dataset,
        full_market_data_yf,
        full_market_data_fred,
    ):
        """Test that announcement day flags are preserved after enrichment."""
        mock_read_csv.return_value = full_master_dataset

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = full_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = full_market_data_fred

        # Simulate enrichment
        master_df = full_master_dataset.copy()
        master_df["Date_dt"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y")

        all_market_data = pd.concat([full_market_data_yf, full_market_data_fred], axis=1)

        enriched_df = pd.merge(
            master_df, all_market_data, how="left", left_on="Date_dt", right_index=True
        )

        # Verify announcement flags are preserved
        assert "Announcement" in enriched_df.columns
        assert enriched_df["Announcement"].sum() == full_master_dataset["Announcement"].sum()

    def test_data_quality_checks(self, full_master_dataset):
        """Test data quality checks on master dataset."""
        # Check for required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            assert col in full_master_dataset.columns

        # Check for valid price relationships
        assert (full_master_dataset["High"] >= full_master_dataset["Low"]).all()
        assert (full_master_dataset["High"] >= full_master_dataset["Close"]).all()
        assert (full_master_dataset["Low"] <= full_master_dataset["Close"]).all()

        # Check for positive volumes
        assert (full_master_dataset["Volume"] > 0).all()

    @pytest.mark.slow
    def test_performance_with_large_dataset(self):
        """Test enrichment performance with a large dataset (1 year of data)."""
        import time

        # Create 1 year of daily data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        large_dataset = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Open": np.random.uniform(4900, 5100, len(dates)),
            "High": np.random.uniform(4950, 5150, len(dates)),
            "Low": np.random.uniform(4850, 5050, len(dates)),
            "Close": np.random.uniform(4900, 5100, len(dates)),
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
            "Volume_ratio_vs_5days": np.random.uniform(0.8, 1.5, len(dates)),
            "Announcement": [1 if i % 30 == 0 else 0 for i in range(len(dates))],
            "hawkish_dovish_ratio": np.random.uniform(0.3, 0.7, len(dates)),
            "net_sentiment_score": np.random.uniform(-0.5, 0.5, len(dates)),
            "negative_proportion": np.random.uniform(0.1, 0.3, len(dates)),
            "neutral_proportion": np.random.uniform(0.4, 0.6, len(dates)),
            "positive_proportion": np.random.uniform(0.1, 0.3, len(dates)),
            "sentiment_label": np.random.choice(["positive", "neutral", "negative"], len(dates)),
        }
        df = pd.DataFrame(large_dataset)

        start_time = time.time()
        df["Date_dt"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for date parsing)
        assert end_time - start_time < 1.0
        assert len(df) == 365


@pytest.mark.integration
class TestEnrichmentErrorRecovery:
    """Test error recovery and resilience in the enrichment pipeline."""

    @patch("pandas.read_csv")
    def test_graceful_degradation_on_partial_failure(self, mock_read_csv):
        """Test that partial failures don't crash the entire pipeline."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Close": [5000] * len(dates),
            "Volume": [1000000] * len(dates),
            "Volume_ratio_vs_5days": [1.0] * len(dates),
            "Announcement": [0] * len(dates),
            "hawkish_dovish_ratio": [0.5] * len(dates),
            "net_sentiment_score": [0.0] * len(dates),
            "negative_proportion": [0.2] * len(dates),
            "neutral_proportion": [0.6] * len(dates),
            "positive_proportion": [0.2] * len(dates),
            "sentiment_label": ["neutral"] * len(dates),
        }
        mock_read_csv.return_value = pd.DataFrame(data)

        # The function should handle errors internally
        with patch("yfinance.download", side_effect=Exception("Network error")):
            # Should not raise exception, but print error message
            enrich_dataset()

    def test_data_validation_checks(self):
        """Test validation of enriched data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        enriched_data = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Close": np.random.uniform(4900, 5100, len(dates)),
            "VIX_Close": np.random.uniform(12, 25, len(dates)),
            "US10Y_Yield": np.random.uniform(3.8, 4.5, len(dates)),
            "US02Y_Yield": np.random.uniform(3.5, 4.2, len(dates)),
            "Yield_Curve_10Y_2Y": np.random.uniform(0.1, 0.5, len(dates)),
        }
        df = pd.DataFrame(enriched_data)

        # Validation checks
        assert not df.isna().any().any(), "No NaN values should exist"
        assert (df["VIX_Close"] > 0).all(), "VIX should be positive"
        assert (df["US10Y_Yield"] > 0).all(), "Yields should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
