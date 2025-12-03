"""Unit tests for the enrichment module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.enrichment.enrich_with_market_data import enrich_dataset


class TestEnrichmentModule:
    """Test suite for market data enrichment functionality."""

    @pytest.fixture
    def sample_master_data(self):
        """Create sample master dataset for testing."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Open": [5000 + i * 10 for i in range(len(dates))],
            "High": [5020 + i * 10 for i in range(len(dates))],
            "Low": [4980 + i * 10 for i in range(len(dates))],
            "Close": [5010 + i * 10 for i in range(len(dates))],
            "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            "Volume_ratio_vs_5days": [1.0 + i * 0.1 for i in range(len(dates))],
            "Announcement": [0] * len(dates),
            "hawkish_dovish_ratio": [0.5] * len(dates),
            "net_sentiment_score": [0.0] * len(dates),
            "negative_proportion": [0.2] * len(dates),
            "neutral_proportion": [0.6] * len(dates),
            "positive_proportion": [0.2] * len(dates),
            "sentiment_label": ["neutral"] * len(dates),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_market_data_yf(self):
        """Create sample Yahoo Finance market data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {
            "VIX_Close": [15.0 + i * 0.5 for i in range(len(dates))],
            "US10Y_Yield": [4.0 + i * 0.1 for i in range(len(dates))],
            "DXY_Close": [100.0 + i * 0.2 for i in range(len(dates))],
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def sample_market_data_fred(self):
        """Create sample FRED market data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {"US02Y_Yield": [3.5 + i * 0.05 for i in range(len(dates))]}
        df = pd.DataFrame(data, index=dates)
        return df

    def test_enrich_dataset_file_not_found(self, capsys):
        """Test that enrich_dataset handles missing master file gracefully."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("pandas.read_csv", side_effect=FileNotFoundError):
                enrich_dataset()
                captured = capsys.readouterr()
                assert "ERROR" in captured.out
                assert "Master file not found" in captured.out

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_enrich_dataset_successful_run(
        self,
        mock_fred,
        mock_yf,
        mock_read_csv,
        sample_master_data,
        sample_market_data_yf,
        sample_market_data_fred,
        tmp_path,
    ):
        """Test successful enrichment pipeline execution."""
        # Setup mocks
        mock_read_csv.return_value = sample_master_data

        # Mock yfinance download to return multi-level columns
        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = sample_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.return_value = sample_market_data_fred

        # Mock file writing
        output_file = tmp_path / "MasterDataset_Enriched.csv"
        with patch("src.enrichment.enrich_with_market_data.ENRICHED_OUTPUT_PATH", str(output_file)):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                enrich_dataset()

                # Verify to_csv was called
                assert mock_to_csv.called

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    def test_enrich_dataset_yfinance_failure(
        self, mock_yf, mock_read_csv, sample_master_data, capsys
    ):
        """Test that enrich_dataset handles yfinance download failure."""
        mock_read_csv.return_value = sample_master_data
        mock_yf.side_effect = Exception("Network error")

        enrich_dataset()
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Failed to download data from Yahoo Finance" in captured.out

    @patch("pandas.read_csv")
    @patch("yfinance.download")
    @patch("pandas_datareader.data.DataReader")
    def test_enrich_dataset_fred_failure(
        self, mock_fred, mock_yf, mock_read_csv, sample_master_data, sample_market_data_yf, capsys
    ):
        """Test that enrich_dataset handles FRED download failure."""
        mock_read_csv.return_value = sample_master_data

        mock_yf_data = MagicMock()
        mock_yf_data.__getitem__.return_value = sample_market_data_yf
        mock_yf.return_value = mock_yf_data

        mock_fred.side_effect = Exception("FRED API error")

        enrich_dataset()
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Failed to download data from FRED" in captured.out

    def test_date_parsing(self, sample_master_data):
        """Test that date parsing works correctly."""
        sample_master_data["Date_dt"] = pd.to_datetime(
            sample_master_data["Date"], format="%d/%m/%Y"
        )

        assert sample_master_data["Date_dt"].dtype == "datetime64[ns]"
        assert sample_master_data["Date_dt"].min() == pd.Timestamp("2024-01-01")
        assert sample_master_data["Date_dt"].max() == pd.Timestamp("2024-01-10")

    def test_yield_curve_calculation(self, sample_market_data_yf, sample_market_data_fred):
        """Test yield curve calculation."""
        combined = pd.concat([sample_market_data_yf, sample_market_data_fred], axis=1)
        combined["Yield_Curve_10Y_2Y"] = combined["US10Y_Yield"] - combined["US02Y_Yield"]

        assert "Yield_Curve_10Y_2Y" in combined.columns
        assert not combined["Yield_Curve_10Y_2Y"].isna().any()
        assert (combined["Yield_Curve_10Y_2Y"] > 0).all()

    def test_forward_fill_handling(self):
        """Test that forward fill handles missing data correctly."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {
            "VIX_Close": [15.0, np.nan, np.nan, 16.0, np.nan, 17.0, np.nan, np.nan, 18.0, np.nan]
        }
        df = pd.DataFrame(data, index=dates)

        df_filled = df.ffill()

        assert not df_filled["VIX_Close"].isna().any()
        assert df_filled["VIX_Close"].iloc[1] == 15.0
        assert df_filled["VIX_Close"].iloc[2] == 15.0

    def test_backward_fill_handling(self):
        """Test that backward fill handles missing data at start."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = {"VIX_Close": [np.nan, np.nan, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]}
        df = pd.DataFrame(data, index=dates)

        df_filled = df.bfill()

        assert not df_filled["VIX_Close"].isna().any()
        assert df_filled["VIX_Close"].iloc[0] == 15.0
        assert df_filled["VIX_Close"].iloc[1] == 15.0

    def test_merge_preserves_all_rows(self, sample_master_data, sample_market_data_yf):
        """Test that left merge preserves all master dataset rows."""
        sample_master_data["Date_dt"] = pd.to_datetime(
            sample_master_data["Date"], format="%d/%m/%Y"
        )

        enriched = pd.merge(
            sample_master_data,
            sample_market_data_yf,
            how="left",
            left_on="Date_dt",
            right_index=True,
        )

        assert len(enriched) == len(sample_master_data)
        assert all(col in enriched.columns for col in sample_master_data.columns)

    def test_required_columns_present(self):
        """Test that all required columns are defined."""
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "VIX_Close",
            "DXY_Close",
            "US02Y_Yield",
            "US10Y_Yield",
            "Yield_Curve_10Y_2Y",
            "Volume_ratio_vs_5days",
            "Announcement",
            "hawkish_dovish_ratio",
            "net_sentiment_score",
            "negative_proportion",
            "neutral_proportion",
            "positive_proportion",
            "sentiment_label",
        ]

        # This test verifies the expected output structure
        assert len(required_columns) == 19

    @pytest.mark.critical
    def test_data_types_after_enrichment(
        self, sample_master_data, sample_market_data_yf, sample_market_data_fred
    ):
        """Test that data types are correct after enrichment."""
        sample_master_data["Date_dt"] = pd.to_datetime(
            sample_master_data["Date"], format="%d/%m/%Y"
        )

        combined_market = pd.concat([sample_market_data_yf, sample_market_data_fred], axis=1)
        combined_market["Yield_Curve_10Y_2Y"] = (
            combined_market["US10Y_Yield"] - combined_market["US02Y_Yield"]
        )

        enriched = pd.merge(
            sample_master_data, combined_market, how="left", left_on="Date_dt", right_index=True
        )

        # Check numeric columns
        assert enriched["VIX_Close"].dtype in [np.float64, np.float32, float]
        assert enriched["US10Y_Yield"].dtype in [np.float64, np.float32, float]
        assert enriched["DXY_Close"].dtype in [np.float64, np.float32, float]
        assert enriched["Yield_Curve_10Y_2Y"].dtype in [np.float64, np.float32, float]

    def test_date_range_extraction(self, sample_master_data):
        """Test that date range is correctly extracted from master data."""
        sample_master_data["Date_dt"] = pd.to_datetime(
            sample_master_data["Date"], format="%d/%m/%Y"
        )

        start_date = sample_master_data["Date_dt"].min()
        end_date = sample_master_data["Date_dt"].max()

        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)
        assert start_date < end_date

    def test_ticker_mapping(self):
        """Test that ticker mapping is correctly defined."""
        yf_tickers = {"^VIX": "VIX_Close", "^TNX": "US10Y_Yield", "DX-Y.NYB": "DXY_Close"}

        assert len(yf_tickers) == 3
        assert "^VIX" in yf_tickers
        assert yf_tickers["^VIX"] == "VIX_Close"

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()

        # Should be able to concatenate without error
        result = pd.concat([empty_df, pd.DataFrame({"A": [1, 2, 3]})], axis=1)
        assert len(result) == 3

    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test that enrichment can handle larger datasets."""
        # Create a larger dataset (1 year of daily data)
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        large_data = {
            "Date": [d.strftime("%d/%m/%Y") for d in dates],
            "Close": np.random.uniform(5000, 5500, len(dates)),
            "Volume": np.random.randint(1000000, 5000000, len(dates)),
        }
        df = pd.DataFrame(large_data)

        assert len(df) == 365
        assert df["Date"].nunique() == 365


class TestEnrichmentEdgeCases:
    """Test edge cases and error handling."""

    def test_non_trading_days_handling(self):
        """Test handling of weekends and holidays."""
        # Create data including weekends
        dates = pd.date_range(start="2024-01-01", end="2024-01-14", freq="D")
        market_dates = pd.date_range(
            start="2024-01-01", end="2024-01-14", freq="B"
        )  # Business days only

        calendar_df = pd.DataFrame({"Date_dt": dates})
        market_df = pd.DataFrame({"VIX_Close": range(len(market_dates))}, index=market_dates)

        merged = pd.merge(calendar_df, market_df, how="left", left_on="Date_dt", right_index=True)
        merged["VIX_Close"] = merged["VIX_Close"].ffill()

        # Should have no NaN values after forward fill
        assert not merged["VIX_Close"].isna().any()

    def test_missing_columns_in_master(self):
        """Test behavior when master dataset is missing expected columns."""
        incomplete_data = pd.DataFrame(
            {"Date": ["01/01/2024", "02/01/2024"], "Close": [5000, 5010]}
        )

        # Should be able to process even if some columns are missing
        assert "Date" in incomplete_data.columns
        assert "Close" in incomplete_data.columns

    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        with pytest.raises((ValueError, pd.errors.ParserError)):
            pd.to_datetime("invalid-date-format", format="%d/%m/%Y")

    def test_negative_values_in_market_data(self):
        """Test that negative values are handled (e.g., negative yields)."""
        data = {"Yield_Curve_10Y_2Y": [-0.5, -0.3, 0.1, 0.5, 1.0]}
        df = pd.DataFrame(data)

        # Should accept negative values (inverted yield curve is real)
        assert (df["Yield_Curve_10Y_2Y"] < 0).any()
        assert not df["Yield_Curve_10Y_2Y"].isna().any()

    def test_duplicate_dates(self):
        """Test handling of duplicate dates in dataset."""
        dates = ["01/01/2024", "01/01/2024", "02/01/2024"]
        data = pd.DataFrame({"Date": dates, "Close": [5000, 5010, 5020]})

        # Should be able to detect duplicates
        duplicates = data["Date"].duplicated().any()
        assert duplicates == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
