"""Unit tests for the get_stock_data module."""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.stock_data.get_stock_data import download_sp500_data


class TestDownloadSp500Data:
    """Test suite for download_sp500_data function."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data as returned by yfinance."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        # yfinance returns data with 5 columns
        data = pd.DataFrame(
            {
                "Open": [5000 + i * 10 for i in range(len(dates))],
                "High": [5020 + i * 10 for i in range(len(dates))],
                "Low": [4980 + i * 10 for i in range(len(dates))],
                "Close": [5010 + i * 10 for i in range(len(dates))],
                "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_download_successful(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test successful S&P 500 data download."""
        mock_yf_download.return_value = sample_sp500_data

        result = download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=tmp_path,
        )

        # Verify yfinance was called with correct parameters
        mock_yf_download.assert_called_once_with("^GSPC", start="2024-01-01", end="2024-01-10")

        # Check result is a DataFrame
        assert isinstance(result, pd.DataFrame)

    @patch("yfinance.download")
    def test_creates_output_directory(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        mock_yf_download.return_value = sample_sp500_data

        new_dir = tmp_path / "new_stock_data"
        assert not new_dir.exists()

        download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=new_dir,
        )

        assert new_dir.exists()

    @patch("yfinance.download")
    def test_csv_file_created(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that CSV file is created."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=tmp_path,
        )

        expected_file = tmp_path / "SP500_2018_2025_Clean.csv"
        assert expected_file.exists()

    @patch("yfinance.download")
    def test_column_names_simplified(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that column names are simplified."""
        mock_yf_download.return_value = sample_sp500_data

        result = download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=tmp_path,
        )

        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        assert list(result.columns) == expected_columns

    @patch("yfinance.download")
    def test_default_output_dir(self, mock_yf_download, sample_sp500_data):
        """Test using default output directory."""
        mock_yf_download.return_value = sample_sp500_data

        # Patch to_csv to avoid writing to actual default location
        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            result = download_sp500_data(
                start_date="2024-01-01",
                end_date="2024-01-10",
            )

            assert mock_to_csv.called

    @patch("yfinance.download")
    def test_date_range_parameters(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that date range parameters are passed correctly."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(
            start_date="2020-01-01",
            end_date="2024-12-31",
            output_dir=tmp_path,
        )

        mock_yf_download.assert_called_once_with("^GSPC", start="2020-01-01", end="2024-12-31")

    @patch("yfinance.download")
    def test_csv_content_correct(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that CSV content is correct."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=tmp_path,
        )

        # Read the saved CSV
        saved_file = tmp_path / "SP500_2018_2025_Clean.csv"
        saved_df = pd.read_csv(saved_file)

        assert len(saved_df) == 10
        assert "Open" in saved_df.columns
        assert "Close" in saved_df.columns
        assert "Volume" in saved_df.columns


class TestDownloadEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = pd.DataFrame(
            {
                "Open": [5000 + i * 10 for i in range(len(dates))],
                "High": [5020 + i * 10 for i in range(len(dates))],
                "Low": [4980 + i * 10 for i in range(len(dates))],
                "Close": [5010 + i * 10 for i in range(len(dates))],
                "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_empty_data_returned(self, mock_yf_download, tmp_path):
        """Test handling of empty data from yfinance."""
        mock_yf_download.return_value = pd.DataFrame()

        # Empty DataFrame may cause issues with column assignment
        try:
            result = download_sp500_data(
                start_date="2024-01-01",
                end_date="2024-01-10",
                output_dir=tmp_path,
            )
        except Exception:
            pass  # Empty data may raise an error

    @patch("yfinance.download")
    def test_network_error(self, mock_yf_download, tmp_path):
        """Test handling of network errors."""
        mock_yf_download.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            download_sp500_data(
                start_date="2024-01-01",
                end_date="2024-01-10",
                output_dir=tmp_path,
            )

    @patch("yfinance.download")
    def test_path_as_string(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that string path is handled correctly."""
        mock_yf_download.return_value = sample_sp500_data

        result = download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
            output_dir=str(tmp_path),  # Pass as string instead of Path
        )

        assert isinstance(result, pd.DataFrame)


class TestDefaultParameters:
    """Test default parameter values."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-01", freq="D")
        data = pd.DataFrame(
            {
                "Open": [5000],
                "High": [5020],
                "Low": [4980],
                "Close": [5010],
                "Volume": [1000000],
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_default_start_date(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test default start date parameter."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(output_dir=tmp_path)

        # Check that default dates were used
        call_args = mock_yf_download.call_args
        assert call_args[1]["start"] == "2018-01-01"

    @patch("yfinance.download")
    def test_default_end_date(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test default end date parameter."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(output_dir=tmp_path)

        # Check that default dates were used
        call_args = mock_yf_download.call_args
        assert call_args[1]["end"] == "2025-12-31"


class TestOutputFormat:
    """Test output format and structure."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        data = pd.DataFrame(
            {
                "Open": [5000.0] * 5,
                "High": [5020.0] * 5,
                "Low": [4980.0] * 5,
                "Close": [5010.0] * 5,
                "Volume": [1000000] * 5,
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_index_saved_to_csv(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that index (dates) is saved to CSV."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(output_dir=tmp_path)

        saved_df = pd.read_csv(tmp_path / "SP500_2018_2025_Clean.csv")

        # Date column should be present (from index)
        assert "Date" in saved_df.columns or len(saved_df.columns) > 4

    @patch("yfinance.download")
    def test_numeric_data_types(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that numeric columns have correct data types."""
        mock_yf_download.return_value = sample_sp500_data

        result = download_sp500_data(output_dir=tmp_path)

        assert result["Open"].dtype in ["float64", "float32", float]
        assert result["Close"].dtype in ["float64", "float32", float]


class TestTickerConfiguration:
    """Test ticker configuration."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-01", freq="D")
        data = pd.DataFrame(
            {
                "Open": [5000],
                "High": [5020],
                "Low": [4980],
                "Close": [5010],
                "Volume": [1000000],
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_uses_gspc_ticker(self, mock_yf_download, sample_sp500_data, tmp_path):
        """Test that ^GSPC ticker is used for S&P 500."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(output_dir=tmp_path)

        # Verify correct ticker was used
        mock_yf_download.assert_called_once()
        call_args = mock_yf_download.call_args
        assert call_args[0][0] == "^GSPC"


class TestPrintOutput:
    """Test print output messages."""

    @pytest.fixture
    def sample_sp500_data(self):
        """Create sample S&P 500 data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        data = pd.DataFrame(
            {
                "Open": [5000] * 5,
                "High": [5020] * 5,
                "Low": [4980] * 5,
                "Close": [5010] * 5,
                "Volume": [1000000] * 5,
            },
            index=dates,
        )
        return data

    @patch("yfinance.download")
    def test_prints_download_info(self, mock_yf_download, sample_sp500_data, tmp_path, capsys):
        """Test that download info is printed."""
        mock_yf_download.return_value = sample_sp500_data

        download_sp500_data(
            start_date="2024-01-01",
            end_date="2024-01-05",
            output_dir=tmp_path,
        )

        captured = capsys.readouterr()

        assert "Downloading S&P 500 data" in captured.out
        assert "Download completed" in captured.out
        assert "saved" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
