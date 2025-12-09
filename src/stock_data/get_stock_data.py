import yfinance as yf
import os
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default output path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stock_data"


def download_sp500_data(start_date="2018-01-01", end_date="2025-12-31", output_dir=None):
    """
    Download S&P 500 data from Yahoo Finance and save to CSV.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (Path): Output directory for CSV file

    Returns:
        pd.DataFrame: Downloaded S&P 500 data
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker = "^GSPC"

    print(f"Downloading S&P 500 data ({ticker}) from {start_date} to {end_date}...")
    sp500_data = yf.download(ticker, start=start_date, end=end_date)
    print("[Download completed]")

    # Clean and Simplify Column Names
    sp500_data.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Save the Data to a CSV File
    file_name = output_dir / "SP500_2018_2025_Clean.csv"
    sp500_data.to_csv(file_name, index=True)

    print("-" * 50)
    print(f"Clean dataset successfully saved as: {file_name}")
    print("\nFirst 5 rows of the dataset:")
    print(sp500_data.head())
    print("-" * 50)

    return sp500_data


if __name__ == "__main__":
    download_sp500_data()
