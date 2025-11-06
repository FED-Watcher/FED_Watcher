import yfinance as yf
import pandas as pd
import os

# 1. Define Parameters
ticker = "^GSPC"
start_date = "2018-01-01"
end_date = "2025-01-01"  # Use your desired end date (or leave blank for today)

# 2. Download the Data
# Note: yf.download() uses auto_adjust=True by default, which means
# it only returns 5 columns: Open, High, Low, Close, and Volume.
print(f"Downloading S&P 500 data ({ticker}) from {start_date} to {end_date}...")
sp500_data = yf.download(ticker, start=start_date, end=end_date)
print("[Download completed]")

# 3. Clean and Simplify Column Names
# We must use exactly 5 names to match the 5 columns downloaded
# The order returned by yfinance is: Open, High, Low, Close, Volume
sp500_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# 4. Save the Data to a CSV File
file_name = "SP500_2018_2025_Clean.csv"
# index=True saves the 'Date' (which is the index) as the first column
sp500_data.to_csv(file_name, index=True)

print("-" * 50)
print(f"Clean dataset successfully saved as: {os.path.abspath(file_name)}")
print("\nFirst 5 rows of the dataset:")
print(sp500_data.head())
print("-" * 50)