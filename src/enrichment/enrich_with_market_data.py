import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

# --- Configuration ---
MASTER_FILE_PATH = "MasterDataset_Final.csv"
ENRICHED_OUTPUT_PATH = "MasterDataset_Enriched.csv"  # Saving as CSV is standard practice


def enrich_dataset():
    """
    Main pipeline to load the master dataset, fetch additional market data,
    and merge them into a new, enriched dataset.
    """
    print("=" * 80)
    print("🚀 Starting Market Data Enrichment Pipeline")
    print("=" * 80)

    # --- Step 1: Load Existing Master Dataset ---
    print(f"📁 1. Loading existing master dataset from '{MASTER_FILE_PATH}'...")
    try:
        master_df = pd.read_csv(MASTER_FILE_PATH)
    except FileNotFoundError:
        print(
            f"❌ ERROR: Master file not found at '{MASTER_FILE_PATH}'. Please ensure the file is in the correct directory."
        )
        return

    # Convert date column to datetime objects for proper merging
    master_df["Date_dt"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y")

    # Determine the date range needed
    start_date = master_df["Date_dt"].min()
    end_date = master_df["Date_dt"].max()
    print(
        f"   - Date range identified: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    # --- Step 2: Fetch Market Data from yfinance ---
    print("\n🌐 2. Fetching VIX, 10Y Yield, and DXY data from Yahoo Finance...")

    # Tickers for Yahoo Finance
    yf_tickers = {
        "^VIX": "VIX_Close",
        "^TNX": "US10Y_Yield",  # 10-Year Treasury Note Yield
        "DX-Y.NYB": "DXY_Close",  # US Dollar Index
    }

    try:
        market_data_yf = yf.download(list(yf_tickers.keys()), start=start_date, end=end_date)

        # Select and rename the 'Close' price columns
        market_data_yf_clean = market_data_yf["Close"].copy()
        market_data_yf_clean.rename(columns=yf_tickers, inplace=True)
        print("   - Successfully fetched Yahoo Finance data.")

    except Exception as e:
        print(f"❌ ERROR: Failed to download data from Yahoo Finance. Error: {e}")
        return

    # --- Step 3: Fetch 2Y Treasury Yield from FRED ---
    print("\n🏛️ 3. Fetching 2Y Treasury Yield data from FRED...")

    # FRED ticker for 2-Year Treasury is 'DGS2'
    try:
        us02y_yield = web.DataReader("DGS2", "fred", start_date, end_date)
        us02y_yield.rename(columns={"DGS2": "US02Y_Yield"}, inplace=True)
        print("   - Successfully fetched 2Y Treasury data.")
    except Exception as e:
        print(f"❌ ERROR: Failed to download data from FRED. Error: {e}")
        return

    # --- Step 4: Combine Market Data and Create Yield Curve ---
    print("\n🔗 4. Combining all market data and creating derived features...")

    # Combine the two data sources
    all_market_data = pd.concat([market_data_yf_clean, us02y_yield], axis=1)

    # The Yield Curve: 10-Year Yield minus 2-Year Yield
    all_market_data["Yield_Curve_10Y_2Y"] = (
        all_market_data["US10Y_Yield"] - all_market_data["US02Y_Yield"]
    )
    print("   - Calculated 'Yield_Curve_10Y_2Y' feature.")

    # --- Step 5: Merge Market Data into Master Dataset ---
    print("\n🔄 5. Merging new market data with the master dataset...")

    # Use a left merge to keep all original rows from the master dataset
    # Merging on the master's datetime column and the market data's index (which is also datetime)
    enriched_df = pd.merge(
        master_df, all_market_data, how="left", left_on="Date_dt", right_index=True
    )

    print("   - Initial merge complete. Handling non-trading days...")

    # Forward-fill NaNs for weekends/holidays
    # This assumes the value from the last trading day holds until the next one
    cols_to_fill = ["VIX_Close", "US10Y_Yield", "DXY_Close", "US02Y_Yield", "Yield_Curve_10Y_2Y"]
    enriched_df[cols_to_fill] = enriched_df[cols_to_fill].ffill()

    # Handle any potential NaNs at the very beginning of the dataset
    enriched_df[cols_to_fill] = enriched_df[cols_to_fill].bfill()

    print("   - Missing values handled.")

    # --- Step 6: Final Cleanup and Save ---
    print(f"\n💾 6. Saving enriched dataset to '{ENRICHED_OUTPUT_PATH}'...")

    # Reorder columns for clarity and remove the temporary date column
    final_cols_order = [
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
        "Yield_Curve_10Y_2Y",  # New Columns
        "Volume_ratio_vs_5days",
        "Announcement",
        "hawkish_dovish_ratio",
        "net_sentiment_score",
        "negative_proportion",
        "neutral_proportion",
        "positive_proportion",
        "sentiment_label",
    ]

    # Ensure all new columns are present before reordering
    final_df = enriched_df[final_cols_order].copy()

    final_df.to_csv(ENRICHED_OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("✅ Enrichment Pipeline Complete!")
    print("=" * 80)
    print("\n📊 Final Dataset Preview (first 10 rows):")
    print(final_df.head(10).to_string())

    print("\n🔍 Final Dataset Info:")
    final_df.info()


if __name__ == "__main__":
    enrich_dataset()
