import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Configuration ---
MASTER_FILE_PATH = PROJECT_ROOT / "data" / "MasterDataset_Final.csv"
ENRICHED_OUTPUT_PATH = PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv"


def enrich_dataset(master_path=None, output_path=None):
    """
    Main pipeline to load the master dataset, fetch additional market data,
    and merge them into a new, enriched dataset.

    Args:
        master_path (Path): Path to master dataset CSV
        output_path (Path): Path to save enriched dataset
    """
    if master_path is None:
        master_path = MASTER_FILE_PATH
    if output_path is None:
        output_path = ENRICHED_OUTPUT_PATH

    print("=" * 80)
    print("Starting Market Data Enrichment Pipeline")
    print("=" * 80)

    # --- Step 1: Load Existing Master Dataset ---
    print(f"1. Loading existing master dataset from '{master_path}'...")
    try:
        master_df = pd.read_csv(master_path)
    except FileNotFoundError:
        print(
            f"ERROR: Master file not found at '{master_path}'. "
            "Please run master_pipeline.py first."
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
    print("\n2. Fetching VIX, 10Y Yield, and DXY data from Yahoo Finance...")

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
        print(f"ERROR: Failed to download data from Yahoo Finance. Error: {e}")
        return

    # --- Step 3: Fetch 2Y Treasury Yield from FRED ---
    print("\n3. Fetching 2Y Treasury Yield data from FRED...")

    # FRED ticker for 2-Year Treasury is 'DGS2'
    try:
        us02y_yield = web.DataReader("DGS2", "fred", start_date, end_date)
        us02y_yield.rename(columns={"DGS2": "US02Y_Yield"}, inplace=True)
        print("   - Successfully fetched 2Y Treasury data.")
    except Exception as e:
        print(f"ERROR: Failed to download data from FRED. Error: {e}")
        return

    # --- Step 4: Combine Market Data and Create Yield Curve ---
    print("\n4. Combining all market data and creating derived features...")

    # Combine the two data sources
    all_market_data = pd.concat([market_data_yf_clean, us02y_yield], axis=1)

    # The Yield Curve: 10-Year Yield minus 2-Year Yield
    all_market_data["Yield_Curve_10Y_2Y"] = (
        all_market_data["US10Y_Yield"] - all_market_data["US02Y_Yield"]
    )
    print("   - Calculated 'Yield_Curve_10Y_2Y' feature.")

    # --- Step 5: Merge Market Data into Master Dataset ---
    print("\n5. Merging new market data with the master dataset...")

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
    print(f"\n6. Saving enriched dataset to '{output_path}'...")

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

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    # --- Step 7: Generate Summary Statistics ---
    print("\n7. Generating summary statistics...")

    # Announcement statistics
    announcement_days = final_df[final_df["Announcement"] == 1]
    non_announcement_days = final_df[final_df["Announcement"] == 0]

    # Sentiment distribution on announcement days
    sentiment_counts = announcement_days["sentiment_label"].value_counts().sort_index()
    sentiment_labels = {1.0: "Positive", 0.0: "Neutral", -1.0: "Negative"}

    print("\n" + "=" * 80)
    print("ENRICHMENT PIPELINE COMPLETE")
    print("=" * 80)

    print(f"\nOutput: {output_path}")

    print("\n" + "-" * 40)
    print("DATASET OVERVIEW")
    print("-" * 40)
    print(f"  Total trading days:      {len(final_df):,}")
    print(
        f"  Date range:              {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"  Total columns:           {len(final_df.columns)}")

    print("\n" + "-" * 40)
    print("ANNOUNCEMENT STATISTICS")
    print("-" * 40)
    print(f"  Fed announcement days:   {len(announcement_days)}")
    print(f"  Non-announcement days:   {len(non_announcement_days)}")
    print(f"  Announcement ratio:      {len(announcement_days) / len(final_df) * 100:.2f}%")

    print("\n" + "-" * 40)
    print("SENTIMENT DISTRIBUTION (Announcement Days)")
    print("-" * 40)
    for label_val, count in sentiment_counts.items():
        label_name = sentiment_labels.get(label_val, f"Unknown ({label_val})")
        pct = count / len(announcement_days) * 100
        print(f"  {label_name:12s}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "-" * 40)
    print("MARKET INDICATORS (All Days)")
    print("-" * 40)
    print(
        f"  VIX Range:               {final_df['VIX_Close'].min():.2f} - {final_df['VIX_Close'].max():.2f}"
    )
    print(f"  VIX Mean:                {final_df['VIX_Close'].mean():.2f}")
    print(
        f"  DXY Range:               {final_df['DXY_Close'].min():.2f} - {final_df['DXY_Close'].max():.2f}"
    )
    print(
        f"  US 2Y Yield Range:       {final_df['US02Y_Yield'].min():.2f}% - {final_df['US02Y_Yield'].max():.2f}%"
    )
    print(
        f"  US 10Y Yield Range:      {final_df['US10Y_Yield'].min():.2f}% - {final_df['US10Y_Yield'].max():.2f}%"
    )
    print(
        f"  Yield Curve Range:       {final_df['Yield_Curve_10Y_2Y'].min():.2f} - {final_df['Yield_Curve_10Y_2Y'].max():.2f}"
    )

    # Yield curve inversion stats
    inverted_days = (final_df["Yield_Curve_10Y_2Y"] < 0).sum()
    print(
        f"  Yield Curve Inversions:  {inverted_days} days ({inverted_days / len(final_df) * 100:.1f}%)"
    )

    print("\n" + "-" * 40)
    print("MARKET INDICATORS (Announcement Days Only)")
    print("-" * 40)
    print(f"  VIX Mean:                {announcement_days['VIX_Close'].mean():.2f}")
    print(f"  DXY Mean:                {announcement_days['DXY_Close'].mean():.2f}")
    print(f"  US 2Y Yield Mean:        {announcement_days['US02Y_Yield'].mean():.2f}%")
    print(f"  US 10Y Yield Mean:       {announcement_days['US10Y_Yield'].mean():.2f}%")
    print(f"  Yield Curve Mean:        {announcement_days['Yield_Curve_10Y_2Y'].mean():.2f}")

    print("\n" + "-" * 40)
    print("SENTIMENT SCORES (Announcement Days)")
    print("-" * 40)
    print(f"  Net Sentiment Mean:      {announcement_days['net_sentiment_score'].mean():.4f}")
    print(
        f"  Net Sentiment Range:     {announcement_days['net_sentiment_score'].min():.4f} to {announcement_days['net_sentiment_score'].max():.4f}"
    )
    print(
        f"  Hawkish/Dovish Ratio:    {announcement_days['hawkish_dovish_ratio'].mean():.2f} (mean)"
    )

    print("\n" + "-" * 40)
    print("S&P 500 STATISTICS")
    print("-" * 40)
    print(f"  Starting Price:          ${final_df['Close'].iloc[0]:,.2f}")
    print(f"  Ending Price:            ${final_df['Close'].iloc[-1]:,.2f}")
    total_return = (final_df["Close"].iloc[-1] / final_df["Close"].iloc[0] - 1) * 100
    print(f"  Total Return:            {total_return:+.2f}%")
    print(f"  Average Daily Volume:    {final_df['Volume'].mean():,.0f}")

    print("\n" + "=" * 80)
    print("Dataset ready for model training!")
    print("=" * 80)


if __name__ == "__main__":
    enrich_dataset()
