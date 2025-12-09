import pandas as pd
import numpy as np
import random
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default paths relative to project root
SP500_PATH = PROJECT_ROOT / "data" / "processed" / "stock_data" / "SP500_2018_2025_Clean.csv"
SENTIMENT_PATH = PROJECT_ROOT / "data" / "sentiment_results" / "semantic_document_sentiments.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "MasterDataset_Final.csv"


def run_pipeline(sp500_path=None, sentiment_path=None, output_path=None):
    """
    Combine S&P 500 stock data with Powell speech sentiment analysis results.

    Args:
        sp500_path (Path): Path to S&P 500 CSV file
        sentiment_path (Path): Path to sentiment analysis results CSV
        output_path (Path): Path to save the master dataset
    """
    if sp500_path is None:
        sp500_path = SP500_PATH
    if sentiment_path is None:
        sentiment_path = SENTIMENT_PATH
    if output_path is None:
        output_path = OUTPUT_PATH

    print("=" * 70)
    print("MASTER PIPELINE - Combining Stock Data with Sentiment Analysis")
    print("=" * 70)
    print(f"S&P 500 data: {sp500_path}")
    print(f"Sentiment data: {sentiment_path}")
    print(f"Output: {output_path}")
    print()

    # 1. Load datasets
    print("1. Loading datasets...")
    sp500 = pd.read_csv(sp500_path)
    powell = pd.read_csv(sentiment_path)
    print(f"   S&P 500 rows: {len(sp500)}")
    print(f"   Sentiment rows: {len(powell)}")

    # 2. Parse dates
    print("2. Parsing dates...")
    # S&P 500 from yfinance uses YYYY-MM-DD format
    sp500["__DATE__"] = pd.to_datetime(sp500["Date"], format="%Y-%m-%d", errors="coerce").dt.date
    powell["__DATE__"] = pd.to_datetime(
        powell["meeting_date"], format="%Y-%m-%d", errors="coerce"
    ).dt.date

    # 3. Calculate sentiment proportions from chunk counts
    print("3. Processing sentiment features...")
    total_chunks = (
        powell["positive_chunk_count"]
        + powell["negative_chunk_count"]
        + powell["neutral_chunk_count"]
    )

    powell["positive_proportion"] = powell["positive_chunk_count"] / total_chunks
    powell["negative_proportion"] = powell["negative_chunk_count"] / total_chunks
    powell["neutral_proportion"] = powell["neutral_chunk_count"] / total_chunks

    # Calculate hawkish/dovish ratio (negative = hawkish, positive = dovish)
    # Avoid division by zero
    powell["hawkish_dovish_ratio"] = np.where(
        powell["negative_chunk_count"] > 0,
        powell["positive_chunk_count"] / powell["negative_chunk_count"],
        powell["positive_chunk_count"] + 1,  # If no negative, use positive + 1
    )

    # Keep only necessary Powell columns
    powell_trim = powell[
        [
            "__DATE__",
            "hawkish_dovish_ratio",
            "net_sentiment_score",
            "negative_proportion",
            "neutral_proportion",
            "positive_proportion",
        ]
    ].copy()

    # 4. Merge (explicit left join - everything from SP500 stays)
    print("4. Merging datasets...")
    master = sp500.merge(powell_trim, how="left", on="__DATE__")
    print(f"   Merged dataset rows: {len(master)}")

    # 5. Create Announcement flag
    master["Announcement"] = np.where(master["net_sentiment_score"].notna(), 1, 0)
    announcement_count = master["Announcement"].sum()
    print(f"   Announcement days: {announcement_count}")

    # 6. Calculate 5-day Volume ratio
    print("5. Calculating volume ratio...")
    master = master.sort_values("__DATE__")
    master["Volume_ratio_vs_5days"] = (
        master["Volume"] / master["Volume"].rolling(5, min_periods=5).mean()
    )
    mask = master["Volume_ratio_vs_5days"].isna()
    master.loc[mask, "Volume_ratio_vs_5days"] = [
        random.uniform(0.8, 1.2) for _ in range(mask.sum())
    ]

    # 7. Sentiment label (1 = pos, 0 = neutral, -1 = neg)
    print("6. Assigning sentiment labels...")

    def sentiment_label(row):
        p, n, u = row["positive_proportion"], row["negative_proportion"], row["neutral_proportion"]
        if pd.isna(p) or pd.isna(n) or pd.isna(u):
            return np.nan
        if p > n and p > u:
            return 1
        elif u > p and u > n:
            return 0
        elif n > p and n > u:
            return -1
        return np.nan

    master["sentiment_label"] = master.apply(sentiment_label, axis=1)

    # 8. Format date for output (DD/MM/YYYY as expected by downstream scripts)
    master["Date"] = pd.to_datetime(master["Date"]).dt.strftime("%d/%m/%Y")

    # 9. Reorder and fill missing values
    master = master[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Volume_ratio_vs_5days",
            "Announcement",
            "hawkish_dovish_ratio",
            "net_sentiment_score",
            "negative_proportion",
            "neutral_proportion",
            "positive_proportion",
            "sentiment_label",
        ]
    ]

    # Fill NaNs in numeric Powell columns with 0
    numeric_cols = [
        "hawkish_dovish_ratio",
        "net_sentiment_score",
        "negative_proportion",
        "neutral_proportion",
        "positive_proportion",
    ]
    master[numeric_cols] = master[numeric_cols].fillna(0)

    # 10. Save as CSV
    print(f"7. Saving master dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print("MASTER PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total rows: {len(master)}")
    print(f"Announcement days: {announcement_count}")
    print()
    print("First 10 rows:")
    print(master.head(10).to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
