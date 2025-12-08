import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Feature Definitions
SENTIMENT_FEATURES = [
    "hawkish_dovish_ratio",
    "net_sentiment_score",
    "negative_proportion",
    "neutral_proportion",
    "positive_proportion",
]
MACRO_FEATURES = ["VIX_Close", "DXY_Close", "US02Y_Yield", "US10Y_Yield", "Yield_Curve_10Y_2Y"]
CONTEXTUAL_FEATURES = ["Volume_ratio_vs_5days"]
ALL_FEATURES = SENTIMENT_FEATURES + MACRO_FEATURES + CONTEXTUAL_FEATURES


def apply_thresholds(return_pct, t):
    """
    Maps return percentage to class ID (-2 to 2).
    """
    if return_pct > t["strong"]:
        return 2  # Strong Rise
    elif return_pct > t["modest"]:
        return 1  # Modest Rise
    elif return_pct >= -t["modest"]:
        return 0  # Neutral
    elif return_pct >= -t["strong"]:
        return -1  # Modest Drop
    else:
        return -2  # Strong Drop


def load_raw_data(filepath):
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Filter for announcements
    df = df[df["Announcement"] == 1].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate Returns
    df["Next_Close"] = df["Close"].shift(-1)
    df["market_return"] = (df["Next_Close"] - df["Close"]) / df["Close"] * 100

    # Drop invalid rows
    df = df.dropna(subset=ALL_FEATURES + ["market_return"])
    return df


def prepare_data_with_thresholds(df_raw, threshold_config, test_size_pct=0.3):
    df = df_raw.copy()

    # 1. Create Raw Target (-2 to 2)
    raw_target = df["market_return"].apply(lambda x: apply_thresholds(x, threshold_config))

    # 2. Map to 0-4 for XGBoost (CRITICAL STEP)
    # -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3, 2 -> 4
    class_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    df["target"] = raw_target.map(class_mapping)

    # Split Chronologically
    split_idx = int(len(df) * (1 - test_size_pct))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Prepare X and y
    X_train_raw = train_df[ALL_FEATURES].values
    y_train = train_df["target"].values
    X_test_raw = test_df[ALL_FEATURES].values
    y_test = test_df["target"].values

    # Scale Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Compute Sample Weights
    try:
        sample_weights = compute_sample_weight("balanced", y_train)
    except ValueError:
        sample_weights = np.ones(len(y_train))

    return X_train, X_test, y_train, y_test, scaler, sample_weights, ALL_FEATURES
