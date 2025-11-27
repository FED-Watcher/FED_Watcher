"""
Data preparation module for Fed Market Prediction project.
Handles loading the enriched dataset, creating target variables, and optional shuffling.
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load the enriched master dataset.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe with Date as index
    """
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date")
    df = df.set_index("Date")

    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def create_binary_target(df, horizon=1):
    """
    Create binary target variable for market direction prediction.
    Target = 1 if Close price increases after 'horizon' days, 0 otherwise.

    Args:
        df (pd.DataFrame): Input dataframe with Close prices
        horizon (int): Number of days ahead to predict

    Returns:
        pd.DataFrame: Dataframe with new 'target' column
    """
    df["future_close"] = df["Close"].shift(-horizon)
    df["return"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["target"] = (df["return"] > 0).astype(int)
    df_clean = df[:-horizon].copy()

    print("\nCreated binary target variable:")
    print(f"  Horizon: {horizon} day(s)")
    print(
        f"  Target distribution: Up={df_clean['target'].sum()} ({df_clean['target'].mean()*100:.1f}%), Down={(df_clean['target']==0).sum()}"
    )

    return df_clean


def select_features(df):
    """
    Select relevant features for modeling.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        tuple: (feature_columns list, df with selected columns)
    """
    exclude_cols = ["future_close", "return", "target", "Open", "High", "Low"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nSelected features: {len(feature_cols)} -> {feature_cols}")

    return feature_cols, df


def prepare_announcement_data(df):
    """
    Filter data to only include Fed announcement days.

    Args:
        df (pd.DataFrame): Input dataframe with Announcement column

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    announcement_data = df[df["Announcement"] == 1].copy()

    print("\nFiltered to announcement days:")
    print(f"  Total announcement days: {len(announcement_data)}")
    print(f"  Date range: {announcement_data.index.min()} to {announcement_data.index.max()}")

    return announcement_data


def split_data(df, feature_cols, test_size=0.2, random_state=42, shuffle_data=True):
    """
    Split data into train and test sets, optionally shuffling first.

    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (list): List of feature column names
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        shuffle_data (bool): If True, shuffle before splitting

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[feature_cols]
    y = df["target"]

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\nSplit data:")
    print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def prepare_pipeline(
    filepath, use_announcement_only=True, horizon=1, test_size=0.2, shuffle_data=True
):
    """
    Complete data preparation pipeline.

    Args:
        filepath (str): Path to the enriched dataset
        use_announcement_only (bool): If True, only use announcement days
        horizon (int): Prediction horizon in days
        test_size (float): Test set proportion
        shuffle_data (bool): If True, shuffle data before splitting

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols)
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    df = load_data(filepath)
    df = create_binary_target(df, horizon=horizon)

    if use_announcement_only:
        df = prepare_announcement_data(df)

    feature_cols, df = select_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df, feature_cols, test_size=test_size, shuffle_data=shuffle_data
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(X_train.head(20))
    return X_train, X_test, y_train, y_test, feature_cols
