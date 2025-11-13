"""
Data preparation module for Fed Market Prediction project.
Handles loading the enriched dataset and creating target variables.
"""

import pandas as pd
import numpy as np
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
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    print(f"✓ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def create_binary_target(df, horizon=1):
    """
    Create binary target variable for market direction prediction.
    Target = 1 if Close price increases after 'horizon' days, 0 otherwise.
    
    Args:
        df (pd.DataFrame): Input dataframe with Close prices
        horizon (int): Number of days ahead to predict (default=1 for 24h)
        
    Returns:
        pd.DataFrame: Dataframe with new 'target' column
    """
    # Calculate future return
    df['future_close'] = df['Close'].shift(-horizon)
    df['return'] = (df['future_close'] - df['Close']) / df['Close']
    
    # Create binary target: 1 if positive return, 0 otherwise
    df['target'] = (df['return'] > 0).astype(int)
    
    # Remove rows where we can't calculate target (last 'horizon' rows)
    df_clean = df[:-horizon].copy()
    
    print(f"\n✓ Created binary target variable:")
    print(f"  Horizon: {horizon} day(s)")
    print(f"  Target distribution:")
    print(f"    Up (1): {df_clean['target'].sum()} ({df_clean['target'].mean()*100:.1f}%)")
    print(f"    Down (0): {(df_clean['target']==0).sum()} ({(1-df_clean['target'].mean())*100:.1f}%)")
    
    return df_clean


def select_features(df):
    """
    Select relevant features for modeling.
    Excludes non-predictive columns and forward-looking data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (feature_columns list, df with selected columns)
    """
    # Features to exclude (target-related and helper columns)
    exclude_cols = ['future_close', 'return', 'target']
    
    # Also exclude OHLC except Close (to avoid multicollinearity)
    # Keep Close as it's the base for our predictions
    exclude_cols.extend(['Open', 'High', 'Low'])
    
    # Select all other columns as features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n✓ Selected features: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")
    
    return feature_cols, df


def prepare_announcement_data(df):
    """
    Filter data to only include Fed announcement days and next day.
    This focuses the model on post-announcement market reactions.
    
    Args:
        df (pd.DataFrame): Input dataframe with Announcement column
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Keep only rows where Announcement = 1
    announcement_data = df[df['Announcement'] == 1].copy()
    
    print(f"\n✓ Filtered to announcement days:")
    print(f"  Total announcement days: {len(announcement_data)}")
    print(f"  Date range: {announcement_data.index.min()} to {announcement_data.index.max()}")
    
    return announcement_data


def split_data(df, feature_cols, test_size=0.2, random_state=42):
    """
    Split data into train and test sets chronologically.
    Uses chronological split to prevent data leakage.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (list): List of feature column names
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Prepare features and target
    X = df[feature_cols]
    y = df['target']
    
    # Chronological split (don't shuffle time series data)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n✓ Split data chronologically:")
    print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"    Date range: {X_train.index.min()} to {X_train.index.max()}")
    print(f"    Target distribution: Up={y_train.sum()} ({y_train.mean()*100:.1f}%), Down={len(y_train)-y_train.sum()}")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"    Date range: {X_test.index.min()} to {X_test.index.max()}")
    print(f"    Target distribution: Up={y_test.sum()} ({y_test.mean()*100:.1f}%), Down={len(y_test)-y_test.sum()}")
    
    return X_train, X_test, y_train, y_test


def prepare_pipeline(filepath, use_announcement_only=True, horizon=1, test_size=0.2):
    """
    Complete data preparation pipeline.
    
    Args:
        filepath (str): Path to the enriched dataset
        use_announcement_only (bool): If True, only use announcement days
        horizon (int): Prediction horizon in days
        test_size (float): Test set proportion
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols)
    """
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(filepath)
    
    # Step 2: Create binary target
    df = create_binary_target(df, horizon=horizon)
    
    # Step 3: Filter to announcement days (optional)
    if use_announcement_only:
        df = prepare_announcement_data(df)
    
    # Step 4: Select features
    feature_cols, df = select_features(df)
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, test_size=test_size)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, feature_cols
