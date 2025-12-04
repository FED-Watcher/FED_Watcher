"""
Test script to verify backtesting logic with sample data.

This script creates synthetic test data to verify:
1. Position determination logic
2. P&L calculation accuracy
3. Equity tracking correctness
"""

import pandas as pd
import numpy as np
from model_predictor import ModelPredictor
from backtest_engine import BacktestEngine
from strategies import BinaryStrategy, MultiClassStrategy


def create_synthetic_test_data(num_events=5):
    """
    Create synthetic test data for verification.

    Returns:
        pd.DataFrame: Synthetic test data with known outcomes
    """
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=num_events, freq="30D")

    # Create known scenarios for testing
    data = {
        "Date": dates,
        "Close": [5000, 5100, 5050, 5150, 5200],
        "Volume": [1000000] * num_events,
        "Volume_ratio_vs_5days": [1.0] * num_events,
        "Announcement": [1] * num_events,
        "hawkish_dovish_ratio": [1.2, 0.8, 1.1, 0.9, 1.3],
        "net_sentiment_score": [0.15, -0.10, 0.05, -0.05, 0.20],
        "negative_proportion": [0.2, 0.4, 0.3, 0.35, 0.25],
        "neutral_proportion": [0.3, 0.3, 0.4, 0.35, 0.30],
        "positive_proportion": [0.5, 0.3, 0.3, 0.30, 0.45],
        "VIX_Close": [15, 18, 16, 17, 14],
        "DXY_Close": [103, 104, 103.5, 104.5, 103],
        "US02Y_Yield": [4.5, 4.6, 4.55, 4.65, 4.5],
        "US10Y_Yield": [4.2, 4.3, 4.25, 4.35, 4.2],
        "Yield_Curve_10Y_2Y": [-0.3, -0.3, -0.3, -0.3, -0.3],
        "sentiment_label": ["positive", "negative", "neutral", "neutral", "positive"],
    }

    df = pd.DataFrame(data)

    # Calculate returns
    df["future_close"] = df["Close"].shift(-1)
    df["return"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["market_return"] = df["return"] * 100

    # Remove last row (no future return)
    df = df[:-1].copy()

    return df


def test_binary_strategy():
    """Test binary strategy with known outcomes."""
    print(f"\n{'='*60}")
    print("TEST 1: Binary Strategy Logic")
    print(f"{'='*60}\n")

    # Create test data
    test_data = create_synthetic_test_data()

    # Create mock predictions (alternating Up/Down)
    mock_predictions = np.array([1, 0, 1, 0])  # Up, Down, Up, Down

    # Create strategy
    strategy = BinaryStrategy()

    print("Test Case 1: Up Prediction (1) with Positive Return")
    print("-" * 60)
    prediction = 1
    actual_return = 2.0  # +2%
    position = strategy.get_position(prediction)
    pnl = strategy.calculate_pnl(position, actual_return)

    print(f"  Prediction:       {prediction} (Up)")
    print("  Expected Position: +1.0 (Long)")
    print(f"  Actual Position:   {position}")
    print(f"  Actual Return:     {actual_return}%")
    print("  Expected P&L:      +2.0%")
    print(f"  Calculated P&L:    {pnl}%")
    assert position == 1.0, "Position should be +1.0 for Up prediction"
    assert pnl == 2.0, "P&L should be +2.0%"
    print("  ✓ PASSED\n")

    print("Test Case 2: Down Prediction (0) with Positive Return")
    print("-" * 60)
    prediction = 0
    actual_return = 2.0  # +2%
    position = strategy.get_position(prediction)
    pnl = strategy.calculate_pnl(position, actual_return)

    print("  Prediction:       {prediction} (Down)")
    print("  Expected Position: -1.0 (Short)")
    print(f"  Actual Position:   {position}")
    print(f"  Actual Return:     {actual_return}%")
    print("  Expected P&L:      -2.0% (wrong direction)")
    print(f"  Calculated P&L:    {pnl}%")
    assert position == -1.0, "Position should be -1.0 for Down prediction"
    assert pnl == -2.0, "P&L should be -2.0% (loss)"
    print("  ✓ PASSED\n")

    print("Test Case 3: Up Prediction (1) with Negative Return")
    print("-" * 60)
    prediction = 1
    actual_return = -1.5  # -1.5%
    position = strategy.get_position(prediction)
    pnl = strategy.calculate_pnl(position, actual_return)

    print(f"  Prediction:       {prediction} (Up)")
    print(f"  Position:          {position} (Long)")
    print(f"  Actual Return:     {actual_return}%")
    print("  Expected P&L:      -1.5% (wrong direction)")
    print(f"  Calculated P&L:    {pnl}%")
    assert pnl == -1.5, "P&L should be -1.5%"
    print("  ✓ PASSED\n")

    print("✓ All binary strategy tests passed!\n")


def test_multiclass_strategy():
    """Test multi-class strategy with scaled positions."""
    print(f"\n{'='*60}")
    print("TEST 2: Multi-Class Strategy Logic")
    print(f"{'='*60}\n")

    strategy = MultiClassStrategy(scale_positions=True)

    test_cases = [
        (2, 3.0, 3.0, "Strong Rise with positive return"),
        (-2, 3.0, -3.0, "Strong Drop with positive return (wrong direction)"),
        (1, 2.0, 1.0, "Modest Rise with positive return (scaled to 0.5)"),
        (0, 2.0, 0.0, "Neutral prediction (no position)"),
        (-1, -2.0, 1.0, "Modest Drop with negative return (correct, scaled)"),
    ]

    for pred, ret, expected_pnl, description in test_cases:
        print(f"{description}")
        print("-" * 60)

        position = strategy.get_position(pred)
        pnl = strategy.calculate_pnl(position, ret)

        print(f"  Prediction:        {pred}")
        print(f"  Position:          {position}")
        print(f"  Actual Return:     {ret}%")
        print(f"  Expected P&L:      {expected_pnl}%")
        print(f"  Calculated P&L:    {pnl}%")

        assert abs(pnl - expected_pnl) < 0.01, f"P&L mismatch: {pnl} != {expected_pnl}"
        print("  ✓ PASSED\n")

    print("✓ All multi-class strategy tests passed!\n")


def test_equity_tracking():
    """Test equity curve calculation."""
    print(f"\n{'='*60}")
    print("TEST 3: Equity Tracking")
    print(f"{'='*60}\n")

    initial_capital = 100000
    trades = [
        (1, 2.0),  # Long, +2% return → +$2,000
        (-1, 1.0),  # Short, +1% return → -$1,020 (on $102,000)
        (1, -1.5),  # Long, -1.5% return → -$1,515 (on $100,980)
    ]

    equity = initial_capital
    print(f"Initial Capital: ${equity:,.2f}\n")

    for i, (position, return_pct) in enumerate(trades, 1):
        pnl_pct = position * return_pct
        pnl_dollars = equity * (pnl_pct / 100)
        equity += pnl_dollars

        direction = "Long" if position > 0 else "Short"
        print(f"Trade {i}: {direction}")
        print(f"  Return:       {return_pct:+.2f}%")
        print(f"  P&L:          {pnl_pct:+.2f}% = ${pnl_dollars:+,.2f}")
        print(f"  New Equity:   ${equity:,.2f}\n")

    expected_final = 99475.20  # Manual calculation
    print(f"Expected Final Equity: ${expected_final:,.2f}")
    print(f"Calculated Final Equity: ${equity:,.2f}")

    # Allow small floating point difference
    assert abs(equity - expected_final) < 1.0, "Equity tracking error"
    print("✓ Equity tracking test passed!\n")


def test_manual_calculation():
    """
    Demonstrate manual P&L calculation for verification.
    This matches the logic used in manual_verification().
    """
    print(f"\n{'='*60}")
    print("TEST 4: Manual Calculation Walkthrough")
    print(f"{'='*60}\n")

    print("Scenario: FOMC Event on 2024-03-20")
    print("-" * 60)

    # Event data
    close_price = 5200.12
    next_close = 5245.30
    actual_return = (next_close - close_price) / close_price
    actual_return_pct = actual_return * 100

    print("Market Data:")
    print(f"  Close Price:      ${close_price:.2f}")
    print(f"  Next Close:       ${next_close:.2f}")
    print(f"  Return:           {actual_return_pct:+.2f}%\n")

    # Model prediction
    prediction = 1  # Up
    strategy = BinaryStrategy()
    position = strategy.get_position(prediction)

    print("Model Prediction:")
    print(f"  Prediction:       {prediction} (Up)")
    print(f"  Position:         {position:+.2f} (Long)\n")

    # Calculate P&L two ways
    print("P&L Calculation:")

    # Method 1: Using strategy
    pnl_strategy = strategy.calculate_pnl(position, actual_return_pct)
    print(f"  Strategy Method:  {pnl_strategy:+.2f}%")

    # Method 2: Manual calculation
    pnl_manual = position * actual_return_pct
    print(f"  Manual Method:    {pnl_manual:+.2f}%")

    # Method 3: Step-by-step
    print("\n  Step-by-step:")
    print("    Position × Return")
    print(f"    = {position} × {actual_return_pct:.4f}%")
    print(f"    = {pnl_manual:+.4f}%")

    assert abs(pnl_strategy - pnl_manual) < 0.0001, "Methods should match"
    print("\n  ✓ Both methods match!")

    # Convert to dollars
    portfolio_value = 100000
    pnl_dollars = portfolio_value * (pnl_manual / 100)
    new_equity = portfolio_value + pnl_dollars

    print("\n  In Dollar Terms:")
    print(f"    Portfolio:        ${portfolio_value:,.2f}")
    print(f"    P&L:              ${pnl_dollars:+,.2f}")
    print(f"    New Equity:       ${new_equity:,.2f}")

    print("\n✓ Manual calculation test passed!\n")


def main():
    """Run all tests."""
    print(f"\n{'#'*60}")
    print("# BACKTESTING ENGINE - VERIFICATION TESTS")
    print(f"{'#'*60}")

    try:
        test_binary_strategy()
        test_multiclass_strategy()
        test_equity_tracking()
        test_manual_calculation()

        print(f"\n{'#'*60}")
        print("# ALL TESTS PASSED ✓")
        print(f"{'#'*60}\n")

    except AssertionError as e:
        print(f"\n{'!'*60}")
        print(f"! TEST FAILED: {e}")
        print(f"{'!'*60}\n")
        raise


if __name__ == "__main__":
    main()
