"""
Main script to run backtesting on FED Watcher models.

This script demonstrates:
1. Loading trained models (binary and multi-class)
2. Running backtest simulations
3. Manual verification on specific FOMC events
4. Generating performance reports and visualizations
"""

import sys
import pandas as pd
from pathlib import Path

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Add project root to path for imports (must be before src imports)
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.model_predictor import ModelPredictor
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategies import (
    BinaryStrategy,
    BinaryLongOnlyStrategy,
    MultiClassStrategy,
)
from src.backtesting.report_generator import generate_html_report


def load_test_data(data_path=None, announcement_only=True):
    """
    Load and prepare test data for backtesting.

    Args:
        data_path (str): Path to enriched dataset
        announcement_only (bool): Filter to announcement days only

    Returns:
        tuple: (train_data, test_data)
    """
    print(f"\n{'=' * 60}")
    print("LOADING TEST DATA")
    print(f"{'=' * 60}")

    # Use default path if not provided
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "MasterDataset_Enriched.csv"
    else:
        data_path = Path(data_path)

    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"[OK] Dataset loaded: {len(df)} rows")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Filter to announcement days if requested
    if announcement_only:
        df = df[df["Announcement"] == 1].copy()
        print(f"[OK] Filtered to announcement days: {len(df)} events")

    # Calculate forward returns
    df["future_close"] = df["Close"].shift(-1)
    df["return"] = (df["future_close"] - df["Close"]) / df["Close"]

    # Remove last row (no future return)
    df = df[:-1].copy()

    # Calculate market return percentage for multi-class
    df["market_return"] = df["return"] * 100

    print("[OK] Forward returns calculated")

    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()

    print(f"\nTrain set: {len(train_data)} events")
    print(f"  Date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Test set: {len(test_data)} events")
    print(f"  Date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"{'=' * 60}\n")

    return train_data, test_data


def manual_verification(test_data, predictor, strategy, num_events=3):
    """
    Manually verify backtest logic on specific events.

    Args:
        test_data (pd.DataFrame): Test dataset
        predictor (ModelPredictor): Model predictor
        strategy (Strategy): Trading strategy
        num_events (int): Number of events to verify
    """
    print(f"\n{'=' * 60}")
    print(f"MANUAL VERIFICATION: Testing {num_events} FOMC Events")
    print(f"{'=' * 60}\n")

    # Select events to verify (first, middle, last)
    if len(test_data) >= num_events:
        indices = [0, len(test_data) // 2, -1][:num_events]
    else:
        indices = range(len(test_data))

    for idx in indices:
        event = test_data.iloc[idx]
        event_df = pd.DataFrame([event])

        # Get prediction
        prediction = predictor.predict(event_df)[0]
        pred_label = predictor.get_prediction_label(prediction)

        # Get position from strategy
        position = strategy.get_position(prediction)

        # Calculate P&L
        actual_return_pct = event["return"] * 100
        pnl_pct = strategy.calculate_pnl(position, actual_return_pct)

        print(f"Event {idx + 1}: {event['Date'].strftime('%Y-%m-%d')}")
        print(f"{'-' * 60}")
        print("  Market Data:")
        print(f"    Close Price:        ${event['Close']:.2f}")
        print(f"    Next Close:         ${event['future_close']:.2f}")
        print(f"    Actual Return:      {actual_return_pct:+.2f}%")
        print("\n  Model Prediction:")
        print(f"    Prediction:         {prediction} ({pred_label})")
        print(
            f"    Position Taken:     {position:+.2f} ({'Long' if position > 0 else 'Short' if position < 0 else 'Flat'})"
        )
        print("\n  Trade Result:")
        print(f"    P&L:                {pnl_pct:+.2f}%")

        # Manual calculation verification
        manual_pnl = position * actual_return_pct
        print(
            f"    Manual Check:       {manual_pnl:+.2f}% [OK]"
            if abs(manual_pnl - pnl_pct) < 0.01
            else f"    Manual Check:       {manual_pnl:+.2f}% [MISMATCH]"
        )

        # Show some features
        print("\n  Key Features:")
        print(f"    VIX:                {event['VIX_Close']:.2f}")
        print(f"    US10Y Yield:        {event['US10Y_Yield']:.2f}%")
        print(f"    Net Sentiment:      {event['net_sentiment_score']:.4f}")

        print("\n")


def run_binary_backtest(test_data, model_path=None, output_dir=None):
    """
    Run backtest with binary classification model.

    Args:
        test_data (pd.DataFrame): Test dataset
        model_path (str): Path to trained binary model
        output_dir (str): Directory to save results
    """
    print(f"\n{'=' * 60}")
    print("BINARY MODEL BACKTEST")
    print(f"{'=' * 60}\n")

    # Use default paths if not provided
    if model_path is None:
        model_path = (
            PROJECT_ROOT
            / "src"
            / "models"
            / "binary_model"
            / "models"
            / "xgboost_binary_classifier.pkl"
        )
    else:
        model_path = Path(model_path)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "src" / "backtesting" / "outputs" / "binary"
    else:
        output_dir = Path(output_dir)

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Binary model not found at: {model_path}\n"
            f"Please train the model first:\n"
            f"  cd {PROJECT_ROOT / 'src' / 'models' / 'binary_model'}\n"
            f"  python main.py"
        )

    print(f"Loading model from: {model_path}")

    # Load model
    predictor = ModelPredictor(model_path=str(model_path), model_type="binary")

    # Define strategy
    strategy = BinaryStrategy(name="Binary Long/Short")

    # Initialize backtest engine with realistic costs
    engine = BacktestEngine(
        predictor=predictor,
        strategy=strategy,
        initial_capital=100000,
        horizon=1,
        commission_pct=0.1,  # 0.1% commission per trade
        slippage_pct=0.05,  # 0.05% slippage
    )

    # Run manual verification first
    manual_verification(test_data, predictor, strategy, num_events=3)

    # Run full backtest
    results = engine.run(test_data, date_col="Date", return_col="return", verbose=True)

    # Generate all plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("\nGenerating visualizations...")
    engine.plot_equity_curve(save_path=f"{output_dir}/equity_curve.png", show_trades=True)
    engine.plot_returns_distribution(save_path=f"{output_dir}/returns_distribution.png")
    engine.plot_drawdown(save_path=f"{output_dir}/drawdown.png")
    engine.plot_monthly_returns(save_path=f"{output_dir}/monthly_returns.png")
    engine.plot_trade_analysis(save_path=f"{output_dir}/trade_analysis.png")

    # Export results
    engine.export_results(output_dir=output_dir)

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(engine, output_dir, strategy_name=strategy.name)

    return engine, results


def run_multiclass_backtest(
    test_data, model_path=None, scaler_path=None, metadata_path=None, output_dir=None
):
    """
    Run backtest with multi-class classification model.

    Args:
        test_data (pd.DataFrame): Test dataset
        model_path (str): Path to trained multi-class model
        scaler_path (str): Path to feature scaler
        metadata_path (str): Path to model metadata
        output_dir (str): Directory to save results
    """
    print(f"\n{'=' * 60}")
    print("MULTI-CLASS MODEL BACKTEST")
    print(f"{'=' * 60}\n")

    # Use default paths if not provided
    multi_base = PROJECT_ROOT / "src" / "models" / "multi_classification" / "src" / "outputs"

    if model_path is None:
        model_path = multi_base / "enhanced_multi_class_model.pkl"
    else:
        model_path = Path(model_path)

    if scaler_path is None:
        scaler_path = multi_base / "feature_scaler.pkl"
    else:
        scaler_path = Path(scaler_path)

    if metadata_path is None:
        metadata_path = multi_base / "enhanced_model_metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "src" / "backtesting" / "outputs" / "multiclass"
    else:
        output_dir = Path(output_dir)

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Multi-class model not found at: {model_path}\n"
            f"Please train the model first:\n"
            f"  python {PROJECT_ROOT / 'src' / 'models' / 'multi_classification' / 'src' / 'enhanced_multi_class_training.py'}"
        )

    print(f"Loading model from: {model_path}")

    # Load model
    predictor = ModelPredictor(
        model_path=str(model_path),
        model_type="multi_class",
        scaler_path=str(scaler_path) if scaler_path.exists() else None,
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
    )

    # Define strategy (scaled positions by magnitude)
    strategy = MultiClassStrategy(name="Multi-Class Scaled Positions", scale_positions=True)

    # Initialize backtest engine with realistic costs
    engine = BacktestEngine(
        predictor=predictor,
        strategy=strategy,
        initial_capital=100000,
        horizon=1,
        commission_pct=0.1,  # 0.1% commission per trade
        slippage_pct=0.05,  # 0.05% slippage
    )

    # Run manual verification first
    manual_verification(test_data, predictor, strategy, num_events=3)

    # Run full backtest
    results = engine.run(test_data, date_col="Date", return_col="return", verbose=True)

    # Generate all plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("\nGenerating visualizations...")
    engine.plot_equity_curve(save_path=f"{output_dir}/equity_curve.png", show_trades=True)
    engine.plot_returns_distribution(save_path=f"{output_dir}/returns_distribution.png")
    engine.plot_drawdown(save_path=f"{output_dir}/drawdown.png")
    engine.plot_monthly_returns(save_path=f"{output_dir}/monthly_returns.png")
    engine.plot_trade_analysis(save_path=f"{output_dir}/trade_analysis.png")

    # Export results
    engine.export_results(output_dir=output_dir)

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(engine, output_dir, strategy_name=strategy.name)

    return engine, results


def compare_strategies(test_data):
    """
    Compare multiple strategies on the same test data.

    Args:
        test_data (pd.DataFrame): Test dataset
    """
    print(f"\n{'=' * 60}")
    print("STRATEGY COMPARISON")
    print(f"{'=' * 60}\n")

    strategies_to_test = []

    # Binary model strategies
    binary_model_path = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "binary_model"
        / "models"
        / "xgboost_binary_classifier.pkl"
    )
    if binary_model_path.exists():
        binary_predictor = ModelPredictor(model_path=str(binary_model_path), model_type="binary")

        strategies_to_test.extend(
            [
                (binary_predictor, BinaryStrategy(name="Binary: Long/Short")),
                (binary_predictor, BinaryLongOnlyStrategy(name="Binary: Long Only")),
            ]
        )

    # Multi-class model strategies
    multi_base = PROJECT_ROOT / "src" / "models" / "multi_classification" / "src" / "outputs"
    multi_model_path = multi_base / "enhanced_multi_class_model.pkl"
    multi_scaler_path = multi_base / "feature_scaler.pkl"
    multi_metadata_path = multi_base / "enhanced_model_metadata.json"

    if multi_model_path.exists():
        multi_predictor = ModelPredictor(
            model_path=str(multi_model_path),
            model_type="multi_class",
            scaler_path=str(multi_scaler_path) if multi_scaler_path.exists() else None,
            metadata_path=str(multi_metadata_path) if multi_metadata_path.exists() else None,
        )

        strategies_to_test.extend(
            [
                (
                    multi_predictor,
                    MultiClassStrategy(name="Multi-Class: Scaled", scale_positions=True),
                ),
                (
                    multi_predictor,
                    MultiClassStrategy(name="Multi-Class: Binary", scale_positions=False),
                ),
            ]
        )

    # Run backtests for each strategy
    results_comparison = []

    for predictor, strategy in strategies_to_test:
        engine = BacktestEngine(
            predictor=predictor, strategy=strategy, initial_capital=100000, horizon=1
        )

        engine.run(test_data, date_col="Date", return_col="return", verbose=False)

        results_comparison.append(
            {
                "Strategy": strategy.name,
                "Total Return (%)": engine.metrics["total_return_pct"],
                "Win Rate (%)": engine.metrics["win_rate"] * 100,
                "Sharpe Ratio": engine.metrics["sharpe_ratio"],
                "Max Drawdown (%)": engine.metrics["max_drawdown_pct"],
                "Profit Factor": engine.metrics["profit_factor"],
                "Total Trades": engine.metrics["total_trades"],
            }
        )

    # Display comparison table
    comparison_df = pd.DataFrame(results_comparison)
    print("\nSTRATEGY PERFORMANCE COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    # Save comparison
    output_dir = PROJECT_ROOT / "src" / "backtesting" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
    print(f"\n[OK] Comparison saved to: {output_dir / 'strategy_comparison.csv'}")


def main():
    """Main execution function."""
    print(f"\n{'#' * 60}")
    print("# FED WATCHER - BACKTESTING ENGINE")
    print("# Event-Driven Strategy Simulation")
    print(f"{'#' * 60}\n")

    # Load test data (use default path)
    train_data, test_data = load_test_data(announcement_only=True)

    # Run binary model backtest
    binary_engine, binary_results = run_binary_backtest(test_data)

    # Run multi-class model backtest (if model exists)
    multi_model_path = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "multi_classification"
        / "src"
        / "outputs"
        / "enhanced_multi_class_model.pkl"
    )
    if multi_model_path.exists():
        multi_engine, multi_results = run_multiclass_backtest(test_data)
    else:
        print(f"\n⚠ Multi-class model not found at {multi_model_path}")
        print("  Skipping multi-class backtest.")
        print("  Run the multi-class training script first:")
        print("  python src/models/multi_classification/src/enhanced_multi_class_training.py\n")

    # Compare strategies
    compare_strategies(test_data)

    print(f"\n{'#' * 60}")
    print("# BACKTESTING COMPLETE")
    print("# Results saved to src/backtesting/outputs/")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
