# FED Watcher - Backtesting Engine

## Overview

The backtesting engine simulates event-driven trading strategies based on FED Watcher model predictions. It provides a framework for:
- Loading trained models (binary and multi-class)
- Executing trading strategies based on predictions
- Calculating profit & loss (P&L) for each trade
- Tracking portfolio equity over time
- Generating performance metrics and visualizations

## Architecture

```
src/backtesting/
├── __init__.py                 # Package initializer
├── model_predictor.py          # Unified model interface (Task 1)
├── strategies.py               # Trading strategy definitions (Task 2)
├── backtest_engine.py          # Event-driven backtesting engine (Task 2)
├── run_backtest.py             # Main execution script (Task 3)
└── README.md                   # This file (Task 4)
```

## Key Components

### 1. ModelPredictor (`model_predictor.py`)

**Purpose**: Unified interface for loading and running predictions from trained models.

**Features**:
- Loads binary and multi-class models from `.pkl` files
- Handles feature preprocessing and scaling
- Provides standardized prediction interface
- Supports probability predictions

**Usage**:
```python
from src.backtesting.model_predictor import ModelPredictor

# Binary model
binary_predictor = ModelPredictor(
    model_path='src/models/binary_model/models/xgboost_binary_classifier.pkl',
    model_type='binary'
)

# Multi-class model (requires scaler)
multi_predictor = ModelPredictor(
    model_path='src/models/multi_classification/outputs/enhanced_multi_class_model.pkl',
    model_type='multi_class',
    scaler_path='src/models/multi_classification/outputs/feature_scaler.pkl',
    metadata_path='src/models/multi_classification/outputs/enhanced_model_metadata.json'
)

# Make predictions
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)
```

### 2. Trading Strategies (`strategies.py`)

**Purpose**: Define how predictions are converted to trading positions.

**Available Strategies**:

1. **BinaryStrategy**: Long on Up prediction, Short on Down
   - Up (1) → Position = +1.0 (Long)
   - Down (0) → Position = -1.0 (Short)

2. **BinaryLongOnlyStrategy**: Long on Up, Flat on Down
   - Up (1) → Position = +1.0 (Long)
   - Down (0) → Position = 0 (Flat)

3. **MultiClassStrategy**: Scaled positions by magnitude
   - Strong Rise (+2) → Position = +1.0
   - Modest Rise (+1) → Position = +0.5
   - Neutral (0) → Position = 0
   - Modest Drop (-1) → Position = -0.5
   - Strong Drop (-2) → Position = -1.0

4. **ThresholdStrategy**: Only trade when confidence exceeds threshold
5. **KellyStrategy**: Kelly criterion position sizing

**Usage**:
```python
from src.backtesting.strategies import BinaryStrategy, MultiClassStrategy

# Simple binary strategy
strategy = BinaryStrategy(name='Binary Long/Short')

# Multi-class with scaled positions
strategy = MultiClassStrategy(name='Scaled Positions', scale_positions=True)
```

### 3. BacktestEngine (`backtest_engine.py`)

**Purpose**: Event-driven backtesting engine that simulates trading over time.

**Features**:
- Chronological iteration through test data
- Position sizing based on strategy
- P&L calculation for each trade
- Portfolio equity tracking
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Visualization (equity curves, return distributions)

**Usage**:
```python
from src.backtesting.backtest_engine import BacktestEngine

# Initialize engine
engine = BacktestEngine(
    predictor=predictor,
    strategy=strategy,
    initial_capital=100000,
    horizon=1  # 1-day holding period
)

# Run backtest
results = engine.run(test_data, date_col='Date', return_col='return')

# View results
engine.print_summary()
engine.plot_equity_curve(save_path='equity_curve.png')
engine.plot_returns_distribution(save_path='returns_dist.png')
engine.export_results(output_dir='results/')
```

## Critical Assumptions

The backtesting engine makes the following explicit assumptions:

### Trading Assumptions
1. **Zero Transaction Costs**: No commissions, fees, or taxes
2. **No Slippage**: Trades executed at exact closing price
3. **Full Liquidity**: Can always enter/exit positions at market price
4. **No Position Limits**: Unlimited position sizes
5. **Fixed Holding Period**: Positions held for exactly `horizon` days (default 1 day)
6. **Close-to-Close Returns**: All returns calculated using closing prices
7. **No Overnight Risk**: Assumes immediate execution on announcement day

### Model Assumptions
1. **Perfect Information**: Model has access to all features on announcement day
2. **No Look-Ahead Bias**: Only uses information available at prediction time
3. **Chronological Testing**: Test set is strictly chronological (no shuffling)
4. **Fixed Model**: Model parameters don't change during backtest

### Data Assumptions
1. **Complete Data**: All required features available for each event
2. **No Missing Values**: Dataset is complete
3. **Correct Labels**: Actual returns accurately reflect market movements

## How to Run

### Quick Start

1. **Ensure models are trained**:
```bash
# Train binary model
cd src/models/binary_model
python main.py

# Train multi-class model
cd src/models/multi_classification
python src/enhanced_multi_class_training.py
```

2. **Run backtest**:
```bash
# From project root
python src/backtesting/run_backtest.py
```

### Custom Backtest

```python
from src.backtesting import ModelPredictor, BacktestEngine, BinaryStrategy
import pandas as pd

# Load data
test_data = pd.read_csv('MasterDataset_Enriched.csv')
# ... prepare data ...

# Load model
predictor = ModelPredictor(
    model_path='path/to/model.pkl',
    model_type='binary'
)

# Define strategy
strategy = BinaryStrategy()

# Run backtest
engine = BacktestEngine(predictor, strategy, initial_capital=100000)
results = engine.run(test_data, date_col='Date', return_col='return')

# Analyze results
engine.print_summary()
```

## Understanding the Output

### Trade-by-Trade Results (`trades.csv`)

| Column | Description |
|--------|-------------|
| `date` | FOMC announcement date |
| `prediction` | Model's prediction (numeric) |
| `prediction_label` | Human-readable prediction |
| `position` | Position taken (-1 to +1) |
| `actual_return_pct` | Actual market return (%) |
| `pnl_pct` | P&L for this trade (%) |
| `pnl_dollars` | P&L in dollars |
| `equity` | Portfolio value after trade |

### Portfolio Equity (`equity_curve.csv`)

Time series of portfolio value over all trades.

### Performance Metrics (`metrics.json`)

```json
{
    "total_trades": 7,
    "total_return_pct": 12.5,
    "final_equity": 112500.0,
    "winning_trades": 4,
    "losing_trades": 3,
    "win_rate": 0.571,
    "avg_win_pct": 3.2,
    "avg_loss_pct": -1.8,
    "sharpe_ratio": 1.45,
    "max_drawdown_pct": -5.2,
    "profit_factor": 2.1
}
```

## Manual Verification (Task 3)

The `run_backtest.py` script includes manual verification on 3 historical FOMC events:

```python
def manual_verification(test_data, predictor, strategy, num_events=3):
    """
    Manually verify backtest logic on specific events.

    For each event:
    1. Shows market data (close prices, actual return)
    2. Shows model prediction and position taken
    3. Calculates P&L two ways (strategy and manual)
    4. Verifies they match
    """
```

**Example Output**:
```
Event 1: 2024-03-20
------------------------------------------------------------
  Market Data:
    Close Price:        $5200.12
    Next Close:         $5245.30
    Actual Return:      +0.87%

  Model Prediction:
    Prediction:         1 (Up)
    Position Taken:     +1.00 (Long)

  Trade Result:
    P&L:                +0.87%
    Manual Check:       +0.87% ✓

  Key Features:
    VIX:                14.25
    US10Y Yield:        4.35%
    Net Sentiment:      0.1234
```

## Performance Metrics Explained

### Return Metrics
- **Total Return**: Overall portfolio return (%)
- **Average Win**: Average return on winning trades (%)
- **Average Loss**: Average return on losing trades (%)

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
  - Calculation: `sqrt(252) * mean(returns) / std(returns)`
  - Annualized for daily trading

- **Max Drawdown**: Largest peak-to-trough decline (%)
  - Most negative point in equity curve

- **Profit Factor**: Total wins / Total losses
  - > 1.0 means profitable
  - > 2.0 is considered good

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Total events traded
- **Neutral Trades**: Trades with 0 P&L (flat positions)

## Strategy Comparison

Run `compare_strategies()` to test multiple strategies on the same data:

```python
from src.backtesting.run_backtest import compare_strategies

compare_strategies(test_data)
```

Output:
```
STRATEGY PERFORMANCE COMPARISON
=================================================================================================
Strategy                      Total Return (%)  Win Rate (%)  Sharpe Ratio  Max Drawdown (%)
Binary: Long/Short            12.5              57.1          1.45          -5.2
Binary: Long Only             8.3               57.1          1.12          -3.1
Multi-Class: Scaled           15.7              62.5          1.78          -4.8
Multi-Class: Binary           10.2              62.5          1.34          -6.1
=================================================================================================
```

## P&L Calculation Logic

### For Each Event (Trade):

1. **Get Prediction**: Model predicts market direction/magnitude
2. **Determine Position**: Strategy converts prediction to position size
   - Long: +1.0 (expect price to rise)
   - Short: -1.0 (expect price to fall)
   - Flat: 0 (no position)
   - Scaled: -1.0 to +1.0 (size by confidence/magnitude)

3. **Calculate P&L**:
   ```
   P&L (%) = Position × Actual Return (%)
   P&L ($) = Portfolio Equity × (P&L % / 100)
   ```

4. **Update Equity**:
   ```
   New Equity = Old Equity + P&L ($)
   ```

### Example:
- Portfolio: $100,000
- Prediction: Up (position = +1.0 long)
- Actual Return: +2.5%
- P&L: +1.0 × 2.5% = +2.5%
- P&L Dollars: $100,000 × 0.025 = $2,500
- New Equity: $102,500

## Extending the Framework

### Adding New Strategies

1. Create a new class inheriting from `Strategy`:

```python
from src.backtesting.strategies import Strategy

class CustomStrategy(Strategy):
    def __init__(self, name='Custom Strategy', threshold=0.5):
        super().__init__(name)
        self.threshold = threshold

    def get_position(self, prediction, **kwargs):
        # Your logic here
        if prediction > self.threshold:
            return 1.0
        else:
            return -1.0
```

2. Use it in backtesting:

```python
strategy = CustomStrategy(threshold=0.6)
engine = BacktestEngine(predictor, strategy)
```

### Adding New Metrics

Extend `BacktestEngine._calculate_metrics()`:

```python
# In backtest_engine.py
def _calculate_metrics(self):
    # ... existing metrics ...

    # Add custom metric
    self.metrics['custom_metric'] = your_calculation()
```

## Troubleshooting

### Common Issues

1. **"Missing required features" error**
   - Ensure test data has all features used in training
   - Check feature names match exactly

2. **"Model file not found"**
   - Train models first using training scripts
   - Verify paths in `run_backtest.py`

3. **"Invalid return values"**
   - Check that `return` column is calculated correctly
   - Ensure returns are decimal (not %, so 0.05 not 5.0)

4. **Unrealistic results**
   - Review assumptions section
   - Remember: zero transaction costs and no slippage
   - Check for look-ahead bias in features

## Files Generated

After running backtest, the following files are created:

```
backtesting_results/
├── binary/
│   ├── trades.csv                  # Trade-by-trade results
│   ├── equity_curve.csv            # Portfolio value over time
│   ├── metrics.json                # Performance metrics
│   ├── equity_curve.png            # Equity curve visualization
│   └── returns_distribution.png    # Return distribution plots
├── multiclass/
│   └── [same structure]
└── strategy_comparison.csv         # Comparison across strategies
```

## References

### Related Files
- Model training: `src/models/binary_model/main.py`
- Multi-class training: `src/models/multi_classification/src/enhanced_multi_class_training.py`
- Data preparation: `src/models/binary_model/src/data_preparation.py`

### Key Concepts
- Event-driven backtesting: Sequential processing of events
- Walk-forward analysis: Strict chronological ordering
- Out-of-sample testing: Test set never seen during training

## Contact & Support

For questions about the backtesting engine:
1. Review this documentation
2. Check code comments in source files
3. Examine example in `run_backtest.py`

---

**Project**: FED Watcher
**Module**: Backtesting Engine
**Version**: 1.0
**Last Updated**: 2025-01-25