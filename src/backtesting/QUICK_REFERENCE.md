# Backtesting Engine - Quick Reference

## Installation

```bash
# All dependencies already in requirements.txt
pip install -r requirements.txt
```

## Quick Start (3 Commands)

```bash
# 1. Verify tests pass
python src/backtesting/test_backtest.py

# 2. Run complete backtest
python src/backtesting/run_backtest.py

# 3. View results
ls backtesting_results/
```

## Common Use Cases

### Run Binary Model Backtest

```python
from src.backtesting import ModelPredictor, BacktestEngine, BinaryStrategy
import pandas as pd

# Load data
df = pd.read_csv('MasterDataset_Enriched.csv')
# ... prepare data with returns ...

# Load model
predictor = ModelPredictor(
    model_path='src/models/binary_model/models/xgboost_binary_classifier.pkl',
    model_type='binary'
)

# Run backtest
strategy = BinaryStrategy()
engine = BacktestEngine(predictor, strategy, initial_capital=100000)
results = engine.run(df, date_col='Date', return_col='return')

# View results
engine.print_summary()
engine.plot_equity_curve()
```

### Run Multi-Class Model Backtest

```python
from src.backtesting import ModelPredictor, BacktestEngine, MultiClassStrategy

# Load model with scaler
predictor = ModelPredictor(
    model_path='src/models/multi_classification/outputs/enhanced_multi_class_model.pkl',
    model_type='multi_class',
    scaler_path='src/models/multi_classification/outputs/feature_scaler.pkl',
    metadata_path='src/models/multi_classification/outputs/enhanced_model_metadata.json'
)

# Run backtest with scaled positions
strategy = MultiClassStrategy(scale_positions=True)
engine = BacktestEngine(predictor, strategy)
results = engine.run(df, date_col='Date', return_col='return')
```

### Compare Multiple Strategies

```python
strategies = [
    BinaryStrategy(name='Long/Short'),
    BinaryLongOnlyStrategy(name='Long Only'),
    MultiClassStrategy(name='Scaled', scale_positions=True),
    ThresholdStrategy(name='High Confidence', threshold=0.7)
]

for strategy in strategies:
    engine = BacktestEngine(predictor, strategy)
    engine.run(df, date_col='Date', return_col='return', verbose=False)
    print(f"{strategy.name}: {engine.metrics['total_return_pct']:.2f}%")
```

## Available Strategies

| Strategy | Position Logic | Use Case |
|----------|---------------|----------|
| `BinaryStrategy` | Long on Up, Short on Down | Full long/short trading |
| `BinaryLongOnlyStrategy` | Long on Up, Flat on Down | Long-only portfolio |
| `MultiClassStrategy` | Scaled by magnitude | Nuanced position sizing |
| `ThresholdStrategy` | Filter by confidence | Reduce low-confidence trades |
| `KellyStrategy` | Kelly criterion sizing | Optimal bet sizing |

## Output Files

### trades.csv
```
date,prediction,position,actual_return_pct,pnl_pct,pnl_dollars,equity
2024-03-20,1,1.0,0.87,0.87,870.00,100870.00
```

### metrics.json
```json
{
    "total_return_pct": 12.5,
    "win_rate": 0.571,
    "sharpe_ratio": 1.45,
    "max_drawdown_pct": -5.2
}
```

## Key Metrics

- **Total Return**: Overall portfolio return (%)
- **Win Rate**: % of profitable trades
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Worst peak-to-trough decline (%)
- **Profit Factor**: Total wins / Total losses (>1 is profitable)

## Command Line Options

### Run with specific model

```bash
# Binary model only
python src/backtesting/run_backtest.py --model binary

# Multi-class model only
python src/backtesting/run_backtest.py --model multiclass
```

### Export results to custom directory

```python
engine.export_results(output_dir='custom_results/')
```

### Save plots

```python
engine.plot_equity_curve(save_path='my_equity_curve.png')
engine.plot_returns_distribution(save_path='my_returns.png')
```

## Manual Verification Example

```python
from src.backtesting.run_backtest import manual_verification

# Verify 3 specific events
manual_verification(test_data, predictor, strategy, num_events=3)
```

Output shows:
- Market data (close prices, return)
- Model prediction and position
- P&L calculation
- Manual verification (✓ if correct)

## P&L Calculation

```
Position × Actual Return (%) = P&L (%)
Portfolio × (P&L % / 100) = P&L ($)
```

Example:
- Position: +1.0 (Long)
- Return: +2.5%
- P&L: 1.0 × 2.5% = +2.5%
- On $100k: $100k × 0.025 = $2,500 profit

## Assumptions

⚠️ **Remember**: Backtest assumes:
- Zero transaction costs
- No slippage
- Perfect liquidity
- 1-day holding period
- Close-to-close returns

Real trading will differ!

## Troubleshooting

### "Missing required features" error
→ Ensure test data has all features used in training

### "Model file not found"
→ Train models first:
```bash
cd src/models/binary_model && python main.py
```

### "Invalid return values"
→ Returns should be decimal (0.05 not 5.0)

### Unrealistic results
→ Review assumptions section
→ Check for look-ahead bias

## Getting Help

1. Read `README.md` for detailed docs
2. Check `test_backtest.py` for examples
3. Review code comments in source files
4. See `IMPLEMENTATION_SUMMARY.md` for architecture

## File Locations

```
src/backtesting/
├── model_predictor.py       # Load models
├── strategies.py            # Define strategies
├── backtest_engine.py       # Run simulations
├── run_backtest.py          # Main script
├── test_backtest.py         # Verification tests
├── README.md                # Full documentation
└── QUICK_REFERENCE.md       # This file
```

## Next Steps

After running backtest:
1. Review `backtesting_results/` outputs
2. Analyze equity curve for drawdowns
3. Check win rate and profit factor
4. Compare strategies
5. Adjust based on insights
6. Re-test with new parameters

---

**Quick Links**:
- Full Docs: `README.md`
- Implementation Details: `IMPLEMENTATION_SUMMARY.md`
- Project Overview: `../../CLAUDE.md`