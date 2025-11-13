# 🚀 QUICK START GUIDE

## TL;DR - Run This Now

```bash
cd fed_market_prediction
python main.py
```

Done! Your model is trained and saved.

---

## What Just Happened?

In 30 seconds, the pipeline:
1. ✅ Loaded your enriched dataset (1,761 rows)
2. ✅ Created binary target (Up/Down)
3. ✅ Filtered to 35 Fed announcement days
4. ✅ Split into train (28) and test (7) sets
5. ✅ Trained XGBoost classifier
6. ✅ Evaluated performance
7. ✅ Saved model to `models/`
8. ✅ Logged metrics to `logs/`

---

## Results You Got

### Test Set Performance:
- **Accuracy: 57.1%** (barely better than coin flip)
- **F1-Score: 66.7%**
- Precision: 100% (when predicts Up, always right)
- Recall: 50% (misses half the Up moves)

### Why Low Performance?
- **Only 35 total samples** (very small!)
- 7 samples in test set (tiny!)
- Model might be overfitting (100% train accuracy)

---

## Where Are My Files?

```
fed_market_prediction/
├── 📊 MasterDataset_Enriched.csv   ← Your data (ready to use!)
│
├── 🐍 main.py                       ← Run this file
│
├── 📂 src/
│   ├── data_preparation.py          ← Loads data, creates targets
│   └── model_training.py            ← Trains model, evaluates
│
├── 🤖 models/
│   └── xgboost_binary_classifier.pkl ← Your trained model
│
├── 📝 logs/
│   └── metrics.json                 ← Performance metrics
│
└── 📖 README.md                     ← Full documentation
```

---

## What Changed from Your Original Plan?

### ❌ You DON'T Need To:
- Load Fed transcripts separately
- Run FinBERT for sentiment analysis
- Merge sentiment with market data
- Write complex NLP pipelines

### ✅ You Already Have:
- All sentiment features pre-computed
- Market context features included
- Clean, ready-to-use dataset
- Working ML pipeline

---

## Your Dataset Has Everything

**18 columns total:**

1. `Date` - Trading date
2. `Close` - S&P 500 closing price
3. `Volume` - Trading volume
4. `VIX_Close` - Market volatility
5. `DXY_Close` - US Dollar strength
6. `US02Y_Yield` - 2-year Treasury yield
7. `US10Y_Yield` - 10-year Treasury yield
8. `Yield_Curve_10Y_2Y` - Yield curve spread
9. `Volume_ratio_vs_5days` - Volume indicator
10. `Announcement` - **Fed announcement flag (0 or 1)**
11. `hawkish_dovish_ratio` - Sentiment ratio
12. `net_sentiment_score` - Overall sentiment
13. `negative_proportion` - % negative
14. `neutral_proportion` - % neutral
15. `positive_proportion` - % positive
16. `sentiment_label` - Sentiment category
17. Plus: Open, High, Low (excluded from features)

**Model uses 15 of these as features**

---

## Top 5 Important Features

Your model found these most predictive:

1. **US02Y_Yield** (23%) - Short-term interest rates
2. **net_sentiment_score** (12%) - Fed sentiment
3. **Yield_Curve_10Y_2Y** (10%) - Economic indicator
4. **Volume_ratio_vs_5days** (10%) - Trading activity
5. **DXY_Close** (9%) - Dollar strength

---

## Sprint Requirements ✅

All acceptance criteria completed:

| Requirement | Status |
|------------|--------|
| Binary target from returns | ✅ Done |
| Train/test split | ✅ Done (chronological) |
| Model persistence | ✅ Saved as .pkl |
| ML metrics logged | ✅ Accuracy, F1, etc. |

---

## Next Steps (Optional)

### Improve Performance:
```python
# In main.py, change this line:
use_announcement_only=False  # Use all 1,761 days
```

### Tune Model:
```python
# In src/model_training.py, adjust params:
params = {
    'max_depth': 5,        # Try different depth
    'learning_rate': 0.05, # Try slower learning
    'n_estimators': 200    # Try more trees
}
```

### Try Different Horizon:
```python
# In main.py, change prediction window:
horizon=3  # Predict 3 days ahead instead of 1
```

---

## Common Questions

**Q: Why only 35 samples?**
A: Filtered to Fed announcement days only. Change `use_announcement_only=False` to use all 1,761 days.

**Q: Why 57% accuracy?**
A: Very small dataset (35 samples). Try using all days or collecting more data.

**Q: Can I change features?**
A: Yes! Edit `select_features()` in `src/data_preparation.py`

**Q: Where's the FinBERT code?**
A: Don't need it! Sentiment already in your enriched dataset.

**Q: How do I use the saved model?**
A:
```python
from src.model_training import load_model
model = load_model('models/xgboost_binary_classifier.pkl')
predictions = model.predict(new_data)
```

---

## Remember

✅ Your code follows **modular functional design** (your preference)
✅ Every function has **clear docstrings**
✅ No black boxes - **you understand each step**
✅ Easy to modify and extend

---

## That's It!

You're done. Model trained. Sprint complete. 🎉

Read README.md for detailed documentation.

Run `python main.py` anytime to retrain.
