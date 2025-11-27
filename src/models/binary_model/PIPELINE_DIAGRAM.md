# 📊 PIPELINE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                     MAIN.PY (Orchestrator)                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION (src/data_preparation.py)            │
├─────────────────────────────────────────────────────────────────┤
│  1. load_data()                                                 │
│     • Read MasterDataset_Enriched.csv                          │
│     • Parse dates, set index                                   │
│     • Result: 1,761 rows × 18 columns                          │
│                                                                 │
│  2. create_binary_target()                                     │
│     • Calculate next-day return                                │
│     • Create target: 1 if Up, 0 if Down                        │
│     • Result: Binary classification target                     │
│                                                                 │
│  3. prepare_announcement_data()                                │
│     • Filter: Announcement == 1                                │
│     • Result: 35 Fed announcement days                         │
│                                                                 │
│  4. select_features()                                          │
│     • Keep 15 features (exclude OHLC except Close)             │
│     • Features: Close, Volume, VIX, DXY, Yields,               │
│                 Sentiment scores, Announcement flag             │
│                                                                 │
│  5. split_data()                                               │
│     • Chronological split (no shuffle!)                        │
│     • Train: 28 samples (80%)                                  │
│     • Test:  7 samples (20%)                                   │
│                                                                 │
│  Output: X_train, X_test, y_train, y_test, feature_cols       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: MODEL TRAINING (src/model_training.py)                │
├─────────────────────────────────────────────────────────────────┤
│  train_xgboost_model()                                         │
│     • Algorithm: XGBoost Binary Classifier                     │
│     • Parameters:                                              │
│         - max_depth: 3                                         │
│         - learning_rate: 0.1                                   │
│         - n_estimators: 100                                    │
│     • Trains on 28 samples                                     │
│     • Result: Trained model                                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: EVALUATION (src/model_training.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  evaluate_model()                                              │
│     • Predictions on train & test sets                         │
│     • Calculate metrics:                                       │
│         ✓ Accuracy                                             │
│         ✓ Precision                                            │
│         ✓ Recall                                               │
│         ✓ F1-Score                                             │
│     • Confusion matrix                                         │
│     • Classification report                                    │
│                                                                 │
│  Results:                                                      │
│     Train: 100% accuracy (overfitting!)                        │
│     Test:  57% accuracy, 67% F1-score                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: FEATURE IMPORTANCE                                    │
├─────────────────────────────────────────────────────────────────┤
│  get_feature_importance()                                      │
│     • Top features:                                            │
│         1. US02Y_Yield (23%)                                   │
│         2. net_sentiment_score (12%)                           │
│         3. Yield_Curve_10Y_2Y (10%)                            │
│         4. Volume_ratio_vs_5days (10%)                         │
│         5. DXY_Close (9%)                                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: SAVE RESULTS                                          │
├─────────────────────────────────────────────────────────────────┤
│  save_model()                                                  │
│     → models/xgboost_binary_classifier.pkl                     │
│                                                                 │
│  save_metrics()                                                │
│     → logs/metrics.json                                        │
│         {                                                      │
│           "train": {"accuracy": 1.0, "f1": 1.0},               │
│           "test": {"accuracy": 0.57, "f1": 0.67}               │
│         }                                                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            ✅ COMPLETE!
```

---

## DATA FLOW

```
MasterDataset_Enriched.csv (1,761 rows)
            │
            ├─ 18 columns total:
            │  • Date, OHLC, Volume
            │  • VIX_Close, DXY_Close
            │  • US02Y_Yield, US10Y_Yield
            │  • Yield_Curve_10Y_2Y
            │  • Volume_ratio_vs_5days
            │  • Announcement
            │  • 6 sentiment features
            │
            ▼
    Filter: Announcement == 1
            │
            ▼
    35 announcement days
            │
            ├─ Create target: next-day direction
            │  (Up=1, Down=0)
            │
            ▼
    Select 15 features
            │
            ├─ Train: 28 samples (2020-2024 early)
            │
            └─ Test:  7 samples (2024 late)
                     │
                     ▼
                 XGBoost Model
                     │
                     ├─ Train accuracy: 100%
                     └─ Test accuracy:  57%
```

---

## KEY DIFFERENCES FROM YOUR ORIGINAL PLAN

### ORIGINAL PLAN:
```
Raw Market Data + Raw Transcripts
         │
         ├─ Load transcripts
         ├─ Run FinBERT (slow!)
         ├─ Extract sentiment
         ├─ Merge with market data
         └─ Build model
```

### NEW REALITY:
```
MasterDataset_Enriched.csv (everything included!)
         │
         └─ Build model (that's it!)
```

**Saved Steps**: Loading transcripts, running FinBERT, merging data

**Time Saved**: Hours of NLP processing

**Code Simplified**: 3 clean modules instead of complex pipeline

---

## MODULAR DESIGN BENEFITS

```
main.py
   ├─ Imports from src.data_preparation
   ├─ Imports from src.model_training
   └─ Calls functions in logical order

✓ Easy to understand
✓ Easy to modify
✓ Easy to extend
✓ No hidden complexity
```

---

## WHERE IS EACH PIECE?

```
Code Organization:
├─ Data loading       → src/data_preparation.py
├─ Target creation    → src/data_preparation.py
├─ Feature selection  → src/data_preparation.py
├─ Train/test split   → src/data_preparation.py
├─ Model training     → src/model_training.py
├─ Evaluation         → src/model_training.py
├─ Saving/loading     → src/model_training.py
└─ Orchestration      → main.py
```

---

## YOUR ROLE: Zero Changes Needed!

The enriched dataset already has everything you need.
The code is ready to run.
The sprint requirements are met.

**Just run**: `python main.py`

**That's it!** 🎉
