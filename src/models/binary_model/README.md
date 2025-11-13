# Fed Market Prediction - Binary Classification

## Project Overview
Machine learning system to predict short-term U.S. stock market movements (24 hours) following Federal Reserve announcements by Jerome Powell, using XGBoost binary classification.

---

## 🔄 WHAT CHANGED WITH THE NEW DATASET

### Your New Dataset: `MasterDataset_Enriched.csv`

#### **Previously**: 
- Basic market data (Date, OHLC, Volume)
- You had to manually run FinBERT to extract sentiment

#### **Now You Have**:
✅ **Market Context Features** (7 new columns):
- `VIX_Close` - Volatility Index
- `DXY_Close` - US Dollar Index  
- `US02Y_Yield` - 2-Year Treasury Yield
- `US10Y_Yield` - 10-Year Treasury Yield
- `Yield_Curve_10Y_2Y` - Yield Curve Spread
- `Volume_ratio_vs_5days` - Volume relative to 5-day average

✅ **Pre-computed NLP Sentiment Features** (6 new columns):
- `Announcement` - Binary flag (1 = Fed announcement day)
- `hawkish_dovish_ratio` - Ratio of hawkish to dovish sentiment
- `net_sentiment_score` - Net sentiment score
- `negative_proportion` - % negative sentences
- `neutral_proportion` - % neutral sentences  
- `positive_proportion` - % positive sentences
- `sentiment_label` - Overall sentiment category

#### **Total Rows**: 1,761 daily observations (2018-2024)
#### **Announcement Days**: 35 Fed announcements

---

## 📁 PROJECT STRUCTURE

```
fed_market_prediction/
├── main.py                          # Main execution script
├── MasterDataset_Enriched.csv       # Your enriched dataset
├── requirements.txt                 # Python dependencies
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_preparation.py          # Data loading & preprocessing
│   └── model_training.py            # Model training & evaluation
├── models/
│   └── xgboost_binary_classifier.pkl # Saved trained model
└── logs/
    └── metrics.json                 # Performance metrics log
```

---

## 🎯 WHAT YOU NEED TO CHANGE

### **NOTHING! Everything is ready to run.**

But here's what the new code does differently:

### 1. **Data Preparation** (`src/data_preparation.py`)
- ✅ Loads enriched dataset directly
- ✅ Creates binary target from market returns
- ✅ Filters to announcement days only (35 samples)
- ✅ Uses 15 features (all the new ones!)
- ✅ Chronological train/test split (80/20)

### 2. **Model Training** (`src/model_training.py`)
- ✅ XGBoost binary classifier
- ✅ Predicts market direction (Up=1, Down=0)
- ✅ Evaluates with Accuracy, Precision, Recall, F1-Score
- ✅ Saves model as `.pkl` file
- ✅ Logs metrics to JSON

### 3. **Main Pipeline** (`main.py`)
- ✅ Orchestrates everything
- ✅ Runs end-to-end in one command

---

## 🚀 HOW TO RUN

### **Option 1: Run Complete Pipeline**
```bash
cd fed_market_prediction
python main.py
```

This will:
1. Load and prepare data
2. Train XGBoost model
3. Evaluate performance
4. Save model and metrics
5. Show feature importance

### **Option 2: Use Saved Model**
```python
from src.model_training import load_model
import pandas as pd

# Load trained model
model = load_model('models/xgboost_binary_classifier.pkl')

# Make predictions on new data
predictions = model.predict(X_new)
```

---

## 📊 CURRENT RESULTS

Based on your enriched dataset (35 announcement days):

### **Train Set** (28 samples):
- Accuracy: 100%
- F1-Score: 100%
- *Note: Perfect training score suggests possible overfitting*

### **Test Set** (7 samples):
- **Accuracy: 57.1%**
- **F1-Score: 66.7%**
- Precision: 100% (when predicts "Up", it's always right)
- Recall: 50% (misses half of actual "Up" moves)

### **Top Features by Importance**:
1. **US02Y_Yield** (23.4%)
2. **net_sentiment_score** (11.7%)
3. **Yield_Curve_10Y_2Y** (10.5%)
4. **Volume_ratio_vs_5days** (10.0%)
5. **DXY_Close** (9.5%)

---

## ✅ SPRINT ACCEPTANCE CRITERIA MET

- ✅ Binary target variable created from market returns
- ✅ Proper train/test chronological split (no data leakage)
- ✅ Model persisted to disk (`.pkl` file)
- ✅ Key ML metrics logged (Accuracy, F1-Score, Precision, Recall)

---

## 🔧 KEY DESIGN DECISIONS

### 1. **Only Announcement Days**
- Filters to 35 days where `Announcement=1`
- Focuses model on post-Fed-announcement reactions
- Can be changed: set `use_announcement_only=False` in `main.py`

### 2. **24-Hour Prediction Window**
- Predicts market direction 1 day ahead (`horizon=1`)
- Can be adjusted in `prepare_pipeline()` call

### 3. **Chronological Split**
- Train: 2020-09 to 2024-01 (28 samples)
- Test: 2024-03 to 2024-12 (7 samples)
- Prevents lookahead bias

### 4. **15 Features Used**
All features except OHLC (kept only Close to avoid multicollinearity):
- Close price
- Volume
- VIX, DXY indices
- Treasury yields
- Sentiment scores
- Announcement flag

---

## 🎓 UNDERSTANDING THE CODE

### **Modular Structure**
Each file has a clear purpose:
- `data_preparation.py` - Data handling
- `model_training.py` - ML operations
- `main.py` - Orchestration

### **Function-Based Approach**
We use organized functions (your preferred hybrid approach):
- Easy to understand
- Easy to modify
- Professional structure
- No unnecessary OOP complexity

### **Proper Python Package**
- `src/` directory with `__init__.py`
- Import syntax: `from src.data_preparation import ...`
- Follows Python best practices

---

## 🔮 NEXT STEPS / IMPROVEMENTS

### 1. **Address Small Dataset**
- Only 35 announcement days → limited samples
- Consider using all days (not just announcements)
- Or collect more historical Fed data

### 2. **Improve Model Performance**
- Test set accuracy is 57% (barely better than random)
- Try hyperparameter tuning
- Feature engineering
- Different models (Random Forest, Neural Networks)

### 3. **Handle Overfitting**
- Train accuracy 100%, Test accuracy 57%
- Add regularization
- Reduce model complexity
- Use cross-validation

### 4. **Multi-Class Prediction** 
- Current: Binary (Up/Down)
- Sprint also mentions: Return range prediction
- Can extend to classify magnitude of moves

---

## 📝 DEPENDENCIES

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
```

Install: `pip install -r requirements.txt`

---

## 🤝 WORKING WITH THIS PROJECT

### To Modify Features:
Edit `select_features()` in `src/data_preparation.py`

### To Change Model Parameters:
Edit `train_xgboost_model()` in `src/model_training.py`

### To Adjust Train/Test Split:
Change `test_size` parameter in `prepare_pipeline()` call in `main.py`

### To Use All Days (Not Just Announcements):
Set `use_announcement_only=False` in `main.py`

---

## 📧 Questions?

Review the docstrings in each function - they explain parameters and return values clearly.

---

**Project Status**: ✅ Sprint Complete - Ready for Review
