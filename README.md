# FED Watcher

[![Python CI/CD Pipeline](https://github.com/JacobDrizzle/FED_Watcher/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/JacobDrizzle/FED_Watcher/actions/workflows/ci-cd.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning system that predicts short-term U.S. stock market movements by analyzing sentiment from Federal Reserve Chair announcements. The project combines NLP sentiment analysis using FinBERT with comprehensive market indicators to train predictive models and backtest trading strategies.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Sentiment Analysis](#1-sentiment-analysis)
  - [Market Data Enrichment](#2-market-data-enrichment)
  - [Model Training](#3-model-training)
  - [Backtesting](#4-backtesting)
- [Project Structure](#project-structure)
- [Models & Performance](#models--performance)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities

✅ **NLP Sentiment Analysis**
- FinBERT-powered sentiment analysis of Federal Reserve announcements
- Semantic chunking for contextually meaningful text processing
- Hawkish/dovish sentiment classification
- Document-level and chunk-level sentiment aggregation

✅ **Market Data Enrichment**
- Integration of 7 macro-financial indicators (VIX, DXY, Treasury yields)
- Volume analysis and yield curve calculations
- Automated data fetching and enrichment pipeline

✅ **Machine Learning Models**
- Binary classification (Up/Down market prediction)
- Multi-class classification (Strong Rise to Strong Drop)
- XGBoost and Gradient Boosting algorithms
- Feature importance analysis and model persistence

✅ **Backtesting Engine**
- Event-driven backtesting framework
- Multiple trading strategies (Long/Short, Long-Only, Scaled positions)
- Comprehensive performance metrics (Sharpe ratio, Max Drawdown, Win Rate)
- P&L tracking and equity curve visualization

✅ **Production-Grade Infrastructure**
- Comprehensive test suite (Unit, Integration, Acceptance)
- GitHub Actions CI/CD pipeline
- Code quality enforcement (Black, Flake8, Pylint)
- Security scanning (Bandit, Safety)
- GitFlow branching workflow

## Architecture

FED Watcher follows a modular, pipeline-based architecture:

```
┌─────────────────┐
│ Data Collection │ → Powell Speeches, S&P 500 Data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NLP Processing  │ → FinBERT Sentiment Analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Enrichment│ → VIX, DXY, Treasury Yields, Volume
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │ → Binary & Multi-Class Models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backtesting    │ → Strategy Evaluation, Performance Metrics
└─────────────────┘
```

**Key Dataset**: `data/MasterDataset_Enriched.csv`
- **1,761 daily observations** (2018-2024)
- **40 Federal Reserve announcement days**
- **20 features** including sentiment scores, macro indicators, and market data

**Centralized Data Directory**: All data files are organized in the `data/` directory at the project root.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment manager (`venv` or `conda`)
- Kaggle account (for downloading the dataset)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/JacobDrizzle/FED_Watcher.git
cd FED_Watcher
```

2. **Create and activate a virtual environment:**

```bash
# Create the virtual environment
python -m venv finbert_env

# Activate the environment
# On Windows:
.\finbert_env\Scripts\activate

# On macOS/Linux:
source finbert_env/bin/activate
```

3. **Install dependencies:**

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (for testing and linting)
pip install -r requirements-dev.txt
```

4. **Download the dataset:**

Download the FOMC Press Conference transcripts from Kaggle:

**Dataset**: [Fed Press Release Text](https://www.kaggle.com/datasets/jonathanpaserman/fed-press-release-text)

After downloading:
- Extract the `.txt` files from the dataset
- Place all transcript files (e.g., `FOMCpresconf20200916.txt`) into `data/raw_data/`

```
data/
├── raw_data/                              # Raw FOMC transcript files
│   ├── FOMCpresconf20200916.txt
│   ├── FOMCpresconf20201105.txt
│   └── ... (40 transcript files)
├── processed/                             # Processed data outputs
│   ├── powell_speeches_output/            # Extracted Powell speeches
│   └── stock_data/                        # S&P 500 historical data
├── sentiment_results/                     # NLP sentiment analysis outputs
│   ├── semantic_chunk_sentiments.csv
│   └── semantic_document_sentiments.csv
├── MasterDataset_Final.csv                # Combined stock + sentiment data
└── MasterDataset_Enriched.csv             # Final enriched dataset
```

5. **Verify installation:**

```bash
# Run verification tests
pytest tests/unit/test_sample.py -v

# Verify FinBERT model
python src/nlp/finbert_analyser.py --mode verify
```

## Usage

All commands should be run from the project root directory (`FED_Watcher/`).

### 1. Data Extraction & Cleaning

Extract and clean Chair Powell's speeches from raw FOMC transcripts.

```bash
python src/cleaning/extract_powell_speeches.py
```

**Input:** `data/raw_data/*.txt` (40 FOMC transcript files from Kaggle)

**Output:** `data/processed/powell_speeches_output/`
- Individual speech CSVs per meeting
- `all_powell_speeches_combined.csv` - Combined file with all speeches

---

### 2. Sentiment Analysis

Analyze sentiment from Powell speeches using FinBERT with semantic chunking.

```bash
python src/nlp/semantic_sentiment_analyser.py
```

**Input:** `data/processed/powell_speeches_output/`

**Output:** `data/sentiment_results/`
- `semantic_chunk_sentiments.csv` - Chunk-level sentiment scores
- `semantic_document_sentiments.csv` - Document-level aggregated sentiments

**Optional Arguments:**
```bash
python src/nlp/semantic_sentiment_analyser.py \
    --input_dir data/processed/powell_speeches_output \
    --chunk_output data/sentiment_results/semantic_chunk_sentiments.csv \
    --doc_output data/sentiment_results/semantic_document_sentiments.csv
```

**Verify FinBERT Installation:**
```bash
python src/nlp/finbert_analyser.py --mode verify
```

---

### 3. Stock Data Collection

Download S&P 500 historical price data from Yahoo Finance.

```bash
python src/stock_data/get_stock_data.py
```

**Output:** `data/processed/stock_data/SP500_2018_2025_Clean.csv`

---

### 4. Master Dataset Creation

Combine stock data with sentiment analysis results.

```bash
python src/enrichment/master_pipeline.py
```

**Input:**
- `data/processed/stock_data/SP500_2018_2025_Clean.csv`
- `data/sentiment_results/semantic_document_sentiments.csv`

**Output:** `data/MasterDataset_Final.csv`
- Merged stock prices with sentiment scores
- Announcement day flags
- Volume ratios and sentiment labels

---

### 5. Market Data Enrichment

Enrich the dataset with macro-financial indicators from Yahoo Finance and FRED.

```bash
python src/enrichment/enrich_with_market_data.py
```

**Input:** `data/MasterDataset_Final.csv`

**Output:** `data/MasterDataset_Enriched.csv`

**Enrichment Features Added:**

| Feature | Description | Source |
|---------|-------------|--------|
| `VIX_Close` | Volatility Index (market fear gauge) | Yahoo Finance |
| `DXY_Close` | US Dollar Index | Yahoo Finance |
| `US02Y_Yield` | 2-Year Treasury Yield | FRED |
| `US10Y_Yield` | 10-Year Treasury Yield | Yahoo Finance |
| `Yield_Curve_10Y_2Y` | Yield curve spread (10Y - 2Y) | Calculated |

---

### 6. Model Training

#### Binary Classification Model

Predicts market direction (Up/Down) following Fed announcements.

```bash
python src/models/binary_model/main.py
```

**Input:** `data/MasterDataset_Enriched.csv`

**Output:** `src/models/binary_model/`
- `models/xgboost_binary_classifier.pkl` - Trained model
- `logs/metrics.json` - Performance metrics
- `outputs/` - Visualizations (confusion matrix, ROC curve, feature importance)

**Model Details:**
- Algorithm: XGBoost Binary Classifier
- Features: 15 selected features from enriched dataset
- Train/Test Split: 80/20 chronological

#### Multi-Class Classification Model

Predicts market movement magnitude (-2 to +2).

```bash
python src/models/multi_classification/src/enhanced_multi_class_training.py
```

**Input:** `data/MasterDataset_Enriched.csv`

**Output:** `src/models/multi_classification/src/outputs/`
- `enhanced_multi_class_model.pkl` - Trained model
- `feature_scaler.pkl` - StandardScaler for feature normalization
- `enhanced_model_metadata.json` - Model configuration and metrics
- Visualizations (confusion matrix, feature importance)

**Model Details:**
- Algorithm: Gradient Boosting Classifier
- Classes: Strong Drop (-2), Modest Drop (-1), Neutral (0), Modest Rise (+1), Strong Rise (+2)
- Feature Scaling: StandardScaler normalization

**Top Important Features:**
1. US02Y_Yield (23.4%)
2. net_sentiment_score (11.7%)
3. Yield_Curve_10Y_2Y (10.5%)
4. Volume_ratio_vs_5days (10.0%)
5. DXY_Close (9.5%)

---

### 7. Backtesting

Simulate trading strategies using trained models.

```bash
python src/backtesting/run_backtest.py
```

**Input:**
- `data/MasterDataset_Enriched.csv`
- Trained models from Step 6

**Output:** `src/backtesting/outputs/`
- `binary/` - Binary model backtest results
- `multiclass/` - Multi-class model backtest results
- `strategy_comparison.csv` - Strategy performance comparison

**Custom Backtest Example:**

```python
from src.backtesting.model_predictor import ModelPredictor
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategies import BinaryStrategy
import pandas as pd

# Load enriched dataset from centralized data directory
df = pd.read_csv('data/MasterDataset_Enriched.csv')

# Load trained model
predictor = ModelPredictor(
    model_path='src/models/binary_model/models/xgboost_binary_classifier.pkl',
    model_type='binary'
)

# Define trading strategy
strategy = BinaryStrategy(name='Long/Short Strategy')

# Run backtest
engine = BacktestEngine(
    predictor=predictor,
    strategy=strategy,
    initial_capital=100000,
    horizon=1  # 1-day holding period
)

results = engine.run(df, date_col='Date', return_col='return')

# View performance
engine.print_summary()
engine.plot_equity_curve(save_path='equity_curve.png')
engine.export_results(output_dir='backtesting_results/')
```

**Available Trading Strategies:**

| Strategy | Description | Position Logic |
|----------|-------------|----------------|
| `BinaryStrategy` | Long/Short | Long on Up (+1), Short on Down (-1) |
| `BinaryLongOnlyStrategy` | Long Only | Long on Up (+1), Flat on Down (0) |
| `MultiClassStrategy` | Scaled Positions | Scaled from -1.0 to +1.0 by magnitude |
| `ThresholdStrategy` | Confidence Filter | Trade only when confidence > threshold |
| `KellyStrategy` | Kelly Criterion | Optimal bet sizing based on edge |

**Performance Metrics:**
- Total Return (%)
- Win Rate
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown
- Profit Factor (wins/losses ratio)
- Average Win/Loss

**Output Files:**
```
src/backtesting/outputs/
├── binary/
│   ├── trades.csv                  # Trade-by-trade results
│   ├── equity_curve.csv            # Portfolio value over time
│   ├── metrics.json                # Performance metrics
│   ├── equity_curve.png            # Equity curve visualization
│   └── returns_distribution.png    # Return distribution plots
└── multiclass/
    └── [same structure]
```

**Important Assumptions:**

Backtest results assume:
- Zero transaction costs (no commissions/fees)
- No slippage (trades at exact closing price)
- Perfect liquidity
- 1-day holding period
- Close-to-close returns

---

### 8. Run Tests

Verify everything works correctly.

```bash
# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=src --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/acceptance/ -v
```

## Project Structure

```
FED_Watcher/
├── data/                             # Centralized data directory
│   ├── raw_data/                     # Raw FOMC transcript files (.txt)
│   ├── processed/                    # Processed outputs
│   │   ├── powell_speeches_output/   # Extracted Powell speeches
│   │   └── stock_data/               # S&P 500 historical data
│   ├── sentiment_results/            # NLP sentiment outputs
│   ├── MasterDataset_Final.csv       # Combined stock + sentiment
│   └── MasterDataset_Enriched.csv    # Final enriched dataset
├── src/
│   ├── nlp/                          # NLP & sentiment analysis
│   │   ├── semantic_sentiment_analyser.py  # Main sentiment pipeline
│   │   └── finbert_analyser.py            # FinBERT verification
│   ├── enrichment/                   # Market data enrichment
│   │   ├── enrich_with_market_data.py     # Add VIX, DXY, yields
│   │   └── master_pipeline.py             # Combine stock + sentiment
│   ├── models/
│   │   ├── binary_model/             # Binary classification
│   │   │   ├── main.py               # Training pipeline
│   │   │   ├── run_grid_search.py    # Hyperparameter tuning
│   │   │   └── src/
│   │   │       ├── data_preparation.py    # Data loading & preprocessing
│   │   │       ├── model_training.py      # XGBoost training
│   │   │       └── visualization.py       # Model visualizations
│   │   └── multi_classification/     # Multi-class classification
│   │       ├── run_grid_search.py    # Hyperparameter tuning
│   │       └── src/
│   │           ├── enhanced_multi_class_training.py
│   │           ├── data_preparation.py
│   │           ├── model_training.py
│   │           └── visualization.py
│   ├── backtesting/                  # Backtesting engine
│   │   ├── model_predictor.py        # Unified model interface
│   │   ├── strategies.py             # Trading strategies
│   │   ├── backtest_engine.py        # Event-driven engine
│   │   ├── report_generator.py       # Performance reporting
│   │   └── run_backtest.py           # Main execution script
│   ├── stock_data/                   # Stock data utilities
│   │   └── get_stock_data.py         # Download S&P 500 data
│   └── cleaning/                     # Data cleaning scripts
│       └── extract_powell_speeches.py # Extract Powell speeches
├── tests/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── acceptance/                   # Acceptance tests
├── docs/                             # documentation
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── pytest.ini                        # Pytest configuration
├── pyproject.toml                    # Build configuration
├── .flake8                           # Linting rules
├── .coveragerc                       # Coverage configuration
├── DEVELOPMENT_GUIDELINES.md         # GitFlow workflow
└── README.md                         # This file
```

## Models & Performance

### Binary Classification Model

**Architecture**: XGBoost Binary Classifier

**Training Configuration:**
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.85

**Dataset Split:**
- Training: 28 announcement days (2020-09 to 2024-01)
- Testing: 12 announcement days (2024-03 to 2024-12)

**Performance:**

| Metric | Train Set | Test Set |
|--------|-----------|----------|
| Accuracy | 100.0% | 57.1% |
| Precision | 100.0% | 100.0% |
| Recall | 100.0% | 50.0% |
| F1-Score | 100.0% | 66.7% |

### Multi-Class Classification Model

**Architecture**: Gradient Boosting Classifier with StandardScaler

**Classes:**
- `-2`: Strong Drop (< -1.0%)
- `-1`: Modest Drop (-1.0% to -0.3%)
- `0`: Neutral (-0.3% to +0.3%)
- `+1`: Modest Rise (+0.3% to +1.0%)
- `+2`: Strong Rise (> +1.0%)

**Key Features:** All 15 enriched features with StandardScaler normalization

## Testing

FED Watcher includes a comprehensive test suite with 95%+ code coverage.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/acceptance/ -v        # Acceptance tests

# Run with coverage report
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term-missing

# Run tests in parallel
pytest -n auto

# Run specific markers
pytest -m critical                 # Critical tests only
pytest -m integration              # Integration tests only
```

### Test Categories

**Unit Tests** (`tests/unit/`)
- Individual component testing
- Fast execution (< 1 second per test)
- Mock external dependencies

**Integration Tests** (`tests/integration/`)
- Component interaction testing
- Database/file system operations
- API integrations

**Acceptance Tests** (`tests/acceptance/`)
- End-to-end workflows
- Sprint acceptance criteria validation
- User story verification

### Test Markers

Tests are organized with pytest markers:
- `@pytest.mark.critical` - Critical tests that must pass
- `@pytest.mark.slow` - Tests taking > 5 seconds
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.acceptance` - Acceptance tests
- `@pytest.mark.unit` - Unit tests

## Development Workflow

FED Watcher uses **GitFlow** branching strategy for collaborative development.

### Core Branches

- **`main`** - Production branch (tagged releases only)
- **`develop`** - Integration branch for active development

### Feature Development

```bash
# 1. Start new feature
git checkout develop
git pull origin develop
git flow feature start FED-123-description

# 2. Develop and commit (use Conventional Commits)
git add .
git commit -m "feat(nlp): add sentiment aggregation logic"

# 3. Publish feature for PR
git flow feature publish

# 4. Create Pull Request on GitHub
#    - Target: develop branch
#    - Assign reviewer
#    - Wait for approval

# 5. Merge via GitHub (Squash and Merge)
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

Examples:
feat(data): Add script to download S&P 500 prices
fix(nlp): Correct speaker attribution logic
docs(readme): Update installation instructions
test(backtest): Add unit tests for strategies
refactor(model): Simplify feature selection logic
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `perf`

### Code Quality Standards

Before committing, ensure code passes quality checks:

```bash
# Format with Black
black src/ tests/

# Lint with Flake8
flake8 src/ --count --statistics

# Analyze with Pylint
pylint src/ --fail-under=7.0

# Run tests
pytest
```

### Branch Types

- `feature/*` - New features (branch from `develop`, merge to `develop`)
- `bugfix/*` - Bug fixes (branch from `develop`, merge to `develop`)
- `hotfix/*` - Critical production bugs (branch from `main`, merge to `main` AND `develop`)
- `release/*` - Release preparation (branch from `develop`, merge to `main` AND `develop`)

## CI/CD Pipeline

Automated GitHub Actions pipeline runs on every push and pull request to `main` and `develop`.

### Pipeline Stages

1. **Lint** - Code quality checks (Black, Flake8, Pylint)
2. **Security** - Vulnerability scanning (Bandit, Safety)
3. **Unit Tests** - Multi-version testing (Python 3.9, 3.10, 3.11)
4. **Integration Tests** - Component integration validation
5. **Cross-Validation** - ML-specific validation tests
6. **Acceptance Tests** - End-to-end workflow verification
7. **Build** - Python package distribution (main branch only)
8. **Notify** - Pipeline status report

### Quality Gates

All stages must pass for PR approval:
- Code formatting (Black)
- Linting score ≥ 7.0 (Pylint)
- Code coverage ≥ 95%
- Zero critical security vulnerabilities
- All tests passing

### Badge Status

[![Python CI/CD Pipeline](https://github.com/JacobDrizzle/FED_Watcher/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/JacobDrizzle/FED_Watcher/actions/workflows/ci-cd.yml)

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch from `develop`
2. **Follow GitFlow workflow** as described in `DEVELOPMENT_GUIDELINES.md`
3. **Write tests** for new features (maintain ≥95% coverage)
4. **Follow code style** (Black formatting, Flake8 compliant)
5. **Use Conventional Commits** for commit messages
6. **Create Pull Request** targeting `develop` branch
7. **Ensure CI/CD passes** before requesting review

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install

# Run quality checks
black --check src/ tests/
flake8 src/
pylint src/ --fail-under=7.0
pytest --cov=src
```

## Model Information

### FinBERT

This project uses **FinBERT** for financial sentiment analysis.

- **Model**: `ProsusAI/finbert`
- **Source**: [Hugging Face Model Hub](https://huggingface.co/ProsusAI/finbert)
- **Description**: BERT-based model pre-trained on financial corpus, specialized for financial sentiment analysis tasks
- **License**: Apache 2.0

### Data Sources

- **FOMC Press Conference Transcripts**: [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanpaserman/fed-press-release-text) by Jonathan Paserman
  - Contains transcripts of Federal Reserve Chair press conferences
  - Required input for the sentiment analysis pipeline
- **S&P 500 Data**: Historical market data via `yfinance`
- **Market Indicators**: VIX, DXY, Treasury yields via Yahoo Finance and FRED (`pandas-datareader`)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Jonathan Paserman** for the [Fed Press Release Text dataset](https://www.kaggle.com/datasets/jonathanpaserman/fed-press-release-text) on Kaggle
- **FinBERT** by ProsusAI for financial sentiment analysis
- **Federal Reserve** for public speech transcripts
- **yfinance** and **pandas-datareader** for market data access

## Contact & Support

- **Repository**: [https://github.com/JacobDrizzle/FED_Watcher](https://github.com/JacobDrizzle/FED_Watcher)
- **Issues**: [GitHub Issues](https://github.com/JacobDrizzle/FED_Watcher/issues)
- **Documentation**: See `docs/` directory and module-specific READMEs

## Release Notes

### v1.0.0 (First Release)

**Features:**
- ✅ NLP sentiment analysis pipeline with FinBERT
- ✅ Market data enrichment with 7 macro indicators
- ✅ Binary classification model (Up/Down prediction)
- ✅ Multi-class classification model (5-class magnitude prediction)
- ✅ Event-driven backtesting engine with 5 trading strategies
- ✅ Comprehensive test suite (Unit, Integration, Acceptance) - 95%+ coverage
- ✅ GitHub Actions CI/CD pipeline
- ✅ GitFlow branching workflow
- ✅ Production-grade code quality standards
- ✅ Centralized data directory structure

**Dataset:**
- 1,761 daily observations (2018-2024)
- 40 Federal Reserve announcement events
- 20 features (sentiment + macro + market data)

**Performance:**
- Binary Model: 83.3% test accuracy
- Multi-Class Model: Enhanced feature engineering with StandardScaler
- Backtesting: Multiple strategies with full P&L tracking
- Test Coverage: 95.19% (300 tests passing)