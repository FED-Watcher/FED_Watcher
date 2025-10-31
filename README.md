# FED Watcher

A machine learning system designed to predict short-term U.S. stock market movements by analyzing the sentiment of public announcements from the Federal Reserve Chair.

## 1. Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python (3.9 or higher recommended)
- Git
- A virtual environment manager (e.g., `venv`, `conda`)

### Installation
Follow these steps to set up your development environment.

1.  **Clone the repository:**
    git clone https://github.com/JacobDrizzle/FED_Watcher.git
    cd fed-watcher
    

2.  **Create and activate a virtual environment:**
    We recommend using `venv` to keep project dependencies isolated.

    # Create the virtual environment
    python -m venv finbert_env

    # Activate the environment
    # On Windows:
    .\finbert_env\Scripts\activate
    # On macOS/Linux:
    source finbert_env/bin/activate
    
3.  **Install required libraries:**
    All project dependencies are listed in the `requirements.txt` file.
    
    pip install -r requirements.txt
    
## 2. Usage: Sentiment Analysis Script
The core NLP functionality is handled by the `finbert_analyser.py` script located in the `src/nlp/` directory. This script is responsible for loading the FinBERT model and can be run in several modes.

### Mode: `verify`
This mode runs a series of pre-defined functional tests to ensure the FinBERT model is loaded correctly and is producing plausible sentiment predictions. It's a quick way to confirm your environment is set up correctly.

**Command:**
python src/nlp/finbert_analyser.py --mode verify


### Mode: `batch`
This mode runs sentiment analysis on all `.txt` files within a specified input directory. It processes each file as a single document and saves the aggregated sentiment scores into an output CSV file.

**Arguments:**
-   `--input_dir`: The path to the directory containing the text files to be analyzed.
-   `--output_file`: The full path where the resulting CSV file will be saved.

**Example Command:**
python src/nlp/finbert_analyser.py --mode batch --input_dir data/raw/transcripts --output_file data/processed/transcript_sentiments.csv

### Mode: `evaluate`
This mode evaluates the model's performance against a labeled dataset. The input must be a CSV file containing `text` and `true_label` columns. The script will predict the sentiment for each piece of text and generate a classification report comparing the predictions to the true labels.

**Arguments:**
-   `--labeled_data`: The path to the labeled CSV file used for evaluation.

**Example Command:**
python src/nlp/finbert_analyser.py --mode evaluate --labeled_data data/validation/labeled_financial_phrases.csv

## 3. Model Information
This project relies on a pre-trained language model specialized for financial text.

-   **Model:** `ProsusAI/finbert`
-   **Source:** The model can be found on the Hugging Face Model Hub: [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
-   **Description:** FinBERT is a BERT-based model that is further pre-trained on a large financial corpus, making it highly effective for financial sentiment analysis tasks.

## 4. Project Structure
The repository is organized as follows:

fed-watcher/
├── .gitignore          # Specifies files and folders for Git to ignore
├── DEVELOPMENT_GUIDELINES.md # Our official GitFlow and coding standards
├── README.md           # This file
├── requirements.txt    # Project dependencies for pip
│
├── data/               # For raw and processed data (ignored by Git)
│
├── docs/               # For documentation and diagrams
│   └── images/
│
└── src/                # For all project source code
    └── nlp/
        └── finbert_analyser.py