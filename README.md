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
    
## 2. Usage: Sentiment Analysis Pipeline

The core NLP functionality is handled by the scripts located in the `src/nlp/` directory.

### `semantic_sentiment_analyser.py`

This is the primary script for the sentiment analysis pipeline. It takes raw speech files, uses **semantic chunking** to group related sentences into meaningful paragraphs, runs FinBERT sentiment analysis on each chunk, and aggregates the results.

The script generates two critical output files:
1.  A **chunk-level CSV** with sentiment scores for every individual semantic chunk.
2.  A **document-level CSV** that aggregates all chunk scores for a given speech, providing a single, comprehensive sentiment summary for each event.

**Arguments:**
-   `--input_dir`: **(Required)** Path to the directory containing the source speech files (e.g., `_powell_speeches.csv`).
-   `--chunk_output`: (Optional) Path to save the detailed chunk-level sentiment CSV. Defaults to `sentiment_results/semantic_chunk_sentiments.csv`.
-   `--doc_output`: (Optional) Path to save the aggregated document-level sentiment CSV. Defaults to `sentiment_results/semantic_document_sentiments.csv`.

**Example Command:**
python src/nlp/semantic_sentiment_analyser.py --input_dir data/processed/powell_speeches

### `finbert_analyser.py` (Verification Tool)
This is a lightweight utility script for verifying that the FinBERT model is installed and running correctly in your environment.

**Command:**
python src/nlp/finbert_analyser.py --mode verify

## 3. Model Information
This project relies on a pre-trained language model specialized for financial text.

-   **Model:** `ProsusAI/finbert`
-   **Source:** The model can be found on the Hugging Face Model Hub: [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
-   **Description:** FinBERT is a BERT-based model that is further pre-trained on a large financial corpus, making it highly effective for financial sentiment analysis tasks.