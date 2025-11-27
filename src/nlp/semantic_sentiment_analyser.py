import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
import nltk
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Setup and Model Loading ---


def setup_nltk():
    """Checks for and downloads the required NLTK 'punkt' package for sentence tokenization."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK's 'punkt' package...")
        nltk.download("punkt")


def load_models():
    """
    Loads both the FinBERT model for sentiment analysis and the SentenceTransformer
    model for creating semantic embeddings.
    """
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load FinBERT for sentiment analysis
    finbert_model_name = "ProsusAI/finbert"
    finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model_name)
    finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_name)
    finbert_model.to(device)
    print("FinBERT model loaded successfully.")

    # Load SentenceTransformer for semantic chunking
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded successfully.")

    return finbert_tokenizer, finbert_model, embedding_model, device


# --- Step 2: Core NLP Functions ---


def preprocess_text_for_tokenization(text):
    """
    Uses regex to fix common tokenization errors with numbers (e.g., '8. 4 percent' -> '8.4 percent')
    before splitting into sentences.
    """
    return re.sub(r"(\d)\. (\d)", r"\1.\2", text)


def semantic_chunker(text, embedding_model, percentile_threshold=90):
    """
    Splits a long text into semantically coherent chunks based on sentence similarity.
    A split occurs when the semantic similarity between adjacent sentences drops
    below a dynamically calculated threshold.
    """
    cleaned_text = preprocess_text_for_tokenization(text)
    sentences = nltk.sent_tokenize(cleaned_text)

    if len(sentences) < 3:
        return [" ".join(sentences)]

    # Generate embeddings for each sentence
    embeddings = embedding_model.encode(sentences, show_progress_bar=False)

    # Calculate cosine similarity between adjacent sentences
    similarities = [
        cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(embeddings) - 1)
    ]

    # Use the percentile as a dynamic threshold to find significant topic shifts
    breakpoint_threshold = np.percentile(similarities, percentile_threshold)

    chunks = []
    current_chunk_start_index = 0
    for i, sim in enumerate(similarities):
        if sim < breakpoint_threshold:
            # Join sentences from the start of the current chunk to the breakpoint
            chunks.append(" ".join(sentences[current_chunk_start_index : i + 1]))
            current_chunk_start_index = i + 1

    # Add the final chunk
    chunks.append(" ".join(sentences[current_chunk_start_index:]))

    return chunks


def analyze_sentiment(text, tokenizer, model, device):
    """
    Performs sentiment analysis on a single text chunk using the FinBERT model.
    """
    if not text or not isinstance(text, str):
        return "neutral", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["positive", "negative", "neutral"]
    scores = predictions[0].cpu().tolist()
    results = dict(zip(labels, scores))
    predicted_label = labels[torch.argmax(predictions[0]).item()]

    return predicted_label, results


# --- Step 3: Main Processing Pipeline ---


def process_speech_directory(input_dir, chunk_output_file, doc_output_file):
    """
    Main function to orchestrate the entire analysis pipeline:
    1. Loads models.
    2. Scans for speech files in the input directory.
    3. For each speech:
        a. Performs semantic chunking.
        b. Runs sentiment analysis on each chunk.
    4. Aggregates results for each document.
    5. Saves both chunk-level and document-level results to CSV files.
    """
    setup_nltk()

    tokenizer, model, embedding_model, device = load_models()

    print(f"\nScanning for speech files in '{input_dir}'...")
    try:
        speech_files = [f for f in os.listdir(input_dir) if f.endswith("_powell_speeches.csv")]
        if not speech_files:
            print(
                f"Error: No speech files ending with '_powell_speeches.csv' found in '{input_dir}'."
            )
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    print(f"Found {len(speech_files)} speech files to process.")

    all_chunk_results = []
    all_doc_results = []

    print("\nAnalyzing speeches using semantic chunking...")
    for filename in tqdm(speech_files, desc="Processing Files"):
        filepath = os.path.join(input_dir, filename)

        # Extract date from filename (e.g., '20230503_...')
        date_match = re.search(r"(\d{8})", filename)
        if not date_match:
            print(f"Warning: Could not parse date from filename '{filename}'. Skipping.")
            continue

        meeting_date_str = date_match.group(1)
        meeting_date = f"{meeting_date_str[:4]}-{meeting_date_str[4:6]}-{meeting_date_str[6:]}"

        df = pd.read_csv(filepath)
        if "paragraph_text" not in df.columns:
            print(f"Warning: 'paragraph_text' column not found in '{filename}'. Skipping.")
            continue

        full_text = " ".join(df["paragraph_text"].dropna().astype(str))

        # --- Use the new semantic chunker ---
        chunks = semantic_chunker(full_text, embedding_model, percentile_threshold=90)
        # Filter out any very short or empty chunks that might result
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]

        if not chunks:
            print(f"Warning: No valid chunks created for '{filename}'. Skipping.")
            continue

        # Analyze sentiment for each chunk and collect scores
        date_chunk_scores = []
        for chunk in chunks:
            label, scores = analyze_sentiment(chunk, tokenizer, model, device)

            all_chunk_results.append(
                {
                    "meeting_date": meeting_date,
                    "chunk_text": chunk,
                    "sentiment_label": label,
                    "positive_score": scores["positive"],
                    "negative_score": scores["negative"],
                    "neutral_score": scores["neutral"],
                }
            )
            date_chunk_scores.append(scores)

        # Aggregate the results for the current document
        if date_chunk_scores:
            temp_df = pd.DataFrame(date_chunk_scores)
            avg_scores = temp_df.mean()
            # Get chunk labels specifically for the current date to count them
            chunk_labels = [
                res["sentiment_label"]
                for res in all_chunk_results
                if res["meeting_date"] == meeting_date
            ]
            label_counts = pd.Series(chunk_labels).value_counts()

            doc_result = {
                "meeting_date": meeting_date,
                "positive_chunk_count": label_counts.get("positive", 0),
                "negative_chunk_count": label_counts.get("negative", 0),
                "neutral_chunk_count": label_counts.get("neutral", 0),
                "avg_positive_score": avg_scores["positive"],
                "avg_negative_score": avg_scores["negative"],
                "avg_neutral_score": avg_scores["neutral"],
                "net_sentiment_score": avg_scores["positive"] - avg_scores["negative"],
            }
            all_doc_results.append(doc_result)

    # --- Step 4: Save Results and Final Validation ---

    print(f"\nSaving all chunk-level results to '{chunk_output_file}'...")
    chunk_df = pd.DataFrame(all_chunk_results)
    os.makedirs(os.path.dirname(chunk_output_file), exist_ok=True)
    chunk_df.to_csv(chunk_output_file, index=False)
    print("Chunk-level data saved.")

    print(f"Saving all aggregated document-level results to '{doc_output_file}'...")
    doc_df = pd.DataFrame(all_doc_results).sort_values(by="meeting_date").reset_index(drop=True)

    # Classify final sentiment based on the net score
    def classify_sentiment(net_score, threshold=0.015):
        if net_score > threshold:
            return "Positive"
        elif net_score < -threshold:
            return "Negative"
        else:
            return "Neutral"

    doc_df["final_sentiment_label"] = doc_df["net_sentiment_score"].apply(classify_sentiment)

    os.makedirs(os.path.dirname(doc_output_file), exist_ok=True)
    doc_df.to_csv(doc_output_file, index=False)
    print("Document-level data saved.")

    print("\n--- Final Validation Summary ---")
    if not chunk_df.empty and not doc_df.empty:
        print("\nOverall Chunk-Level Sentiment Distribution:")
        print(chunk_df["sentiment_label"].value_counts(normalize=True).round(3))

        print("\nAggregated Document-Level Sentiment Distribution (Based on Net Score):")
        print(doc_df["final_sentiment_label"].value_counts(normalize=True).round(3))
    else:
        print("\nNo results were generated. Please check your input files.")

    print("\nProcessing complete.")


# --- Step 5: Script Entry Point ---


def main():
    parser = argparse.ArgumentParser(
        description="Process Jerome Powell's speeches using semantic chunking and FinBERT for sentiment analysis."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing speech CSV files.",
    )
    parser.add_argument(
        "--chunk_output",
        type=str,
        default="sentiment_results/semantic_chunk_sentiments.csv",
        help="Path to save the chunk-level output CSV.",
    )
    parser.add_argument(
        "--doc_output",
        type=str,
        default="sentiment_results/semantic_document_sentiments.csv",
        help="Path to save the document-level output CSV.",
    )

    args = parser.parse_args()
    process_speech_directory(args.input_dir, args.chunk_output, args.doc_output)


if __name__ == "__main__":
    main()
