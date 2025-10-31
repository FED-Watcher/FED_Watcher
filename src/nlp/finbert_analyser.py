import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
from sklearn.metrics import accuracy_score, classification_report

def load_finbert():
    """Load the FinBERT model and tokenizer"""
    print("Loading FinBERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = 'ProsusAI/finbert'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    print("Model loaded successfully!")
    return tokenizer, model, device

def analyze_sentiment(text, tokenizer, model, device):
    """Analyze sentiment of a single financial text string"""
    if not text or not isinstance(text, str):
        return 'neutral', {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ['positive', 'negative', 'neutral']
    scores = predictions[0].cpu().tolist()
    results = dict(zip(labels, scores))
    predicted_label = labels[torch.argmax(predictions[0]).item()]

    return predicted_label, results

def run_functional_verification(tokenizer, model, device):
    """Runs sentiment analysis on a predefined set of test sentences to verify functionality."""
    print("\n" + "="*60)
    print("Running Functional Verification")
    print("="*60 + "\n")

    test_cases = [
        {"text": "The economic outlook is strong with robust growth.", "expected": "positive"},
        {"text": "Downward market volatility remains a significant concern for investors.", "expected": "negative"},
        {"text": "The committee will continue to monitor the data.", "expected": "neutral"}
    ]

    all_passed = True
    for i, case in enumerate(test_cases, 1):
        sentence = case["text"]
        expected = case["expected"]
        predicted, scores = analyze_sentiment(sentence, tokenizer, model, device)
        
        status = "PASS" if predicted == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
            
        print(f"Test Case {i}: {sentence}")
        print(f"Expected: {expected.upper()} | Predicted: {predicted.upper()} -> {status}")
        print(f"Scores: {', '.join([f'{k}: {v:.3f}' for k, v in scores.items()])}")
        print("-" * 60 + "\n")

    print(f"Functional Verification Complete. Overall Status: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def analyze_directory_to_csv(input_dir, output_file, tokenizer, model, device):
    """Analyzes all .txt files in a directory and saves the results to a CSV."""
    print(f"\nStarting batch analysis of directory: {input_dir}")
    
    try:
        filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    if not filenames:
        print("No .txt files found to analyze.")
        return

    results_list = []
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        predicted_label, scores = analyze_sentiment(content, tokenizer, model, device)
        
        results_list.append({
            'filename': filename,
            'sentiment_label': predicted_label,
            'positive_score': scores['positive'],
            'negative_score': scores['negative'],
            'neutral_score': scores['neutral']
        })
        print(f"Analyzed {filename} -> {predicted_label.upper()}")

    df = pd.DataFrame(results_list)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_file, index=False)
    print(f"\nBatch analysis complete. Results saved to {output_file}")

def evaluate_performance(labeled_csv_path, tokenizer, model, device):
    """Evaluates model performance against a labeled validation set."""
    print(f"\nEvaluating performance using labeled data from: {labeled_csv_path}")
    try:
        df = pd.read_csv(labeled_csv_path)
        if not all(col in df.columns for col in ['text', 'true_label']):
            print("Error: Labeled CSV must contain 'text' and 'true_label' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Labeled data file not found at {labeled_csv_path}")
        return

    predictions = []
    for text in df['text']:
        label, _ = analyze_sentiment(text, tokenizer, model, device)
        predictions.append(label)

    true_labels = df['true_label']
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    print(f"\nPerformance Evaluation Report:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

def main():
    parser = argparse.ArgumentParser(description="FinBERT Sentiment Analysis Tool")
    parser.add_argument('--mode', type=str, required=True, choices=['verify', 'batch', 'evaluate'],
                        help="Operating mode: 'verify', 'batch', or 'evaluate'.")
    parser.add_argument('--input_dir', type=str, help="Path to the directory with cleaned text files for batch mode.")
    parser.add_argument('--output_file', type=str, help="Path to save the output CSV file in batch mode.")
    parser.add_argument('--labeled_data', type=str, help="Path to labeled CSV for evaluation mode.")

    args = parser.parse_args()
    
    tokenizer, model, device = load_finbert()

    if args.mode == 'verify':
        run_functional_verification(tokenizer, model, device)
    elif args.mode == 'batch':
        if not args.input_dir or not args.output_file:
            print("Error: --input_dir and --output_file are required for batch mode.")
            return
        analyze_directory_to_csv(args.input_dir, args.output_file, tokenizer, model, device)
    elif args.mode == 'evaluate':
        if not args.labeled_data:
            print("Error: --labeled_data is required for evaluation mode.")
            return
        evaluate_performance(args.labeled_data, tokenizer, model, device)

if __name__ == "__main__":
    main()