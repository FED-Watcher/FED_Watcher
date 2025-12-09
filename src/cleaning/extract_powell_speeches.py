import re
import csv
import os
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default paths relative to project root
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "powell_speeches"


def extract_meeting_date(filename):
    """
    Extract meeting date from filename.
    Expected format: FOMCpresconf20200916.txt
    Returns: 2020-09-16
    """
    match = re.search(r"(\d{8})", filename)
    if match:
        date_str = match.group(1)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        return f"{year}-{month}-{day}"
    return "UNKNOWN_DATE"


def clean_text(text):
    """
    Clean text by removing XML tags, fixing encoding issues, and normalizing.
    """
    # Remove XML-style tags
    text = re.sub(r"<[^>]+>", "", text)

    # Fix common encoding issues - comprehensive list
    encoding_fixes = {
        'â€"': "—",
        'â€"': "-",
        "â€œ": '"',
        "â€\u009d": '"',
        "â€": '"',
        "â€™": "'",
        "â€˜": "'",
        "Â½": "1/2",
        "Â¼": "1/4",
        "Â¾": "3/4",
        "â€¦": "...",
        "Ã¶": "o",
        "Ã¤": "a",
        "Ã¼": "u",
        "Ã©": "e",
        "Ã¨": "e",
        "Ã ": "a",
        "Ã§": "c",
        "Ã±": "n",
        "Ãº": "u",
        "Ã³": "o",
        "Ã­": "i",
        "Ã¡": "a",
        "Ãƒ": "A",
        "Ã": "",
        "â": "",
        "Â": "",
        "€": "",
        "Å": "",
        "Ä": "",
        "Ö": "",
    }

    for wrong, right in encoding_fixes.items():
        text = text.replace(wrong, right)

    # Remove any remaining non-ASCII characters that look like artifacts
    # Keep only: letters, numbers, common punctuation, and spaces
    text = re.sub(r'[^\w\s\-\'".,;:!?()\[\]{}/$%&]', "", text)

    # Fix multiple punctuation marks
    text = re.sub(r"\.{2,}", "...", text)  # Multiple periods to ellipsis
    text = re.sub(r"\s*-\s*-\s*", " - ", text)  # Fix double dashes

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    # Fix common spacing issues around punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)  # Add space after punctuation

    # Remove any remaining double spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_powell_speeches(file_path):
    """
    Extract all speeches by CHAIR POWELL from the transcript.
    Returns a list of dictionaries with meeting_date and paragraph_text.
    """
    # Read the file with error handling for different encodings
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    content = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"Error: Could not read file {file_path} with any encoding")
        return []

    # Extract meeting date from filename
    filename = os.path.basename(file_path)
    meeting_date = extract_meeting_date(filename)

    # Split content by <NAME> tags to identify speakers
    # Pattern: <NAME>SPEAKER NAME</NAME>. Speech text...
    name_pattern = r"<NAME>(.*?)</NAME>\.\s*(.*?)(?=<NAME>|$)"
    matches = re.finditer(name_pattern, content, re.DOTALL)

    powell_speeches = []

    for match in matches:
        speaker = match.group(1).strip()
        speech_text = match.group(2).strip()

        # Check if this is Chair Powell
        if speaker == "CHAIR POWELL":
            # Clean the text
            cleaned_text = clean_text(speech_text)

            # Skip if text is too short (likely an artifact)
            if len(cleaned_text) < 10:
                continue

            # Split into paragraphs
            # Consider both double newlines and single newlines as paragraph separators
            paragraphs = re.split(r"\n\n+", cleaned_text)

            for paragraph in paragraphs:
                # Clean each paragraph
                paragraph = paragraph.replace("\n", " ").strip()
                paragraph = re.sub(r"\s+", " ", paragraph)

                # Skip empty or very short paragraphs
                if len(paragraph) > 20:  # Minimum meaningful paragraph length
                    powell_speeches.append(
                        {"meeting_date": meeting_date, "paragraph_text": paragraph}
                    )

    return powell_speeches


def save_to_csv(data, output_file):
    """
    Save extracted speeches to CSV file.
    """
    if not data:
        print("No data to save!")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["meeting_date", "paragraph_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Successfully saved {len(data)} paragraphs to {output_file}")


def process_single_file(input_file, output_file=None):
    """
    Process a single FOMC transcript file.
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return []

    print(f"Processing: {input_file}")

    # Extract speeches
    speeches = extract_powell_speeches(input_file)

    print(f"Extracted {len(speeches)} paragraphs from Chair Powell's speeches")

    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_powell_speeches.csv"

    # Save to CSV
    save_to_csv(speeches, output_file)

    return speeches


def process_directory(input_dir, output_dir=None):
    """
    Process all .txt files in a directory.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found!")
        return

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all .txt files
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} transcript files")
    print("=" * 60)

    all_speeches = []

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] {txt_file}")
        input_path = os.path.join(input_dir, txt_file)

        if output_dir:
            base_name = os.path.splitext(txt_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}_powell_speeches.csv")
        else:
            output_path = None

        speeches = process_single_file(input_path, output_path)
        all_speeches.extend(speeches)

    # Also create a combined CSV with all speeches
    if output_dir:
        combined_output = os.path.join(output_dir, "all_powell_speeches_combined.csv")
    else:
        combined_output = "all_powell_speeches_combined.csv"

    print("\n" + "=" * 60)
    print("Creating combined file with all speeches...")
    save_to_csv(all_speeches, combined_output)
    print(f"\nTotal speeches across all files: {len(all_speeches)}")
    print(f"Output directory: {output_dir or 'current directory'}")


def main():
    """
    Main function - processes all .txt files from data/raw_data directory.
    Outputs to data/processed/powell_speeches directory.
    """
    print("=" * 60)
    print("FOMC Chair Powell Speech Extractor")
    print("Enhanced version with improved text cleaning")
    print("=" * 60)

    # Use project-relative paths
    input_directory = str(DEFAULT_INPUT_DIR)
    output_directory = str(DEFAULT_OUTPUT_DIR)

    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")

    process_directory(input_directory, output_directory)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
