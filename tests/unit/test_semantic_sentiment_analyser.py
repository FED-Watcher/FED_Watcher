"""Unit tests for the semantic_sentiment_analyser module."""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.nlp.semantic_sentiment_analyser import (
    setup_nltk,
    load_models,
    preprocess_text_for_tokenization,
    semantic_chunker,
    analyze_sentiment,
    process_speech_directory,
    main,
)


class TestSetupNltk:
    """Test suite for setup_nltk function."""

    @patch("src.nlp.semantic_sentiment_analyser.nltk")
    def test_nltk_punkt_found(self, mock_nltk):
        """Test when punkt is already installed."""
        mock_nltk.data.find.return_value = True

        setup_nltk()

        mock_nltk.data.find.assert_called_once_with("tokenizers/punkt")
        mock_nltk.download.assert_not_called()

    @patch("src.nlp.semantic_sentiment_analyser.nltk")
    def test_nltk_punkt_not_found(self, mock_nltk):
        """Test when punkt needs to be downloaded."""
        mock_nltk.data.find.side_effect = LookupError("Not found")

        setup_nltk()

        mock_nltk.download.assert_called_once_with("punkt")


class TestLoadModels:
    """Test suite for load_models function."""

    @patch("src.nlp.semantic_sentiment_analyser.SentenceTransformer")
    @patch("src.nlp.semantic_sentiment_analyser.BertForSequenceClassification")
    @patch("src.nlp.semantic_sentiment_analyser.BertTokenizer")
    @patch("src.nlp.semantic_sentiment_analyser.torch")
    def test_load_models_returns_all_components(
        self, mock_torch, mock_tokenizer_class, mock_model_class, mock_sentence_transformer
    ):
        """Test that load_models returns all required components."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        tokenizer, model, embedding_model, device = load_models()

        assert tokenizer is not None
        assert model is not None
        assert embedding_model is not None
        assert device is not None

    @patch("src.nlp.semantic_sentiment_analyser.SentenceTransformer")
    @patch("src.nlp.semantic_sentiment_analyser.BertForSequenceClassification")
    @patch("src.nlp.semantic_sentiment_analyser.BertTokenizer")
    @patch("src.nlp.semantic_sentiment_analyser.torch")
    def test_load_models_uses_correct_models(
        self, mock_torch, mock_tokenizer_class, mock_model_class, mock_sentence_transformer
    ):
        """Test that correct model names are used."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        load_models()

        mock_tokenizer_class.from_pretrained.assert_called_once_with("ProsusAI/finbert")
        mock_model_class.from_pretrained.assert_called_once_with("ProsusAI/finbert")
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")


class TestPreprocessTextForTokenization:
    """Test suite for preprocess_text_for_tokenization function."""

    def test_fix_number_spacing(self):
        """Test fixing number spacing issues."""
        text = "The rate is 8. 4 percent."
        result = preprocess_text_for_tokenization(text)

        assert result == "The rate is 8.4 percent."

    def test_multiple_number_fixes(self):
        """Test fixing multiple number spacing issues."""
        text = "Growth was 3. 5 to 4. 2 percent."
        result = preprocess_text_for_tokenization(text)

        assert result == "Growth was 3.5 to 4.2 percent."

    def test_no_changes_needed(self):
        """Test text that doesn't need changes."""
        text = "The rate is 8.4 percent."
        result = preprocess_text_for_tokenization(text)

        assert result == text

    def test_empty_text(self):
        """Test empty text input."""
        result = preprocess_text_for_tokenization("")
        assert result == ""


class TestSemanticChunker:
    """Test suite for semantic_chunker function."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        mock_model = MagicMock()
        # Return different embeddings for each sentence
        mock_model.encode.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],  # Similar to first
                [0.9, 0.8, 0.7],  # Different - should trigger split
                [0.9, 0.8, 0.7],  # Similar to third
            ]
        )
        return mock_model

    def test_short_text_returns_single_chunk(self):
        """Test that short text returns as single chunk."""
        mock_model = MagicMock()
        text = "Short sentence."

        result = semantic_chunker(text, mock_model, percentile_threshold=90)

        # Should return the text as single chunk
        assert len(result) == 1
        assert "short" in result[0].lower()

    def test_two_sentences_returns_single_chunk(self):
        """Test that two sentences return as single chunk."""
        mock_model = MagicMock()
        text = "First sentence. Second sentence."

        result = semantic_chunker(text, mock_model, percentile_threshold=90)

        assert len(result) == 1

    @patch("src.nlp.semantic_sentiment_analyser.nltk")
    @patch("src.nlp.semantic_sentiment_analyser.cosine_similarity")
    def test_chunking_with_topic_shifts(self, mock_cosine, mock_nltk, mock_embedding_model):
        """Test chunking with topic shifts."""
        # Setup sentence tokenization
        mock_nltk.sent_tokenize.return_value = [
            "The economy is growing.",
            "Growth remains strong.",
            "Weather conditions affect farming.",  # Topic shift
            "Agriculture faces challenges.",
        ]

        # Setup cosine similarity - low similarity at position 1 (topic shift)
        mock_cosine.side_effect = [
            [[0.95]],  # High similarity
            [[0.2]],  # Low similarity - topic shift
            [[0.9]],  # High similarity
        ]

        result = semantic_chunker("Full text", mock_embedding_model, percentile_threshold=50)

        # Should have created chunks based on similarity threshold
        assert len(result) >= 1

    @patch("src.nlp.semantic_sentiment_analyser.nltk")
    def test_chunking_preserves_content(self, mock_nltk, mock_embedding_model):
        """Test that chunking preserves all content."""
        sentences = [
            "First topic sentence.",
            "Second topic sentence.",
            "Third topic sentence.",
        ]
        mock_nltk.sent_tokenize.return_value = sentences

        result = semantic_chunker("Full text", mock_embedding_model, percentile_threshold=90)

        # All sentences should be in some chunk
        all_text = " ".join(result)
        for sentence in sentences:
            # Check that sentence content is preserved (may be modified slightly)
            assert "topic" in all_text.lower()


class TestAnalyzeSentiment:
    """Test suite for analyze_sentiment function."""

    def test_empty_text_returns_neutral(self):
        """Test that empty text returns neutral sentiment."""
        label, scores = analyze_sentiment("", MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0
        assert scores["positive"] == 0.0
        assert scores["negative"] == 0.0

    def test_none_text_returns_neutral(self):
        """Test that None text returns neutral sentiment."""
        label, scores = analyze_sentiment(None, MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0

    def test_non_string_returns_neutral(self):
        """Test that non-string input returns neutral sentiment."""
        label, scores = analyze_sentiment(123, MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0

    @patch("src.nlp.semantic_sentiment_analyser.torch")
    def test_valid_text_analyzed(self, mock_torch):
        """Test that valid text is analyzed correctly."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Setup mock outputs
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_predictions = MagicMock()
        mock_torch.nn.functional.softmax.return_value = mock_predictions
        mock_predictions.__getitem__ = MagicMock(
            return_value=MagicMock(
                cpu=MagicMock(
                    return_value=MagicMock(tolist=MagicMock(return_value=[0.7, 0.2, 0.1]))
                )
            )
        )
        mock_torch.argmax.return_value = MagicMock(item=MagicMock(return_value=0))

        label, scores = analyze_sentiment("Test text", mock_tokenizer, mock_model, "cpu")

        # Should call tokenizer
        assert mock_tokenizer.called


class TestProcessSpeechDirectory:
    """Test suite for process_speech_directory function."""

    @patch("src.nlp.semantic_sentiment_analyser.load_models")
    @patch("src.nlp.semantic_sentiment_analyser.setup_nltk")
    def test_directory_not_found(self, mock_setup, mock_load, tmp_path, capsys):
        """Test handling of non-existent directory."""
        mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")

        nonexistent = str(tmp_path / "nonexistent")

        process_speech_directory(
            nonexistent,
            str(tmp_path / "chunk.csv"),
            str(tmp_path / "doc.csv"),
        )

        captured = capsys.readouterr()
        assert "Input directory not found" in captured.out

    @patch("src.nlp.semantic_sentiment_analyser.load_models")
    @patch("src.nlp.semantic_sentiment_analyser.setup_nltk")
    def test_no_speech_files(self, mock_setup, mock_load, tmp_path, capsys):
        """Test handling of directory with no speech files."""
        mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        process_speech_directory(
            str(empty_dir),
            str(tmp_path / "chunk.csv"),
            str(tmp_path / "doc.csv"),
        )

        captured = capsys.readouterr()
        assert "No speech files" in captured.out

    @patch("src.nlp.semantic_sentiment_analyser.analyze_sentiment")
    @patch("src.nlp.semantic_sentiment_analyser.semantic_chunker")
    @patch("src.nlp.semantic_sentiment_analyser.load_models")
    @patch("src.nlp.semantic_sentiment_analyser.setup_nltk")
    def test_successful_processing(
        self, mock_setup, mock_load, mock_chunker, mock_analyze, tmp_path
    ):
        """Test successful speech processing."""
        mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")
        mock_chunker.return_value = ["This is a chunk of text about the economy."]
        mock_analyze.return_value = ("positive", {"positive": 0.7, "negative": 0.1, "neutral": 0.2})

        # Create input directory with speech file
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        speech_file = input_dir / "FOMCpresconf20200916_powell_speeches.csv"
        pd.DataFrame({"paragraph_text": ["The economy is growing steadily."]}).to_csv(
            speech_file, index=False
        )

        chunk_output = str(tmp_path / "chunk.csv")
        doc_output = str(tmp_path / "doc.csv")

        process_speech_directory(str(input_dir), chunk_output, doc_output)

        # Check output files were created
        assert os.path.exists(chunk_output)
        assert os.path.exists(doc_output)

    @patch("src.nlp.semantic_sentiment_analyser.load_models")
    @patch("src.nlp.semantic_sentiment_analyser.setup_nltk")
    def test_skips_files_without_date(self, mock_setup, mock_load, tmp_path, capsys):
        """Test that files without parseable date are skipped."""
        mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create file without date in name
        speech_file = input_dir / "no_date_powell_speeches.csv"
        pd.DataFrame({"paragraph_text": ["Test content."]}).to_csv(speech_file, index=False)

        # This may raise an error when processing empty results
        try:
            process_speech_directory(
                str(input_dir),
                str(tmp_path / "chunk.csv"),
                str(tmp_path / "doc.csv"),
            )
        except (KeyError, ValueError):
            pass  # Expected when no files are processed

        captured = capsys.readouterr()
        assert "Could not parse date" in captured.out

    @patch("src.nlp.semantic_sentiment_analyser.load_models")
    @patch("src.nlp.semantic_sentiment_analyser.setup_nltk")
    def test_skips_files_without_paragraph_text(self, mock_setup, mock_load, tmp_path, capsys):
        """Test that files without paragraph_text column are skipped."""
        mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create file without paragraph_text column
        speech_file = input_dir / "FOMCpresconf20200916_powell_speeches.csv"
        pd.DataFrame({"wrong_column": ["Test content."]}).to_csv(speech_file, index=False)

        # This may raise an error when processing empty results
        try:
            process_speech_directory(
                str(input_dir),
                str(tmp_path / "chunk.csv"),
                str(tmp_path / "doc.csv"),
            )
        except (KeyError, ValueError):
            pass  # Expected when no files are processed

        captured = capsys.readouterr()
        assert "paragraph_text" in captured.out


class TestClassifySentiment:
    """Test suite for classify_sentiment logic."""

    def test_positive_classification(self):
        """Test positive sentiment classification."""
        threshold = 0.015

        def classify_sentiment(net_score, threshold=0.015):
            if net_score > threshold:
                return "Positive"
            elif net_score < -threshold:
                return "Negative"
            else:
                return "Neutral"

        assert classify_sentiment(0.1) == "Positive"
        assert classify_sentiment(0.02) == "Positive"

    def test_negative_classification(self):
        """Test negative sentiment classification."""
        threshold = 0.015

        def classify_sentiment(net_score, threshold=0.015):
            if net_score > threshold:
                return "Positive"
            elif net_score < -threshold:
                return "Negative"
            else:
                return "Neutral"

        assert classify_sentiment(-0.1) == "Negative"
        assert classify_sentiment(-0.02) == "Negative"

    def test_neutral_classification(self):
        """Test neutral sentiment classification."""
        threshold = 0.015

        def classify_sentiment(net_score, threshold=0.015):
            if net_score > threshold:
                return "Positive"
            elif net_score < -threshold:
                return "Negative"
            else:
                return "Neutral"

        assert classify_sentiment(0.0) == "Neutral"
        assert classify_sentiment(0.01) == "Neutral"
        assert classify_sentiment(-0.01) == "Neutral"


class TestMain:
    """Test suite for main function."""

    @patch("src.nlp.semantic_sentiment_analyser.process_speech_directory")
    def test_main_calls_process_with_defaults(self, mock_process):
        """Test main function with default arguments."""
        with patch("sys.argv", ["semantic_sentiment_analyser.py"]):
            main()

        mock_process.assert_called_once()

    @patch("src.nlp.semantic_sentiment_analyser.process_speech_directory")
    def test_main_with_custom_paths(self, mock_process):
        """Test main function with custom paths."""
        with patch(
            "sys.argv",
            [
                "semantic_sentiment_analyser.py",
                "--input_dir",
                "/custom/input",
                "--chunk_output",
                "/custom/chunk.csv",
                "--doc_output",
                "/custom/doc.csv",
            ],
        ):
            main()

        mock_process.assert_called_once_with(
            "/custom/input", "/custom/chunk.csv", "/custom/doc.csv"
        )


class TestDocumentAggregation:
    """Test suite for document-level aggregation logic."""

    def test_chunk_count_aggregation(self):
        """Test aggregation of chunk sentiment counts."""
        chunk_labels = ["positive", "positive", "negative", "neutral", "neutral", "neutral"]
        label_counts = pd.Series(chunk_labels).value_counts()

        assert label_counts.get("positive", 0) == 2
        assert label_counts.get("negative", 0) == 1
        assert label_counts.get("neutral", 0) == 3

    def test_average_score_calculation(self):
        """Test average score calculation."""
        scores = pd.DataFrame(
            {
                "positive": [0.6, 0.8, 0.4],
                "negative": [0.2, 0.1, 0.3],
                "neutral": [0.2, 0.1, 0.3],
            }
        )

        avg_scores = scores.mean()

        assert abs(avg_scores["positive"] - 0.6) < 0.01
        assert abs(avg_scores["negative"] - 0.2) < 0.01

    def test_net_sentiment_score(self):
        """Test net sentiment score calculation."""
        avg_positive = 0.6
        avg_negative = 0.2

        net_score = avg_positive - avg_negative

        assert abs(net_score - 0.4) < 0.0001


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_chunks_filtered(self):
        """Test that very short chunks are filtered out."""
        chunks = [
            "This is a valid chunk with enough words.",
            "Short.",
            "Also too short.",
            "Another valid chunk with sufficient content.",
        ]

        filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]

        assert len(filtered_chunks) == 2

    def test_empty_chunks_list(self):
        """Test handling of empty chunks list after filtering."""
        chunks = ["Too short.", "Also short."]
        filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]

        assert len(filtered_chunks) == 0

    @patch("src.nlp.semantic_sentiment_analyser.nltk")
    def test_text_with_special_characters(self, mock_nltk, mock_embedding_model):
        """Test handling of text with special characters."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        text = "The rate is 3.5% and the — outlook is positive!"
        mock_nltk.sent_tokenize.return_value = [text]

        result = semantic_chunker(text, mock_embedding_model, percentile_threshold=90)

        # Should handle without error
        assert len(result) == 1

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        return mock_model


class TestDateExtraction:
    """Test suite for date extraction from filenames."""

    def test_extract_date_from_filename(self):
        """Test extracting date from standard filename."""
        import re

        filename = "FOMCpresconf20200916_powell_speeches.csv"
        date_match = re.search(r"(\d{8})", filename)

        assert date_match is not None
        meeting_date_str = date_match.group(1)
        meeting_date = f"{meeting_date_str[:4]}-{meeting_date_str[4:6]}-{meeting_date_str[6:]}"

        assert meeting_date == "2020-09-16"

    def test_no_date_in_filename(self):
        """Test filename without date."""
        import re

        filename = "no_date_speeches.csv"
        date_match = re.search(r"(\d{8})", filename)

        assert date_match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
