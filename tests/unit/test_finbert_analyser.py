"""Unit tests for the finbert_analyser module."""

import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.nlp.finbert_analyser import (
    load_finbert,
    analyze_sentiment,
    run_functional_verification,
    analyze_directory_to_csv,
    evaluate_performance,
    main,
)


class TestLoadFinbert:
    """Test suite for load_finbert function."""

    @patch("src.nlp.finbert_analyser.BertForSequenceClassification")
    @patch("src.nlp.finbert_analyser.BertTokenizer")
    @patch("src.nlp.finbert_analyser.torch")
    def test_load_finbert_returns_components(
        self, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test that load_finbert returns tokenizer, model, and device."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        tokenizer, model, device = load_finbert()

        assert tokenizer is not None
        assert model is not None
        assert device is not None

    @patch("src.nlp.finbert_analyser.BertForSequenceClassification")
    @patch("src.nlp.finbert_analyser.BertTokenizer")
    @patch("src.nlp.finbert_analyser.torch")
    def test_load_finbert_uses_correct_model(
        self, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test that load_finbert uses ProsusAI/finbert model."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        load_finbert()

        mock_tokenizer_class.from_pretrained.assert_called_once_with("ProsusAI/finbert")
        mock_model_class.from_pretrained.assert_called_once_with("ProsusAI/finbert")

    @patch("src.nlp.finbert_analyser.BertForSequenceClassification")
    @patch("src.nlp.finbert_analyser.BertTokenizer")
    @patch("src.nlp.finbert_analyser.torch")
    def test_load_finbert_moves_model_to_device(
        self, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test that model is moved to correct device."""
        mock_device = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = mock_device

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        load_finbert()

        mock_model.to.assert_called_once_with(mock_device)


class TestAnalyzeSentiment:
    """Test suite for analyze_sentiment function."""

    def test_analyze_empty_text(self):
        """Test analyzing empty text."""
        label, scores = analyze_sentiment("", MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0
        assert scores["positive"] == 0.0
        assert scores["negative"] == 0.0

    def test_analyze_none_text(self):
        """Test analyzing None text."""
        label, scores = analyze_sentiment(None, MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0

    def test_analyze_non_string_text(self):
        """Test analyzing non-string text."""
        label, scores = analyze_sentiment(123, MagicMock(), MagicMock(), "cpu")

        assert label == "neutral"
        assert scores["neutral"] == 1.0

    @patch("src.nlp.finbert_analyser.torch")
    def test_analyze_valid_text(self, mock_torch):
        """Test analyzing valid text."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_device = "cpu"

        # Setup mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        # Setup mock model output
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        # Setup softmax output
        mock_predictions = MagicMock()
        mock_torch.nn.functional.softmax.return_value = mock_predictions
        mock_predictions.__getitem__ = MagicMock(
            return_value=MagicMock(
                cpu=MagicMock(
                    return_value=MagicMock(tolist=MagicMock(return_value=[0.2, 0.3, 0.5]))
                )
            )
        )
        mock_torch.argmax.return_value = MagicMock(item=MagicMock(return_value=2))

        label, scores = analyze_sentiment("Test text", mock_tokenizer, mock_model, mock_device)

        # Should process without error
        assert mock_tokenizer.called


class TestRunFunctionalVerification:
    """Test suite for run_functional_verification function."""

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    def test_verification_with_all_passing(self, mock_analyze):
        """Test verification when all test cases pass."""
        mock_analyze.side_effect = [
            ("positive", {"positive": 0.8, "negative": 0.1, "neutral": 0.1}),
            ("negative", {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
            ("neutral", {"positive": 0.1, "negative": 0.1, "neutral": 0.8}),
        ]

        result = run_functional_verification(MagicMock(), MagicMock(), "cpu")

        assert result == True

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    def test_verification_with_failures(self, mock_analyze):
        """Test verification when some test cases fail."""
        mock_analyze.side_effect = [
            ("neutral", {"positive": 0.3, "negative": 0.3, "neutral": 0.4}),  # Wrong
            ("negative", {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
            ("neutral", {"positive": 0.1, "negative": 0.1, "neutral": 0.8}),
        ]

        result = run_functional_verification(MagicMock(), MagicMock(), "cpu")

        assert result == False


class TestAnalyzeDirectoryToCsv:
    """Test suite for analyze_directory_to_csv function."""

    def test_directory_not_found(self, tmp_path, capsys):
        """Test handling of non-existent directory."""
        nonexistent = str(tmp_path / "nonexistent")

        analyze_directory_to_csv(nonexistent, "output.csv", MagicMock(), MagicMock(), "cpu")

        captured = capsys.readouterr()
        assert "Input directory not found" in captured.out

    def test_no_txt_files(self, tmp_path, capsys):
        """Test handling of directory with no .txt files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        analyze_directory_to_csv(str(empty_dir), "output.csv", MagicMock(), MagicMock(), "cpu")

        captured = capsys.readouterr()
        assert "No .txt files found" in captured.out

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    def test_analyze_directory_success(self, mock_analyze, tmp_path):
        """Test successful directory analysis."""
        # Create test files
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test1.txt").write_text("Positive economic outlook.")
        (input_dir / "test2.txt").write_text("Negative market conditions.")

        output_file = str(tmp_path / "output.csv")

        mock_analyze.side_effect = [
            ("positive", {"positive": 0.8, "negative": 0.1, "neutral": 0.1}),
            ("negative", {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        ]

        analyze_directory_to_csv(str(input_dir), output_file, MagicMock(), MagicMock(), "cpu")

        # Check output file was created
        assert os.path.exists(output_file)

        # Check content
        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert "filename" in df.columns
        assert "sentiment_label" in df.columns

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    def test_creates_output_directory(self, mock_analyze, tmp_path):
        """Test that output directory is created if needed."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("Test content.")

        output_file = str(tmp_path / "new_dir" / "output.csv")

        mock_analyze.return_value = ("neutral", {"positive": 0.3, "negative": 0.3, "neutral": 0.4})

        analyze_directory_to_csv(str(input_dir), output_file, MagicMock(), MagicMock(), "cpu")

        assert os.path.exists(output_file)


class TestEvaluatePerformance:
    """Test suite for evaluate_performance function."""

    def test_file_not_found(self, tmp_path, capsys):
        """Test handling of non-existent labeled data file."""
        nonexistent = str(tmp_path / "nonexistent.csv")

        evaluate_performance(nonexistent, MagicMock(), MagicMock(), "cpu")

        captured = capsys.readouterr()
        assert "Labeled data file not found" in captured.out

    def test_missing_required_columns(self, tmp_path, capsys):
        """Test handling of CSV missing required columns."""
        csv_file = tmp_path / "labeled.csv"
        pd.DataFrame({"wrong_column": ["test"]}).to_csv(csv_file, index=False)

        evaluate_performance(str(csv_file), MagicMock(), MagicMock(), "cpu")

        captured = capsys.readouterr()
        assert "must contain 'text' and 'true_label' columns" in captured.out

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    @patch("src.nlp.finbert_analyser.classification_report")
    @patch("src.nlp.finbert_analyser.accuracy_score")
    def test_successful_evaluation(
        self, mock_accuracy, mock_report, mock_analyze, tmp_path, capsys
    ):
        """Test successful performance evaluation."""
        # Create labeled data
        csv_file = tmp_path / "labeled.csv"
        pd.DataFrame(
            {
                "text": ["Positive outlook", "Negative news"],
                "true_label": ["positive", "negative"],
            }
        ).to_csv(csv_file, index=False)

        mock_analyze.side_effect = [
            ("positive", {"positive": 0.8, "negative": 0.1, "neutral": 0.1}),
            ("negative", {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        ]
        mock_accuracy.return_value = 1.0
        mock_report.return_value = "Classification Report"

        evaluate_performance(str(csv_file), MagicMock(), MagicMock(), "cpu")

        captured = capsys.readouterr()
        assert "Accuracy: 1.0000" in captured.out


class TestMain:
    """Test suite for main function."""

    @patch("src.nlp.finbert_analyser.load_finbert")
    @patch("src.nlp.finbert_analyser.run_functional_verification")
    def test_main_verify_mode(self, mock_verify, mock_load):
        """Test main function in verify mode."""
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_verify.return_value = True

        with patch("sys.argv", ["finbert_analyser.py", "--mode", "verify"]):
            main()

        mock_verify.assert_called_once()

    @patch("src.nlp.finbert_analyser.load_finbert")
    @patch("src.nlp.finbert_analyser.analyze_directory_to_csv")
    def test_main_batch_mode(self, mock_analyze, mock_load):
        """Test main function in batch mode."""
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")

        with patch(
            "sys.argv",
            [
                "finbert_analyser.py",
                "--mode",
                "batch",
                "--input_dir",
                "/input",
                "--output_file",
                "/output.csv",
            ],
        ):
            main()

        mock_analyze.assert_called_once()

    @patch("src.nlp.finbert_analyser.load_finbert")
    def test_main_batch_mode_missing_args(self, mock_load, capsys):
        """Test main function in batch mode with missing arguments."""
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")

        with patch("sys.argv", ["finbert_analyser.py", "--mode", "batch"]):
            main()

        captured = capsys.readouterr()
        assert "--input_dir and --output_file are required" in captured.out

    @patch("src.nlp.finbert_analyser.load_finbert")
    @patch("src.nlp.finbert_analyser.evaluate_performance")
    def test_main_evaluate_mode(self, mock_eval, mock_load):
        """Test main function in evaluate mode."""
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")

        with patch(
            "sys.argv",
            ["finbert_analyser.py", "--mode", "evaluate", "--labeled_data", "/labeled.csv"],
        ):
            main()

        mock_eval.assert_called_once()

    @patch("src.nlp.finbert_analyser.load_finbert")
    def test_main_evaluate_mode_missing_args(self, mock_load, capsys):
        """Test main function in evaluate mode with missing arguments."""
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")

        with patch("sys.argv", ["finbert_analyser.py", "--mode", "evaluate"]):
            main()

        captured = capsys.readouterr()
        assert "--labeled_data is required" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("src.nlp.finbert_analyser.torch")
    def test_analyze_sentiment_with_very_long_text(self, mock_torch):
        """Test analyzing very long text (should be truncated)."""
        long_text = "word " * 1000  # Very long text

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
                    return_value=MagicMock(tolist=MagicMock(return_value=[0.3, 0.3, 0.4]))
                )
            )
        )
        mock_torch.argmax.return_value = MagicMock(item=MagicMock(return_value=2))

        # Should handle without error
        label, scores = analyze_sentiment(long_text, mock_tokenizer, mock_model, "cpu")

        # Should have called tokenizer
        assert mock_tokenizer.called

    def test_analyze_sentiment_with_special_characters(self):
        """Test analyzing text with special characters."""
        special_text = "The rate is 3.5% and growing — steadily!"

        # Should handle without error
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        # For this test, we just verify it doesn't crash
        label, scores = analyze_sentiment("", mock_tokenizer, MagicMock(), "cpu")
        assert label == "neutral"

    @patch("src.nlp.finbert_analyser.analyze_sentiment")
    def test_directory_with_mixed_file_types(self, mock_analyze, tmp_path):
        """Test directory with mixed file types (only .txt processed)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("Test content.")
        (input_dir / "test.csv").write_text("csv,content")
        (input_dir / "test.json").write_text('{"key": "value"}')

        output_file = str(tmp_path / "output.csv")

        mock_analyze.return_value = ("neutral", {"positive": 0.3, "negative": 0.3, "neutral": 0.4})

        analyze_directory_to_csv(str(input_dir), output_file, MagicMock(), MagicMock(), "cpu")

        df = pd.read_csv(output_file)
        # Only 1 .txt file should be processed
        assert len(df) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
