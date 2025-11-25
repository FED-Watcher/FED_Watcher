"""Unit tests for FED-Watcher source code."""
import pytest
from src.utils import hello, add_numbers, validate_data
from src.data_processor import DataProcessor


def test_hello():
    """Test hello function returns correct greeting."""
    result = hello()
    assert result == "Hello from FED-Watcher!"
    assert isinstance(result, str)


def test_add_numbers():
    """Test add_numbers function with various inputs."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(10, -5) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, -1) == -2


def test_validate_data_with_valid_input():
    """Test validate_data with valid inputs."""
    assert validate_data([1, 2, 3]) is True
    assert validate_data("data") is True
    assert validate_data({"key": "value"}) is True


def test_validate_data_with_invalid_input():
    """Test validate_data raises ValueError for empty input."""
    with pytest.raises(ValueError, match="Data cannot be empty"):
        validate_data(None)
    
    with pytest.raises(ValueError, match="Data cannot be empty"):
        validate_data("")
    
    with pytest.raises(ValueError, match="Data cannot be empty"):
        validate_data([])


def test_data_processor_initialization():
    """Test DataProcessor initializes correctly."""
    processor = DataProcessor()
    assert processor.get_count() == 0
    assert processor.data == []


def test_data_processor_add_item():
    """Test adding items to DataProcessor."""
    processor = DataProcessor()
    
    processor.add_item("item1")
    assert processor.get_count() == 1
    assert "item1" in processor.data
    
    processor.add_item("item2")
    assert processor.get_count() == 2
    assert "item2" in processor.data


def test_data_processor_clear():
    """Test clearing DataProcessor."""
    processor = DataProcessor()
    processor.add_item("test1")
    processor.add_item("test2")
    assert processor.get_count() == 2
    
    processor.clear()
    assert processor.get_count() == 0
    assert processor.data == []


@pytest.mark.critical
def test_critical_path():
    """Test critical functionality works end-to-end."""
    # Test utils
    greeting = hello()
    assert "FED-Watcher" in greeting
    
    # Test math
    result = add_numbers(5, 10)
    assert result == 15
    
    # Test data processor
    processor = DataProcessor()
    processor.add_item("test")
    assert len(processor.data) == 1
    
    # Test validation
    assert validate_data(processor.data) is True


def test_imports():
    """Test that all required modules can be imported."""
    # This test is now actually using the imports
    assert callable(hello)
    assert callable(add_numbers)
    assert callable(validate_data)
    assert DataProcessor is not None


def test_basic_functionality():
    """Test basic functionality."""
    processor = DataProcessor()
    processor.add_item(1)
    processor.add_item(2)
    processor.add_item(3)
    
    count = processor.get_count()
    assert count == 3
    assert sum(processor.data) == 6
