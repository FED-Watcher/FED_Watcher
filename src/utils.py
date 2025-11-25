"""Utility functions for FED-Watcher."""


def hello():
    """Return a greeting message."""
    return "Hello from FED-Watcher!"


def add_numbers(a, b):
    """Add two numbers together."""
    return a + b


def validate_data(data):
    """Validate that data is not empty."""
    if not data:
        raise ValueError("Data cannot be empty")
    return True
