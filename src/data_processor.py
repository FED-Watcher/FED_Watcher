"""Data processing utilities."""


class DataProcessor:
    """Simple data processor."""

    def __init__(self):
        """Initialize the processor."""
        self.data = []

    def add_item(self, item):
        """Add an item to the data."""
        self.data.append(item)

    def get_count(self):
        """Return the number of items."""
        return len(self.data)

    def clear(self):
        """Clear all data."""
        self.data = []
