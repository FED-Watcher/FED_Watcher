"""Pytest configuration for the test suite."""

import matplotlib

# Use non-interactive backend for tests to avoid Tkinter issues
matplotlib.use("Agg")
