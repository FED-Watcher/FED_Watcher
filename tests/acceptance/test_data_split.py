"""Acceptance test for data scientist requirement."""

import pytest
from sklearn.model_selection import train_test_split
import numpy as np


def test_user_story_80_20_split():
    """
    USER STORY:
    As a data scientist, when I have a dataset, I want to split
    the data into training and test sets so that I can evaluate
    my model's performance.

    ACCEPTANCE CRITERIA:
    ✅ Dataset must be split into 80% training and 20% test
    ✅ Split should be random to ensure fair representation
    """
    # Given: A dataset
    X = np.random.randn(100, 10)
    y = np.random.choice(["up", "down", "stable"], 100)

    # When: I split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Then: Data is split 80/20
    total = len(X)
    train_ratio = len(X_train) / total
    test_ratio = len(X_test) / total

    assert abs(train_ratio - 0.8) < 0.01, "Train ratio not 80%"
    assert abs(test_ratio - 0.2) < 0.01, "Test ratio not 20%"

    print("✅ User story acceptance criteria met!")
