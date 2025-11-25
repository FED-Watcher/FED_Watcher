"""Cross-validation tests - 80/20 split requirement."""

import pytest
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def test_train_test_split_ratio():
    """
    ACCEPTANCE CRITERIA:
    ✅ Dataset must be split into 80% training and 20% test
    """
    # Create sample dataset
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], 100)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verify 80/20 split
    assert len(X_train) == 80, f"Expected 80 training samples, got {len(X_train)}"
    assert len(X_test) == 20, f"Expected 20 test samples, got {len(X_test)}"

    # Verify no data loss
    assert len(X_train) + len(X_test) == 100

    print(f"✅ Train: {len(X_train)} (80%), Test: {len(X_test)} (20%)")


def test_train_test_split_random():
    """
    ACCEPTANCE CRITERIA:
    ✅ Split should be random to ensure fair representation
    """
    # Create ordered dataset
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)

    # Split with shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Verify not sequential (test set should not be last 20 samples)
    test_indices = X_test.flatten()
    is_sequential = all(test_indices == np.arange(80, 100))

    assert not is_sequential, "Split is sequential, not random"
    print("✅ Split is random")


def test_kfold_cross_validation():
    """Test K-fold cross-validation implementation."""
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], 100)

    # Initialize 5-fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Verify number of splits
    n_splits = kfold.get_n_splits(X)
    assert n_splits == 5

    # Verify each fold
    for train_idx, test_idx in kfold.split(X):
        assert len(test_idx) == 20  # 20% of 100
        assert len(train_idx) == 80  # 80% of 100

    print("✅ K-fold cross-validation working correctly")
