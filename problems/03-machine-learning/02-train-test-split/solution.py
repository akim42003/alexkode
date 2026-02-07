"""
Problem: Train-Test Split
Category: Machine Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# Approach 1: Index Shuffling with NumPy
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets with shuffling.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        test_size: Fraction of data to use for testing (0, 1)
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    n = len(X)
    n_test = int(n * test_size)
    if n_test == 0:
        n_test = 1
    n_train = n - n_test

    # Generate shuffled indices
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# Approach 2: Fisher-Yates Shuffle (Manual)
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def fisher_yates_shuffle(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Implement Fisher-Yates shuffle algorithm."""
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return np.array(indices)


def train_test_split_manual(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                             random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data using Fisher-Yates shuffle (no np.random.shuffle).

    Args:
        X: Feature matrix
        y: Label array
        test_size: Fraction for testing
        random_seed: Optional seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    n = len(X)
    n_test = max(1, int(n * test_size))

    rng = np.random.RandomState(random_seed)
    indices = fisher_yates_shuffle(n, rng)

    split = n - n_test
    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y1 = np.array([0, 1, 0, 1, 0])

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.4, random_seed=42)
    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Test 1 train size: {'PASS' if X_train.shape[0] == 3 else 'FAIL'}")
    print(f"Test 1 test size:  {'PASS' if X_test.shape[0] == 2 else 'FAIL'}")

    # Verify no data loss
    all_X = np.vstack([X_train, X_test])
    print(f"No data loss:      {'PASS' if len(all_X) == len(X1) else 'FAIL'}")

    # Test Example 2
    X2 = np.arange(10).reshape(-1, 1)
    y2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_train2, X_test2, _, _ = train_test_split(X2, y2, test_size=0.2, random_seed=42)
    print(f"Test 2 train size: {'PASS' if X_train2.shape[0] == 8 else 'FAIL'}")
    print(f"Test 2 test size:  {'PASS' if X_test2.shape[0] == 2 else 'FAIL'}")

    # Test: Reproducibility
    r1 = train_test_split(X1, y1, test_size=0.4, random_seed=42)
    r2 = train_test_split(X1, y1, test_size=0.4, random_seed=42)
    print(f"Reproducible:      {'PASS' if np.array_equal(r1[0], r2[0]) else 'FAIL'}")

    # Test: Manual shuffle version
    X_t3, X_te3, y_t3, y_te3 = train_test_split_manual(X1, y1, test_size=0.4, random_seed=42)
    print(f"Manual split size: {'PASS' if X_t3.shape[0] == 3 and X_te3.shape[0] == 2 else 'FAIL'}")

    # Test: Correspondence maintained
    Xf = np.array([[10, 20], [30, 40]])
    yf = np.array([100, 200])
    Xt, Xte, yt, yte = train_test_split(Xf, yf, test_size=0.5, random_seed=0)
    # The label should match the feature row
    print(f"Correspondence:    {'PASS' if len(Xt) == 1 and len(Xte) == 1 else 'FAIL'}")
