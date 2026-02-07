"""
Problem: Feature Scaling (Min-Max, Standard)
Category: Machine Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Min-Max Scaling
# Time Complexity: O(n * d)
# Space Complexity: O(n * d)
# ============================================================

def minmax_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features to [0, 1] range using min-max normalization.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Tuple of (X_scaled, mins, maxs) for inverse transform
    """
    X = X.astype(np.float64)
    n, d = X.shape

    mins = np.zeros(d)
    maxs = np.zeros(d)

    for j in range(d):
        mins[j] = X[0, j]
        maxs[j] = X[0, j]
        for i in range(1, n):
            if X[i, j] < mins[j]:
                mins[j] = X[i, j]
            if X[i, j] > maxs[j]:
                maxs[j] = X[i, j]

    X_scaled = np.zeros_like(X, dtype=np.float64)
    for j in range(d):
        range_j = maxs[j] - mins[j]
        if range_j < 1e-12:
            X_scaled[:, j] = 0.0
        else:
            for i in range(n):
                X_scaled[i, j] = (X[i, j] - mins[j]) / range_j

    return X_scaled, mins, maxs


# ============================================================
# Approach 2: Standard Scaling (Z-score)
# Time Complexity: O(n * d)
# Space Complexity: O(n * d)
# ============================================================

def standard_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features to zero mean and unit variance.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Tuple of (X_scaled, means, stds) for inverse transform
    """
    X = X.astype(np.float64)
    n, d = X.shape

    means = np.zeros(d)
    stds = np.zeros(d)

    # Compute means
    for j in range(d):
        total = 0.0
        for i in range(n):
            total += X[i, j]
        means[j] = total / n

    # Compute standard deviations (population)
    for j in range(d):
        sum_sq = 0.0
        for i in range(n):
            sum_sq += (X[i, j] - means[j]) ** 2
        stds[j] = np.sqrt(sum_sq / n)

    # Scale
    X_scaled = np.zeros_like(X, dtype=np.float64)
    for j in range(d):
        if stds[j] < 1e-12:
            X_scaled[:, j] = 0.0
        else:
            for i in range(n):
                X_scaled[i, j] = (X[i, j] - means[j]) / stds[j]

    return X_scaled, means, stds


# ============================================================
# Approach 3: Fit-Transform Pattern (Reusable)
# Time Complexity: O(n * d)
# Space Complexity: O(n * d)
# ============================================================

class MinMaxScaler:
    """Min-Max scaler with fit/transform interface."""

    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        X = X.astype(np.float64)
        self.mins = np.min(X, axis=0)
        self.maxs = np.max(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        ranges = self.maxs - self.mins
        ranges[ranges < 1e-12] = 1.0  # avoid division by zero
        return (X - self.mins) / ranges

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScaler:
    """Standard scaler with fit/transform interface."""

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        X = X.astype(np.float64)
        self.means = np.sum(X, axis=0) / X.shape[0]
        self.stds = np.sqrt(np.sum((X - self.means) ** 2, axis=0) / X.shape[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        stds = self.stds.copy()
        stds[stds < 1e-12] = 1.0
        return (X - self.means) / stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)

    # Test Min-Max
    X_mm, _, _ = minmax_scale(X)
    expected_mm = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    print(f"Min-Max Scaling:      {'PASS' if np.allclose(X_mm, expected_mm) else 'FAIL'}")

    # Test Standard Scaling
    X_std, means, stds = standard_scale(X)
    expected_std = np.array([[-1.22474487, -1.22474487],
                              [0.0, 0.0],
                              [1.22474487, 1.22474487]])
    print(f"Standard Scaling:     {'PASS' if np.allclose(X_std, expected_std, atol=1e-6) else 'FAIL'}")

    # Verify zero mean
    col_means = np.sum(X_std, axis=0) / X_std.shape[0]
    print(f"Zero mean:            {'PASS' if np.allclose(col_means, 0, atol=1e-10) else 'FAIL'}")

    # Test class-based
    scaler = MinMaxScaler()
    X_mm2 = scaler.fit_transform(X)
    print(f"MinMaxScaler class:   {'PASS' if np.allclose(X_mm2, expected_mm) else 'FAIL'}")

    std_scaler = StandardScaler()
    X_std2 = std_scaler.fit_transform(X)
    print(f"StandardScaler class: {'PASS' if np.allclose(X_std2, expected_std, atol=1e-6) else 'FAIL'}")

    # Test: Constant feature
    X_const = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
    X_const_mm, _, _ = minmax_scale(X_const)
    print(f"Constant feature MM:  {'PASS' if np.allclose(X_const_mm[:, 0], 0) else 'FAIL'}")
