"""
Problem: Z-Score Normalization
Category: Statistics & Probability
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
import math


# ============================================================
# Approach 1: Manual Loop
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def zscore_loop(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalization using explicit loops.

    Args:
        data: 1D array of values

    Returns:
        Z-score normalized array
    """
    n = len(data)

    # Compute mean
    total = 0.0
    for x in data:
        total += x
    mu = total / n

    # Compute population standard deviation
    sum_sq = 0.0
    for x in data:
        sum_sq += (x - mu) ** 2
    sigma = math.sqrt(sum_sq / n)

    # Normalize
    if sigma < 1e-12:
        return np.zeros(n, dtype=np.float64)

    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = (data[i] - mu) / sigma
    return result


# ============================================================
# Approach 2: Vectorized with NumPy
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def zscore_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalization using numpy operations (no np.mean/np.std).

    Args:
        data: 1D array of values

    Returns:
        Z-score normalized array
    """
    data = data.astype(np.float64)
    n = len(data)

    mu = np.sum(data) / n
    sigma = np.sqrt(np.sum((data - mu) ** 2) / n)

    if sigma < 1e-12:
        return np.zeros(n, dtype=np.float64)

    return (data - mu) / sigma


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    data1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    expected1 = np.array([-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356])

    for name, fn in [("Loop", zscore_loop), ("Vectorized", zscore_vectorized)]:
        result = fn(data1)
        print(f"Test 1 ({name}):     {'PASS' if np.allclose(result, expected1) else 'FAIL'}")

    # Verify mean ≈ 0 and std ≈ 1
    z1 = zscore_vectorized(data1)
    print(f"Mean ≈ 0:            {'PASS' if abs(np.sum(z1) / len(z1)) < 1e-10 else 'FAIL'}")
    print(f"Std ≈ 1:             {'PASS' if abs(np.sqrt(np.sum(z1**2) / len(z1)) - 1.0) < 1e-10 else 'FAIL'}")

    # Test Example 2: All identical
    data2 = np.array([10, 10, 10], dtype=np.float64)
    expected2 = np.array([0.0, 0.0, 0.0])

    for name, fn in [("Loop", zscore_loop), ("Vectorized", zscore_vectorized)]:
        result = fn(data2)
        print(f"Test 2 ({name}):     {'PASS' if np.allclose(result, expected2) else 'FAIL'}")

    # Test: Negative values
    data3 = np.array([-5, 0, 5], dtype=np.float64)
    result3 = zscore_vectorized(data3)
    print(f"Negative values:     {'PASS' if abs(result3[1]) < 1e-10 else 'FAIL'} (middle should be 0)")
    print(f"Symmetric:           {'PASS' if abs(result3[0] + result3[2]) < 1e-10 else 'FAIL'}")
