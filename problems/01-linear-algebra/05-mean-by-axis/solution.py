"""
Problem: Calculate Mean by Row/Column
Category: Linear Algebra
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Optional, Union


# ============================================================
# Approach 1: Loop and Sum
# Time Complexity: O(m * n)
# Space Complexity: O(m) or O(n) depending on axis
# ============================================================

def mean_by_axis_loop(matrix: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    """
    Compute the mean along a given axis using explicit loops.

    Args:
        matrix: 2D numpy array of shape (m, n)
        axis: 0 (column means), 1 (row means), or None (global mean)

    Returns:
        1D array of means or a scalar for axis=None
    """
    m, n = matrix.shape

    if axis is None:
        total = 0.0
        for i in range(m):
            for j in range(n):
                total += matrix[i, j]
        return total / (m * n)

    elif axis == 0:
        # Mean of each column: result has shape (n,)
        result = np.zeros(n, dtype=np.float64)
        for j in range(n):
            col_sum = 0.0
            for i in range(m):
                col_sum += matrix[i, j]
            result[j] = col_sum / m
        return result

    elif axis == 1:
        # Mean of each row: result has shape (m,)
        result = np.zeros(m, dtype=np.float64)
        for i in range(m):
            row_sum = 0.0
            for j in range(n):
                row_sum += matrix[i, j]
            result[i] = row_sum / n
        return result

    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or None.")


# ============================================================
# Approach 2: Using np.sum (no np.mean)
# Time Complexity: O(m * n)
# Space Complexity: O(m) or O(n) depending on axis
# ============================================================

def mean_by_axis_sum(matrix: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    """
    Compute the mean along a given axis using np.sum for reduction.

    Args:
        matrix: 2D numpy array of shape (m, n)
        axis: 0 (column means), 1 (row means), or None (global mean)

    Returns:
        1D array of means or a scalar for axis=None
    """
    m, n = matrix.shape

    if axis is None:
        return np.sum(matrix) / (m * n)
    elif axis == 0:
        return np.sum(matrix, axis=0).astype(np.float64) / m
    elif axis == 1:
        return np.sum(matrix, axis=1).astype(np.float64) / n
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or None.")


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    matrix = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=np.float64)

    # Test axis=0 (column means)
    expected_ax0 = np.array([2.5, 3.5, 4.5])
    for name, fn in [("Loop", mean_by_axis_loop), ("Sum", mean_by_axis_sum)]:
        result = fn(matrix, axis=0)
        print(f"Axis=0 ({name}): {'PASS' if np.allclose(result, expected_ax0) else 'FAIL'}")

    # Test axis=1 (row means)
    expected_ax1 = np.array([2.0, 5.0])
    for name, fn in [("Loop", mean_by_axis_loop), ("Sum", mean_by_axis_sum)]:
        result = fn(matrix, axis=1)
        print(f"Axis=1 ({name}): {'PASS' if np.allclose(result, expected_ax1) else 'FAIL'}")

    # Test axis=None (global mean)
    expected_none = 3.5
    for name, fn in [("Loop", mean_by_axis_loop), ("Sum", mean_by_axis_sum)]:
        result = fn(matrix, axis=None)
        print(f"Axis=None ({name}): {'PASS' if np.isclose(result, expected_none) else 'FAIL'}")

    # Test single row
    single = np.array([[10, 20, 30]], dtype=np.float64)
    result = mean_by_axis_loop(single, axis=1)
    print(f"Single Row Mean: {'PASS' if np.allclose(result, [20.0]) else 'FAIL'}")

    # Test single column
    single_col = np.array([[10], [20], [30]], dtype=np.float64)
    result = mean_by_axis_loop(single_col, axis=0)
    print(f"Single Col Mean: {'PASS' if np.allclose(result, [20.0]) else 'FAIL'}")
