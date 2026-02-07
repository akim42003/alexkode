"""
Problem: Reshape and Flatten
Category: Linear Algebra
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Index Mapping
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def reshape_index_mapping(matrix: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reshape a matrix by mapping linear indices to new row/col positions.

    Args:
        matrix: 2D numpy array of shape (m, n)
        target_shape: tuple (p, q) where m*n == p*q

    Returns:
        Reshaped matrix of shape (p, q)
    """
    m, n = matrix.shape
    p, q = target_shape

    if m * n != p * q:
        raise ValueError(f"Cannot reshape ({m}, {n}) into ({p}, {q}): {m*n} != {p*q}")

    result = np.zeros((p, q), dtype=matrix.dtype)
    for k in range(m * n):
        # Source position in original matrix
        src_i, src_j = k // n, k % n
        # Target position in new matrix
        dst_i, dst_j = k // q, k % q
        result[dst_i, dst_j] = matrix[src_i, src_j]
    return result


def flatten_index(matrix: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D matrix into a 1D array using index mapping.

    Args:
        matrix: 2D numpy array of shape (m, n)

    Returns:
        1D numpy array of length m*n
    """
    m, n = matrix.shape
    result = np.zeros(m * n, dtype=matrix.dtype)
    for i in range(m):
        for j in range(n):
            result[i * n + j] = matrix[i, j]
    return result


# ============================================================
# Approach 2: Iterative Collection
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def reshape_collect(matrix: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reshape by collecting all elements into a list, then filling new shape.

    Args:
        matrix: 2D numpy array of shape (m, n)
        target_shape: tuple (p, q)

    Returns:
        Reshaped matrix of shape (p, q)
    """
    m, n = matrix.shape
    p, q = target_shape

    if m * n != p * q:
        raise ValueError(f"Cannot reshape ({m}, {n}) into ({p}, {q}): {m*n} != {p*q}")

    # Collect all elements in row-major order
    elements = []
    for i in range(m):
        for j in range(n):
            elements.append(matrix[i, j])

    # Fill new matrix
    result = np.zeros((p, q), dtype=matrix.dtype)
    idx = 0
    for i in range(p):
        for j in range(q):
            result[i, j] = elements[idx]
            idx += 1
    return result


def flatten_collect(matrix: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D matrix by collecting elements row by row.

    Args:
        matrix: 2D numpy array of shape (m, n)

    Returns:
        1D numpy array of length m*n
    """
    m, n = matrix.shape
    elements = []
    for i in range(m):
        for j in range(n):
            elements.append(matrix[i, j])
    return np.array(elements, dtype=matrix.dtype)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Reshape Example 1
    input1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    result1a = reshape_index_mapping(input1, (4, 2))
    print(f"Reshape Test 1 (Index Mapping):  {'PASS' if np.array_equal(result1a, expected1) else 'FAIL'}")
    result1b = reshape_collect(input1, (4, 2))
    print(f"Reshape Test 1 (Collect):        {'PASS' if np.array_equal(result1b, expected1) else 'FAIL'}")

    # Test Flatten Example 2
    input2 = np.array([[1, 2, 3], [4, 5, 6]])
    expected2 = np.array([1, 2, 3, 4, 5, 6])

    result2a = flatten_index(input2)
    print(f"Flatten Test 2 (Index):          {'PASS' if np.array_equal(result2a, expected2) else 'FAIL'}")
    result2b = flatten_collect(input2)
    print(f"Flatten Test 2 (Collect):        {'PASS' if np.array_equal(result2b, expected2) else 'FAIL'}")

    # Test: Reshape to same shape
    input3 = np.array([[1, 2], [3, 4]])
    result3 = reshape_index_mapping(input3, (2, 2))
    print(f"Reshape Same Shape:              {'PASS' if np.array_equal(result3, input3) else 'FAIL'}")

    # Test: Invalid reshape
    try:
        reshape_index_mapping(input3, (3, 3))
        print("Invalid Reshape:                 FAIL - no exception")
    except ValueError:
        print("Invalid Reshape:                 PASS")

    # Test: Reshape to single row
    result4 = reshape_index_mapping(input3, (1, 4))
    expected4 = np.array([[1, 2, 3, 4]])
    print(f"Reshape to Row:                  {'PASS' if np.array_equal(result4, expected4) else 'FAIL'}")
