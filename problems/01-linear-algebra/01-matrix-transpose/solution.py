"""
Problem: Matrix Transpose
Category: Linear Algebra
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Index Swapping
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def transpose_index_swap(matrix: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix by swapping row and column indices.

    Args:
        matrix: 2D numpy array of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    m, n = matrix.shape
    result = np.zeros((n, m), dtype=matrix.dtype)
    for i in range(m):
        for j in range(n):
            result[j, i] = matrix[i, j]
    return result


# ============================================================
# Approach 2: Row-by-Row Construction
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def transpose_row_construction(matrix: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix by constructing each row of the result
    from the corresponding column of the input.

    Args:
        matrix: 2D numpy array of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    m, n = matrix.shape
    return np.array([matrix[:, j] for j in range(n)])


# ============================================================
# Approach 3: Vectorized with Fancy Indexing
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def transpose_fancy_indexing(matrix: np.ndarray) -> np.ndarray:
    """
    Transpose using NumPy advanced indexing to build the index grid.

    Args:
        matrix: 2D numpy array of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    m, n = matrix.shape
    rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    result = np.empty((n, m), dtype=matrix.dtype)
    result[cols.ravel(), rows.ravel()] = matrix.ravel()
    return result


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    input1 = np.array([[1, 2, 3],
                        [4, 5, 6]])
    expected1 = np.array([[1, 4],
                           [2, 5],
                           [3, 6]])
    result1 = transpose_index_swap(input1)
    print(f"Test 1 (Index Swap):        {'PASS' if np.array_equal(result1, expected1) else 'FAIL'}")
    result1b = transpose_row_construction(input1)
    print(f"Test 1 (Row Construction):   {'PASS' if np.array_equal(result1b, expected1) else 'FAIL'}")
    result1c = transpose_fancy_indexing(input1)
    print(f"Test 1 (Fancy Indexing):     {'PASS' if np.array_equal(result1c, expected1) else 'FAIL'}")

    # Test Example 2
    input2 = np.array([[1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8]])
    expected2 = np.array([[1, 3, 5, 7],
                           [2, 4, 6, 8]])
    result2 = transpose_index_swap(input2)
    print(f"Test 2 (Index Swap):        {'PASS' if np.array_equal(result2, expected2) else 'FAIL'}")
    result2b = transpose_row_construction(input2)
    print(f"Test 2 (Row Construction):   {'PASS' if np.array_equal(result2b, expected2) else 'FAIL'}")
    result2c = transpose_fancy_indexing(input2)
    print(f"Test 2 (Fancy Indexing):     {'PASS' if np.array_equal(result2c, expected2) else 'FAIL'}")

    # Test: Square matrix
    input3 = np.array([[1, 2], [3, 4]])
    expected3 = np.array([[1, 3], [2, 4]])
    result3 = transpose_index_swap(input3)
    print(f"Test 3 (Square Matrix):     {'PASS' if np.array_equal(result3, expected3) else 'FAIL'}")

    # Test: Single row
    input4 = np.array([[1, 2, 3]])
    expected4 = np.array([[1], [2], [3]])
    result4 = transpose_index_swap(input4)
    print(f"Test 4 (Single Row):        {'PASS' if np.array_equal(result4, expected4) else 'FAIL'}")
