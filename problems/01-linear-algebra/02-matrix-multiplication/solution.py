"""
Problem: Matrix Multiplication
Category: Linear Algebra
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Triple Nested Loop
# Time Complexity: O(m * n * p)
# Space Complexity: O(m * p)
# ============================================================

def matmul_triple_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication using three nested loops.

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (n, p)

    Returns:
        Product matrix of shape (m, p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


# ============================================================
# Approach 2: Row-Column Dot Products (Vectorized Inner Loop)
# Time Complexity: O(m * n * p)
# Space Complexity: O(m * p)
# ============================================================

def matmul_row_col(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication by computing dot products of rows and columns
    using numpy element-wise multiply and sum (vectorizes the inner loop).

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (n, p)

    Returns:
        Product matrix of shape (m, p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        for j in range(p):
            C[i, j] = np.sum(A[i, :] * B[:, j])
    return C


# ============================================================
# Approach 3: Row-wise Vectorized
# Time Complexity: O(m * n * p)
# Space Complexity: O(m * p)
# ============================================================

def matmul_row_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication by vectorizing across columns of B for each row of A.

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (n, p)

    Returns:
        Product matrix of shape (m, p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    m = A.shape[0]
    p = B.shape[1]
    C = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        # A[i, :] is (n,), B is (n, p) -> broadcast multiply and sum over axis 0
        C[i, :] = np.sum(A[i, :].reshape(-1, 1) * B, axis=0)
    return C


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    A1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B1 = np.array([[5, 6], [7, 8]], dtype=np.float64)
    expected1 = np.array([[19, 22], [43, 50]], dtype=np.float64)

    for name, fn in [("Triple Loop", matmul_triple_loop),
                     ("Row-Col", matmul_row_col),
                     ("Row Vectorized", matmul_row_vectorized)]:
        result = fn(A1, B1)
        print(f"Test 1 ({name}): {'PASS' if np.allclose(result, expected1) else 'FAIL'}")

    # Test Example 2
    A2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    B2 = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
    expected2 = np.array([[58, 64], [139, 154]], dtype=np.float64)

    for name, fn in [("Triple Loop", matmul_triple_loop),
                     ("Row-Col", matmul_row_col),
                     ("Row Vectorized", matmul_row_vectorized)]:
        result = fn(A2, B2)
        print(f"Test 2 ({name}): {'PASS' if np.allclose(result, expected2) else 'FAIL'}")

    # Test: Incompatible shapes
    try:
        matmul_triple_loop(np.array([[1, 2]]), np.array([[1, 2]]))
        print("Test 3 (Bad Shape): FAIL - no exception")
    except ValueError:
        print("Test 3 (Bad Shape): PASS")

    # Test: Verify against numpy
    np.random.seed(42)
    A3 = np.random.randn(5, 4)
    B3 = np.random.randn(4, 3)
    expected3 = A3 @ B3
    result3 = matmul_triple_loop(A3, B3)
    print(f"Test 4 (Random 5x4 @ 4x3): {'PASS' if np.allclose(result3, expected3) else 'FAIL'}")
