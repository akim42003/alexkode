"""
Problem: Scalar and Element-wise Operations
Category: Linear Algebra
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Loop-based Implementation
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def scalar_multiply(matrix: np.ndarray, scalar: float) -> np.ndarray:
    """
    Multiply every element of a matrix by a scalar.

    Args:
        matrix: 2D numpy array of shape (m, n)
        scalar: numeric scalar value

    Returns:
        Scaled matrix of shape (m, n)
    """
    m, n = matrix.shape
    result = np.zeros_like(matrix, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result[i, j] = matrix[i, j] * scalar
    return result


def elementwise_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Add two matrices element by element.

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (m, n)

    Returns:
        Sum matrix of shape (m, n)
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    m, n = A.shape
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result[i, j] = A[i, j] + B[i, j]
    return result


def elementwise_subtract(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Subtract matrix B from matrix A element by element.

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (m, n)

    Returns:
        Difference matrix of shape (m, n)
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    m, n = A.shape
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result[i, j] = A[i, j] - B[i, j]
    return result


def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the Hadamard (element-wise) product of two matrices.

    Args:
        A: 2D numpy array of shape (m, n)
        B: 2D numpy array of shape (m, n)

    Returns:
        Hadamard product matrix of shape (m, n)
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    m, n = A.shape
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result[i, j] = A[i, j] * B[i, j]
    return result


# ============================================================
# Approach 2: Flat Iteration
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# ============================================================

def scalar_multiply_flat(matrix: np.ndarray, scalar: float) -> np.ndarray:
    """Scalar multiplication using flat iteration."""
    flat = matrix.flatten()
    result = np.array([x * scalar for x in flat], dtype=np.float64)
    return result.reshape(matrix.shape)


def hadamard_product_flat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Hadamard product using flat iteration."""
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    flat_a = A.flatten()
    flat_b = B.flatten()
    result = np.array([a * b for a, b in zip(flat_a, flat_b)], dtype=np.float64)
    return result.reshape(A.shape)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[5, 6], [7, 8]], dtype=np.float64)

    # Test Scalar Multiplication
    expected_scalar = np.array([[3, 6], [9, 12]], dtype=np.float64)
    result = scalar_multiply(A, 3)
    print(f"Scalar Multiply (Loop):  {'PASS' if np.allclose(result, expected_scalar) else 'FAIL'}")
    result_flat = scalar_multiply_flat(A, 3)
    print(f"Scalar Multiply (Flat):  {'PASS' if np.allclose(result_flat, expected_scalar) else 'FAIL'}")

    # Test Element-wise Addition
    expected_add = np.array([[6, 8], [10, 12]], dtype=np.float64)
    result = elementwise_add(A, B)
    print(f"Element-wise Add:        {'PASS' if np.allclose(result, expected_add) else 'FAIL'}")

    # Test Element-wise Subtraction
    expected_sub = np.array([[-4, -4], [-4, -4]], dtype=np.float64)
    result = elementwise_subtract(A, B)
    print(f"Element-wise Subtract:   {'PASS' if np.allclose(result, expected_sub) else 'FAIL'}")

    # Test Hadamard Product
    expected_hadamard = np.array([[5, 12], [21, 32]], dtype=np.float64)
    result = hadamard_product(A, B)
    print(f"Hadamard Product (Loop): {'PASS' if np.allclose(result, expected_hadamard) else 'FAIL'}")
    result_flat = hadamard_product_flat(A, B)
    print(f"Hadamard Product (Flat): {'PASS' if np.allclose(result_flat, expected_hadamard) else 'FAIL'}")

    # Test Shape mismatch
    try:
        elementwise_add(np.array([[1]]), np.array([[1, 2]]))
        print("Shape Mismatch Test:     FAIL - no exception")
    except ValueError:
        print("Shape Mismatch Test:     PASS")
