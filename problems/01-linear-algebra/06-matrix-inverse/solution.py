"""
Problem: Matrix Inverse (2x2 and NxN)
Category: Linear Algebra
Difficulty: Medium

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: 2x2 Closed-Form Formula
# Time Complexity: O(1)
# Space Complexity: O(1)
# ============================================================

def inverse_2x2(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 2x2 matrix using the closed-form formula.

    Args:
        matrix: 2D numpy array of shape (2, 2)

    Returns:
        Inverse matrix of shape (2, 2)
    """
    if matrix.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")

    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular (determinant is zero)")

    return (1.0 / det) * np.array([[d, -b], [-c, a]], dtype=np.float64)


# ============================================================
# Approach 2: Gauss-Jordan Elimination (NxN)
# Time Complexity: O(n^3)
# Space Complexity: O(n^2)
# ============================================================

def inverse_gauss_jordan(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of an NxN matrix using Gauss-Jordan elimination.
    Augments the matrix with the identity and row-reduces.

    Args:
        matrix: 2D numpy array of shape (n, n)

    Returns:
        Inverse matrix of shape (n, n)
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Augment with identity: [A | I]
    augmented = np.hstack([matrix.astype(np.float64), np.eye(n)])

    for col in range(n):
        # Partial pivoting: find the row with the largest absolute value in this column
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row, col]) > abs(augmented[max_row, col]):
                max_row = row

        # Swap rows
        if max_row != col:
            augmented[[col, max_row]] = augmented[[max_row, col]]

        pivot = augmented[col, col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular (zero pivot encountered)")

        # Scale pivot row
        augmented[col] = augmented[col] / pivot

        # Eliminate all other rows in this column
        for row in range(n):
            if row != col:
                factor = augmented[row, col]
                augmented[row] -= factor * augmented[col]

    # Extract the inverse from the right half
    return augmented[:, n:]


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: 2x2
    A1 = np.array([[4, 7], [2, 6]], dtype=np.float64)
    expected1 = np.array([[0.6, -0.7], [-0.2, 0.4]])

    result1a = inverse_2x2(A1)
    print(f"2x2 Formula:       {'PASS' if np.allclose(result1a, expected1) else 'FAIL'}")
    result1b = inverse_gauss_jordan(A1)
    print(f"2x2 Gauss-Jordan:  {'PASS' if np.allclose(result1b, expected1) else 'FAIL'}")

    # Test Example 2: 3x3
    A2 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64)
    expected2 = np.array([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]], dtype=np.float64)

    result2 = inverse_gauss_jordan(A2)
    print(f"3x3 Gauss-Jordan:  {'PASS' if np.allclose(result2, expected2) else 'FAIL'}")

    # Verify A @ A_inv = I
    identity_check = A2 @ result2
    print(f"A @ A_inv = I:     {'PASS' if np.allclose(identity_check, np.eye(3)) else 'FAIL'}")

    # Test: Singular matrix
    singular = np.array([[1, 2], [2, 4]], dtype=np.float64)
    try:
        inverse_2x2(singular)
        print("Singular 2x2:      FAIL - no exception")
    except ValueError:
        print("Singular 2x2:      PASS")

    try:
        inverse_gauss_jordan(singular)
        print("Singular GJ:       FAIL - no exception")
    except ValueError:
        print("Singular GJ:       PASS")

    # Test: Compare with numpy
    np.random.seed(42)
    A3 = np.random.randn(4, 4)
    result3 = inverse_gauss_jordan(A3)
    expected3 = np.linalg.inv(A3)
    print(f"4x4 vs np.linalg:  {'PASS' if np.allclose(result3, expected3) else 'FAIL'}")
