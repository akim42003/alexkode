"""
Problem: Solve Linear Systems (Jacobi Method)
Category: Linear Algebra
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Jacobi Iterative Method
# Time Complexity: O(n^2 * iterations)
# Space Complexity: O(n)
# ============================================================

def jacobi_method(A: np.ndarray, b: np.ndarray, max_iter: int = 1000,
                  tol: float = 1e-8) -> Tuple[np.ndarray, int]:
    """
    Solve Ax = b using the Jacobi iterative method.
    All variables are updated simultaneously using values from the previous iteration.

    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (solution vector x, number of iterations)
    """
    n = A.shape[0]
    A = A.astype(np.float64)
    b = b.astype(np.float64)

    # Check diagonal dominance (warning only)
    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Zero diagonal element at index {i}")

    x = np.zeros(n, dtype=np.float64)

    for iteration in range(max_iter):
        x_new = np.zeros(n, dtype=np.float64)

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            return x_new, iteration + 1

        x = x_new

    return x, max_iter


# ============================================================
# Approach 2: Gauss-Seidel Method
# Time Complexity: O(n^2 * iterations)
# Space Complexity: O(n)
# ============================================================

def gauss_seidel_method(A: np.ndarray, b: np.ndarray, max_iter: int = 1000,
                        tol: float = 1e-8) -> Tuple[np.ndarray, int]:
    """
    Solve Ax = b using the Gauss-Seidel iterative method.
    Uses updated values immediately within the same iteration (typically faster convergence).

    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (solution vector x, number of iterations)
    """
    n = A.shape[0]
    A = A.astype(np.float64)
    b = b.astype(np.float64)

    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Zero diagonal element at index {i}")

    x = np.zeros(n, dtype=np.float64)

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]  # Uses latest x values
            x[i] = (b[i] - sigma) / A[i, i]

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            return x, iteration + 1

    return x, max_iter


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    A1 = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=np.float64)
    b1 = np.array([5, 10, 5], dtype=np.float64)
    expected1 = np.linalg.solve(A1, b1)

    x_jacobi, iters_j = jacobi_method(A1, b1)
    print(f"Jacobi Test 1:       {'PASS' if np.allclose(x_jacobi, expected1, atol=1e-6) else 'FAIL'} (iters={iters_j})")

    x_gs, iters_gs = gauss_seidel_method(A1, b1)
    print(f"Gauss-Seidel Test 1: {'PASS' if np.allclose(x_gs, expected1, atol=1e-6) else 'FAIL'} (iters={iters_gs})")

    # Test Example 2
    A2 = np.array([[10, -1, 2], [-1, 11, -1], [2, -1, 10]], dtype=np.float64)
    b2 = np.array([6, 25, -11], dtype=np.float64)
    expected2 = np.array([1.0, 2.0, -1.0])

    x_jacobi2, iters_j2 = jacobi_method(A2, b2)
    print(f"Jacobi Test 2:       {'PASS' if np.allclose(x_jacobi2, expected2, atol=1e-6) else 'FAIL'} (iters={iters_j2})")

    x_gs2, iters_gs2 = gauss_seidel_method(A2, b2)
    print(f"Gauss-Seidel Test 2: {'PASS' if np.allclose(x_gs2, expected2, atol=1e-6) else 'FAIL'} (iters={iters_gs2})")

    # Verify Ax = b
    residual = np.linalg.norm(A2 @ x_jacobi2 - b2)
    print(f"Residual check:      {'PASS' if residual < 1e-6 else 'FAIL'} (residual={residual:.2e})")

    # Gauss-Seidel typically converges faster
    print(f"GS faster than Jacobi: {iters_gs2 < iters_j2} ({iters_gs2} vs {iters_j2} iterations)")
