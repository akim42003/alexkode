"""
Problem: Eigenvalues and Eigenvectors
Category: Linear Algebra
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Power Iteration
# Time Complexity: O(n^2 * iterations)
# Space Complexity: O(n)
# ============================================================

def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    """
    Find the dominant eigenvalue and eigenvector using power iteration.

    Args:
        A: Square numpy array of shape (n, n)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (dominant_eigenvalue, corresponding_eigenvector)
    """
    n = A.shape[0]
    # Start with a random vector
    np.random.seed(0)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue = 0.0
    for _ in range(max_iter):
        # Multiply by A
        Av = A @ v
        # Compute eigenvalue estimate (Rayleigh quotient)
        new_eigenvalue = v @ Av
        # Normalize
        v_new = Av / np.linalg.norm(Av)

        # Check convergence
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v_new

        eigenvalue = new_eigenvalue
        v = v_new

    return eigenvalue, v


# ============================================================
# Approach 2: QR Algorithm (for all eigenvalues)
# Time Complexity: O(n^3 * iterations)
# Space Complexity: O(n^2)
# ============================================================

def gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QR decomposition using modified Gram-Schmidt process.

    Args:
        A: Square numpy array of shape (n, n)

    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        v = A[:, j].astype(np.float64)
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-12:
            Q[:, j] = 0
        else:
            Q[:, j] = v / R[j, j]

    return Q, R


def qr_algorithm(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """
    Find all eigenvalues using the QR algorithm.
    Repeatedly decomposes A = QR, then forms A = RQ.

    Args:
        A: Square numpy array of shape (n, n)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Array of eigenvalues (diagonal of converged matrix)
    """
    Ak = A.astype(np.float64).copy()
    n = Ak.shape[0]

    for _ in range(max_iter):
        Q, R = gram_schmidt_qr(Ak)
        Ak = R @ Q

        # Check if off-diagonal elements are small enough
        off_diag = np.sum(np.abs(Ak) - np.abs(np.diag(np.diag(Ak))))
        if off_diag < tol * n:
            break

    return np.diag(Ak)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Power Iteration
    A1 = np.array([[2, 1], [1, 3]], dtype=np.float64)
    eigenvalue, eigenvector = power_iteration(A1)
    # Dominant eigenvalue should be (5 + sqrt(5)) / 2 ≈ 3.618
    expected_eigenvalue = (5 + np.sqrt(5)) / 2
    print(f"Power Iteration eigenvalue:  {'PASS' if abs(eigenvalue - expected_eigenvalue) < 1e-6 else 'FAIL'} ({eigenvalue:.4f} ≈ {expected_eigenvalue:.4f})")

    # Verify Av = λv
    residual = np.linalg.norm(A1 @ eigenvector - eigenvalue * eigenvector)
    print(f"Power Iteration Av=λv:       {'PASS' if residual < 1e-6 else 'FAIL'} (residual={residual:.2e})")

    # Test Example 2: QR Algorithm
    A2 = np.array([[4, 1], [2, 3]], dtype=np.float64)
    eigenvalues = qr_algorithm(A2)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    expected_sorted = np.array([5.0, 2.0])
    print(f"QR Algorithm eigenvalues:    {'PASS' if np.allclose(eigenvalues_sorted, expected_sorted, atol=1e-4) else 'FAIL'} ({eigenvalues_sorted})")

    # Test: Symmetric 3x3
    A3 = np.array([[6, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=np.float64)
    eigenvalues3 = qr_algorithm(A3)
    expected3 = np.sort(np.linalg.eigvals(A3))[::-1]
    eigenvalues3_sorted = np.sort(eigenvalues3)[::-1]
    print(f"QR 3x3 symmetric:           {'PASS' if np.allclose(eigenvalues3_sorted, expected3, atol=1e-4) else 'FAIL'}")

    # Test: Compare with numpy
    np.random.seed(42)
    A4 = np.random.randn(4, 4)
    A4 = A4 + A4.T  # make symmetric for reliable QR convergence
    eigenvalues4 = np.sort(qr_algorithm(A4))[::-1]
    expected4 = np.sort(np.linalg.eigvals(A4))[::-1]
    print(f"QR 4x4 vs numpy:            {'PASS' if np.allclose(eigenvalues4, expected4, atol=1e-3) else 'FAIL'}")
