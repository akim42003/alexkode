"""
Problem: Singular Value Decomposition (SVD)
Category: Linear Algebra
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: SVD via Eigendecomposition of A^T A
# Time Complexity: O(n^3 + m*n^2)
# Space Complexity: O(m*n)
# ============================================================

def svd_eigen(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD of A using eigendecomposition of A^T A.

    Steps:
    1. Compute A^T A (symmetric positive semi-definite)
    2. Find eigenvalues and eigenvectors of A^T A
    3. Singular values = sqrt(eigenvalues)
    4. V = eigenvectors of A^T A
    5. U = A V Σ^(-1) for non-zero singular values

    Args:
        A: 2D numpy array of shape (m, n)

    Returns:
        Tuple of (U, sigma, Vt) where:
            U: (m, m) orthogonal matrix
            sigma: 1D array of singular values (descending)
            Vt: (n, n) orthogonal matrix
    """
    A = A.astype(np.float64)
    m, n = A.shape

    # Step 1: Compute A^T A
    AtA = A.T @ A  # (n, n)

    # Step 2: Eigendecomposition (using np.linalg.eigh for symmetric matrix)
    eigenvalues, V = np.linalg.eigh(AtA)

    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Step 3: Singular values
    # Clamp small negative eigenvalues to zero (numerical error)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sigma = np.sqrt(eigenvalues)

    # Step 4: Compute U
    # For non-zero singular values: u_i = (1/σ_i) * A @ v_i
    rank = np.sum(sigma > 1e-10)
    U = np.zeros((m, m), dtype=np.float64)

    for i in range(rank):
        U[:, i] = (A @ V[:, i]) / sigma[i]

    # Complete U to an orthonormal basis using Gram-Schmidt on remaining columns
    # Fill remaining columns with orthonormal vectors
    if rank < m:
        # Use QR on a random extension to fill the null space
        random_cols = np.random.randn(m, m - rank)
        for i in range(m - rank):
            v = random_cols[:, i]
            # Orthogonalize against existing columns
            for j in range(rank + i):
                v = v - (U[:, j] @ v) * U[:, j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                U[:, rank + i] = v / norm

    Vt = V.T

    return U, sigma, Vt


# ============================================================
# Approach 2: SVD via Eigendecomposition of Both Sides
# Time Complexity: O(max(m,n)^3)
# Space Complexity: O(m*n)
# ============================================================

def svd_both_sides(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD using eigendecomposition of both A^T A and A A^T.
    This avoids numerical issues when computing U from A V Σ^(-1).

    Args:
        A: 2D numpy array of shape (m, n)

    Returns:
        Tuple of (U, sigma, Vt)
    """
    A = A.astype(np.float64)
    m, n = A.shape

    # Eigendecompose A^T A for V and singular values
    AtA = A.T @ A
    eigenvalues_v, V = np.linalg.eigh(AtA)
    idx_v = np.argsort(eigenvalues_v)[::-1]
    eigenvalues_v = np.maximum(eigenvalues_v[idx_v], 0.0)
    V = V[:, idx_v]

    # Eigendecompose A A^T for U
    AAt = A @ A.T
    eigenvalues_u, U = np.linalg.eigh(AAt)
    idx_u = np.argsort(eigenvalues_u)[::-1]
    U = U[:, idx_u]

    sigma = np.sqrt(eigenvalues_v)

    # Fix sign ambiguity: ensure U and V are consistent
    # For each singular vector, check that U[:,i] * sigma[i] ≈ A @ V[:,i]
    rank = np.sum(sigma > 1e-10)
    for i in range(rank):
        test = A @ V[:, i]
        if np.dot(test, U[:, i]) < 0:
            U[:, i] = -U[:, i]

    return U, sigma, V.T


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Diagonal matrix
    A1 = np.array([[3, 0], [0, 2]], dtype=np.float64)
    U1, sigma1, Vt1 = svd_eigen(A1)
    print(f"SVD Test 1 singular values: {'PASS' if np.allclose(sigma1, [3, 2]) else 'FAIL'} ({sigma1})")
    # Reconstruction
    S1 = np.zeros_like(A1)
    np.fill_diagonal(S1, sigma1)
    reconstructed1 = U1 @ S1 @ Vt1
    print(f"SVD Test 1 reconstruction:  {'PASS' if np.allclose(reconstructed1, A1) else 'FAIL'}")

    # Test Example 2: Rectangular matrix
    A2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    U2, sigma2, Vt2 = svd_eigen(A2)
    expected_sigma2 = np.linalg.svd(A2, compute_uv=False)
    print(f"SVD Test 2 singular values: {'PASS' if np.allclose(sigma2[:2], expected_sigma2, atol=1e-6) else 'FAIL'}")
    # Reconstruction
    S2 = np.zeros((3, 2))
    S2[0, 0] = sigma2[0]
    S2[1, 1] = sigma2[1]
    reconstructed2 = U2 @ S2 @ Vt2
    print(f"SVD Test 2 reconstruction:  {'PASS' if np.allclose(reconstructed2, A2, atol=1e-6) else 'FAIL'}")

    # Test: Both sides approach
    U2b, sigma2b, Vt2b = svd_both_sides(A2)
    print(f"Both Sides singular values: {'PASS' if np.allclose(sigma2b[:2], expected_sigma2, atol=1e-6) else 'FAIL'}")

    # Test: Compare with numpy on random matrix
    np.random.seed(42)
    A3 = np.random.randn(4, 3)
    U3, sigma3, Vt3 = svd_eigen(A3)
    expected_sigma3 = np.linalg.svd(A3, compute_uv=False)
    print(f"Random 4x3 singular values: {'PASS' if np.allclose(sigma3[:3], expected_sigma3, atol=1e-6) else 'FAIL'}")

    # Verify orthogonality of U and V
    print(f"U orthogonal:               {'PASS' if np.allclose(U3.T @ U3, np.eye(4), atol=1e-6) else 'FAIL'}")
    print(f"V orthogonal:               {'PASS' if np.allclose(Vt3 @ Vt3.T, np.eye(3), atol=1e-6) else 'FAIL'}")
