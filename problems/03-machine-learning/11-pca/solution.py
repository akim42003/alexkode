"""
Problem: Principal Component Analysis (PCA)
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Covariance Matrix + Eigendecomposition
# Time Complexity: O(n*d^2 + d^3)
# Space Complexity: O(d^2)
# ============================================================

def pca_eigen(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using covariance matrix eigendecomposition.

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of principal components to keep

    Returns:
        Tuple of (X_transformed, components, explained_variance_ratio)
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Step 1: Center the data
    mean = np.sum(X, axis=0) / n
    X_centered = X - mean

    # Step 2: Covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top n_components
    components = eigenvectors[:, :n_components].T  # (n_components, d)

    # Explained variance ratio
    total_var = np.sum(eigenvalues)
    explained_ratio = eigenvalues[:n_components] / total_var

    # Project data
    X_transformed = X_centered @ components.T  # (n, n_components)

    return X_transformed, components, explained_ratio


# ============================================================
# Approach 2: SVD-based PCA
# Time Complexity: O(n * d^2)
# Space Complexity: O(n * d)
# ============================================================

def pca_svd(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using SVD of the centered data matrix.
    More numerically stable than eigendecomposition.

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of principal components

    Returns:
        Tuple of (X_transformed, components, explained_variance_ratio)
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Center
    mean = np.sum(X, axis=0) / n
    X_centered = X - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Components are rows of Vt
    components = Vt[:n_components]  # (n_components, d)

    # Explained variance: S^2 / (n-1)
    explained_var = (S ** 2) / (n - 1)
    total_var = np.sum(explained_var)
    explained_ratio = explained_var[:n_components] / total_var

    # Project
    X_transformed = X_centered @ components.T

    return X_transformed, components, explained_ratio


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Perfectly correlated features
    X1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    X_t1, comp1, ratio1 = pca_eigen(X1, n_components=1)

    print(f"Test 1 shape:    {'PASS' if X_t1.shape == (4, 1) else 'FAIL'} ({X_t1.shape})")
    print(f"Test 1 var ≈ 1:  {'PASS' if ratio1[0] > 0.99 else 'FAIL'} ({ratio1[0]:.4f})")

    # Test Example 2
    X2 = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]], dtype=np.float64)
    X_t2, comp2, ratio2 = pca_eigen(X2, n_components=2)
    print(f"\nTest 2 ratios:   {ratio2}")
    print(f"Sum to 1:        {'PASS' if abs(np.sum(ratio2) - 1.0) < 1e-10 else 'FAIL'}")
    print(f"First > 0.9:     {'PASS' if ratio2[0] > 0.9 else 'FAIL'}")

    # Test: SVD matches eigendecomposition
    X_t2_svd, _, ratio2_svd = pca_svd(X2, n_components=2)
    print(f"SVD ratios match: {'PASS' if np.allclose(ratio2, ratio2_svd) else 'FAIL'}")

    # Test: Reconstruction with all components should be near-perfect
    X_t_full, comp_full, _ = pca_eigen(X2, n_components=2)
    mean2 = np.mean(X2, axis=0)
    X_reconstructed = X_t_full @ comp_full + mean2
    print(f"Reconstruction:  {'PASS' if np.allclose(X_reconstructed, X2) else 'FAIL'}")

    # Test: Dimensionality reduction
    np.random.seed(42)
    X3 = np.random.randn(100, 10)
    X_t3, _, ratio3 = pca_svd(X3, n_components=3)
    print(f"\n10D → 3D shape:  {'PASS' if X_t3.shape == (100, 3) else 'FAIL'}")
