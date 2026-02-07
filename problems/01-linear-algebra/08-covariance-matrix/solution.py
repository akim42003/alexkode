"""
Problem: Covariance Matrix
Category: Linear Algebra
Difficulty: Medium

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Matrix Formula (Centered Data)
# Time Complexity: O(n * d^2) where n=samples, d=features
# Space Complexity: O(d^2)
# ============================================================

def covariance_matrix_formula(X: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix using the matrix formula:
    C = (X_centered^T @ X_centered) / (n - 1)

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Covariance matrix of shape (n_features, n_features)
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for sample covariance")

    # Center the data by subtracting column means
    means = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        col_sum = 0.0
        for i in range(n):
            col_sum += X[i, j]
        means[j] = col_sum / n

    X_centered = X.astype(np.float64) - means

    # Covariance = X_centered^T @ X_centered / (n - 1)
    cov = (X_centered.T @ X_centered) / (n - 1)
    return cov


# ============================================================
# Approach 2: Element-wise Computation
# Time Complexity: O(d^2 * n)
# Space Complexity: O(d^2)
# ============================================================

def covariance_matrix_elementwise(X: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix by calculating each element individually.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Covariance matrix of shape (n_features, n_features)
    """
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples for sample covariance")

    X = X.astype(np.float64)

    # Compute means for each feature
    means = np.zeros(d)
    for j in range(d):
        for i in range(n):
            means[j] += X[i, j]
        means[j] /= n

    # Compute each covariance element
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            total = 0.0
            for k in range(n):
                total += (X[k, i] - means[i]) * (X[k, j] - means[j])
            cov[i, j] = total / (n - 1)

    return cov


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    expected1 = np.array([[4.0, 4.0], [4.0, 4.0]])

    for name, fn in [("Formula", covariance_matrix_formula),
                     ("Element-wise", covariance_matrix_elementwise)]:
        result = fn(X1)
        print(f"Test 1 ({name}):    {'PASS' if np.allclose(result, expected1) else 'FAIL'}")

    # Test Example 2
    X2 = np.array([[1, 5], [2, 3], [3, 1]], dtype=np.float64)
    expected2 = np.array([[1.0, -2.0], [-2.0, 4.0]])

    for name, fn in [("Formula", covariance_matrix_formula),
                     ("Element-wise", covariance_matrix_elementwise)]:
        result = fn(X2)
        print(f"Test 2 ({name}):    {'PASS' if np.allclose(result, expected2) else 'FAIL'}")

    # Test: Compare with numpy
    np.random.seed(42)
    X3 = np.random.randn(100, 5)
    expected3 = np.cov(X3, rowvar=False)
    result3 = covariance_matrix_formula(X3)
    print(f"100x5 vs np.cov:       {'PASS' if np.allclose(result3, expected3) else 'FAIL'}")

    # Test: Symmetric matrix
    result_sym = covariance_matrix_formula(X3)
    print(f"Symmetry check:        {'PASS' if np.allclose(result_sym, result_sym.T) else 'FAIL'}")

    # Test: Diagonal = variances
    variances = np.var(X3, axis=0, ddof=1)
    diag = np.diag(result_sym)
    print(f"Diagonal = variances:  {'PASS' if np.allclose(diag, variances) else 'FAIL'}")
