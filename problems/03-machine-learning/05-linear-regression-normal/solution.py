"""
Problem: Linear Regression (Normal Equation)
Category: Machine Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Normal Equation
# Time Complexity: O(n*d^2 + d^3)
# Space Complexity: O(d^2)
# ============================================================

def linear_regression_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression using the normal equation.
    Automatically adds a bias column.
    θ = (X^T X)^(-1) X^T y

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)

    Returns:
        Weight vector of shape (n_features + 1,) where index 0 is the bias
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = X.shape[0]

    # Add bias column (column of ones)
    X_b = np.hstack([np.ones((n, 1)), X])

    # Normal equation: θ = (X^T X)^(-1) X^T y
    XtX = X_b.T @ X_b
    Xty = X_b.T @ y
    theta = np.linalg.inv(XtX) @ Xty

    return theta


def predict_normal(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Make predictions using fitted weights."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    X_b = np.hstack([np.ones((n, 1)), X])
    return X_b @ theta


# ============================================================
# Approach 2: Pseudoinverse (More Stable)
# Time Complexity: O(n*d^2 + d^3)
# Space Complexity: O(n*d)
# ============================================================

def linear_regression_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression using the Moore-Penrose pseudoinverse.
    θ = X^+ y where X^+ is the pseudoinverse.
    More numerically stable than the normal equation.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)

    Returns:
        Weight vector of shape (n_features + 1,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = X.shape[0]
    X_b = np.hstack([np.ones((n, 1)), X])

    # Pseudoinverse via SVD
    U, s, Vt = np.linalg.svd(X_b, full_matrices=False)
    # X^+ = V Σ^+ U^T
    s_inv = np.zeros_like(s)
    for i in range(len(s)):
        if s[i] > 1e-10:
            s_inv[i] = 1.0 / s[i]

    theta = Vt.T @ np.diag(s_inv) @ U.T @ y
    return theta


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: y = 2x
    X1 = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y1 = np.array([2, 4, 6, 8], dtype=np.float64)

    theta1 = linear_regression_normal(X1, y1)
    print(f"Test 1 weights: bias={theta1[0]:.4f}, slope={theta1[1]:.4f}")
    print(f"Test 1 bias ≈ 0:   {'PASS' if abs(theta1[0]) < 1e-10 else 'FAIL'}")
    print(f"Test 1 slope = 2:  {'PASS' if abs(theta1[1] - 2.0) < 1e-10 else 'FAIL'}")

    # Prediction
    pred1 = predict_normal(np.array([[5]]), theta1)
    print(f"Predict x=5 → 10: {'PASS' if abs(pred1[0] - 10.0) < 1e-10 else 'FAIL'}")

    # Test Example 2: y = 3 + x1 + 2*x2
    X2 = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float64)
    y2 = np.array([6, 8, 9, 11], dtype=np.float64)

    theta2 = linear_regression_normal(X2, y2)
    print(f"\nTest 2 weights: {theta2}")
    print(f"bias ≈ 3:  {'PASS' if abs(theta2[0] - 3.0) < 1e-10 else 'FAIL'}")
    print(f"w1 ≈ 1:    {'PASS' if abs(theta2[1] - 1.0) < 1e-10 else 'FAIL'}")
    print(f"w2 ≈ 2:    {'PASS' if abs(theta2[2] - 2.0) < 1e-10 else 'FAIL'}")

    # Test: Pseudoinverse matches
    theta2_pinv = linear_regression_pinv(X2, y2)
    print(f"Pinv matches: {'PASS' if np.allclose(theta2, theta2_pinv) else 'FAIL'}")

    # Test: Predictions match training data
    preds = predict_normal(X2, theta2)
    print(f"Train fit:    {'PASS' if np.allclose(preds, y2) else 'FAIL'}")
