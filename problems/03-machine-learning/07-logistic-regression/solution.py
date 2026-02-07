"""
Problem: Logistic Regression
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple, List


# ============================================================
# Approach 1: Gradient Descent
# Time Complexity: O(n * d * iterations)
# Space Complexity: O(d)
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_regression_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                            n_iter: int = 1000, seed: int = 42) -> Tuple[np.ndarray, List[float]]:
    """
    Fit logistic regression using gradient descent.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        lr: Learning rate
        n_iter: Number of iterations
        seed: Random seed

    Returns:
        Tuple of (weights with bias at index 0, loss history)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    # Add bias
    X_b = np.hstack([np.ones((n, 1)), X])

    # Initialize weights
    rng = np.random.RandomState(seed)
    theta = rng.randn(d + 1) * 0.01

    loss_history = []
    eps = 1e-15

    for _ in range(n_iter):
        # Forward pass
        z = X_b @ theta
        p = sigmoid(z)

        # Binary cross-entropy loss
        p_clipped = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        loss_history.append(loss)

        # Gradient
        gradient = (1.0 / n) * (X_b.T @ (p - y))

        # Update
        theta -= lr * gradient

    return theta, loss_history


def predict_logistic(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict binary labels."""
    X = np.asarray(X, dtype=np.float64)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    probabilities = sigmoid(X_b @ theta)
    return (probabilities >= threshold).astype(int)


def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict probabilities."""
    X = np.asarray(X, dtype=np.float64)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return sigmoid(X_b @ theta)


# ============================================================
# Approach 2: With L2 Regularization
# Time Complexity: O(n * d * iterations)
# Space Complexity: O(d)
# ============================================================

def logistic_regression_l2(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                            n_iter: int = 1000, lambda_reg: float = 0.01,
                            seed: int = 42) -> Tuple[np.ndarray, List[float]]:
    """
    Logistic regression with L2 regularization.

    Args:
        X: Feature matrix
        y: Binary labels
        lr: Learning rate
        n_iter: Iterations
        lambda_reg: Regularization strength
        seed: Random seed

    Returns:
        Tuple of (weights, loss history)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    X_b = np.hstack([np.ones((n, 1)), X])
    rng = np.random.RandomState(seed)
    theta = rng.randn(d + 1) * 0.01

    loss_history = []
    eps = 1e-15

    for _ in range(n_iter):
        z = X_b @ theta
        p = sigmoid(z)

        p_clipped = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        # Add regularization term (don't regularize bias)
        loss += (lambda_reg / (2 * n)) * np.sum(theta[1:] ** 2)
        loss_history.append(loss)

        gradient = (1.0 / n) * (X_b.T @ (p - y))
        # Add regularization gradient (skip bias)
        reg_grad = np.zeros_like(theta)
        reg_grad[1:] = (lambda_reg / n) * theta[1:]
        gradient += reg_grad

        theta -= lr * gradient

    return theta, loss_history


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: 1D linearly separable
    X1 = np.array([[0.5], [1.5], [2.5], [3.5]])
    y1 = np.array([0, 0, 1, 1])

    theta1, losses1 = logistic_regression_gd(X1, y1, lr=1.0, n_iter=1000)
    preds1 = predict_logistic(X1, theta1)
    accuracy1 = np.mean(preds1 == y1)
    print(f"Test 1 accuracy: {'PASS' if accuracy1 == 1.0 else 'FAIL'} ({accuracy1})")
    print(f"Loss decreases:  {'PASS' if losses1[-1] < losses1[0] else 'FAIL'}")

    # Test Example 2: 2D linearly separable
    X2 = np.array([[1, 1], [1, 2], [2, 1], [3, 3], [3, 4], [4, 3]], dtype=np.float64)
    y2 = np.array([0, 0, 0, 1, 1, 1])

    theta2, losses2 = logistic_regression_gd(X2, y2, lr=0.5, n_iter=2000)
    preds2 = predict_logistic(X2, theta2)
    accuracy2 = np.mean(preds2 == y2)
    print(f"Test 2 accuracy: {'PASS' if accuracy2 == 1.0 else 'FAIL'} ({accuracy2})")

    # Test probabilities
    probs = predict_proba(X2, theta2)
    print(f"Probs in [0,1]:  {'PASS' if np.all((probs >= 0) & (probs <= 1)) else 'FAIL'}")

    # Test L2 regularization
    theta_l2, losses_l2 = logistic_regression_l2(X2, y2, lr=0.5, n_iter=2000, lambda_reg=0.1)
    preds_l2 = predict_logistic(X2, theta_l2)
    print(f"L2 accuracy:     {'PASS' if np.mean(preds_l2 == y2) == 1.0 else 'FAIL'}")

    # L2 weights should be smaller than non-regularized
    print(f"L2 smaller wts:  {'PASS' if np.sum(theta_l2[1:]**2) < np.sum(theta2[1:]**2) else 'FAIL'}")
