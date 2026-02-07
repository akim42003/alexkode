"""
Problem: Linear Regression (Gradient Descent)
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple, List, Optional


# ============================================================
# Approach 1: Batch Gradient Descent
# Time Complexity: O(n * d * iterations)
# Space Complexity: O(d)
# ============================================================

def linear_regression_batch_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.01,
                                n_iter: int = 1000, seed: int = 42) -> Tuple[np.ndarray, List[float]]:
    """
    Linear regression using batch gradient descent.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        lr: Learning rate
        n_iter: Number of iterations
        seed: Random seed for weight initialization

    Returns:
        Tuple of (weights with bias at index 0, loss history)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    # Add bias column
    X_b = np.hstack([np.ones((n, 1)), X])

    # Initialize weights
    rng = np.random.RandomState(seed)
    theta = rng.randn(d + 1) * 0.01

    loss_history = []

    for _ in range(n_iter):
        # Predictions
        y_pred = X_b @ theta

        # MSE loss
        errors = y_pred - y
        loss = np.sum(errors ** 2) / n
        loss_history.append(loss)

        # Gradient: (2/n) * X^T * (X*theta - y)
        gradient = (2.0 / n) * (X_b.T @ errors)

        # Update weights
        theta -= lr * gradient

    return theta, loss_history


# ============================================================
# Approach 2: Mini-Batch Gradient Descent
# Time Complexity: O(batch_size * d * iterations)
# Space Complexity: O(d)
# ============================================================

def linear_regression_minibatch_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.01,
                                    n_iter: int = 1000, batch_size: int = 16,
                                    seed: int = 42) -> Tuple[np.ndarray, List[float]]:
    """
    Linear regression using mini-batch gradient descent.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        lr: Learning rate
        n_iter: Number of iterations (epochs)
        batch_size: Size of each mini-batch
        seed: Random seed

    Returns:
        Tuple of (weights with bias, loss history per epoch)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    X_b = np.hstack([np.ones((n, 1)), X])

    rng = np.random.RandomState(seed)
    theta = rng.randn(d + 1) * 0.01

    loss_history = []

    for epoch in range(n_iter):
        # Shuffle data
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            X_batch = X_b[batch_idx]
            y_batch = y[batch_idx]

            y_pred = X_batch @ theta
            errors = y_pred - y_batch
            gradient = (2.0 / len(batch_idx)) * (X_batch.T @ errors)
            theta -= lr * gradient

        # Record full loss at end of epoch
        full_pred = X_b @ theta
        loss = np.sum((full_pred - y) ** 2) / n
        loss_history.append(loss)

    return theta, loss_history


def predict_gd(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict using fitted weights."""
    X = np.asarray(X, dtype=np.float64)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b @ theta


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: y = 2x
    X1 = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y1 = np.array([2, 4, 6, 8], dtype=np.float64)

    theta1, losses1 = linear_regression_batch_gd(X1, y1, lr=0.1, n_iter=1000)
    print(f"Batch GD: bias={theta1[0]:.4f}, slope={theta1[1]:.4f}")
    print(f"Slope ≈ 2:     {'PASS' if abs(theta1[1] - 2.0) < 0.01 else 'FAIL'}")
    print(f"Loss decreases: {'PASS' if losses1[-1] < losses1[0] else 'FAIL'}")
    print(f"Final loss ≈ 0: {'PASS' if losses1[-1] < 0.01 else 'FAIL'} ({losses1[-1]:.6f})")

    # Test Example 2: Multi-feature
    X2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    y2 = np.array([5, 11, 17], dtype=np.float64)

    theta2, losses2 = linear_regression_batch_gd(X2, y2, lr=0.01, n_iter=5000)
    print(f"\nMulti-feature: {theta2}")
    preds2 = predict_gd(X2, theta2)
    print(f"Train fit:     {'PASS' if np.allclose(preds2, y2, atol=0.1) else 'FAIL'}")

    # Test Mini-batch
    theta3, losses3 = linear_regression_minibatch_gd(X1, y1, lr=0.1, n_iter=500, batch_size=2)
    print(f"\nMini-batch: bias={theta3[0]:.4f}, slope={theta3[1]:.4f}")
    print(f"Slope ≈ 2:     {'PASS' if abs(theta3[1] - 2.0) < 0.1 else 'FAIL'}")

    # Test: Convergence (loss history monotonically decreasing for batch GD)
    decreasing = all(losses1[i] >= losses1[i+1] - 1e-10 for i in range(len(losses1)-1))
    print(f"Monotonic loss: {'PASS' if decreasing else 'FAIL'}")
