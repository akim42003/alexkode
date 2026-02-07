"""
Problem: Single Neuron (Perceptron)
Category: Deep Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Perceptron with Step Activation
# Time Complexity: O(epochs * n * d)
# Space Complexity: O(d)
# ============================================================

def step_activation(z: np.ndarray) -> np.ndarray:
    """Heaviside step function: 1 if z >= 0, else 0."""
    return (z >= 0).astype(np.float64)


def perceptron_forward(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Forward pass of a single perceptron.

    Args:
        X: Input (n_samples, n_features)
        weights: Weight vector (n_features,)
        bias: Scalar bias

    Returns:
        Binary predictions (n_samples,)
    """
    z = X @ weights + bias
    return step_activation(z)


def train_perceptron(X: np.ndarray, y: np.ndarray,
                     lr: float = 0.1, epochs: int = 100):
    """
    Train a perceptron using the perceptron learning rule.

    Args:
        X: Training data (n_samples, n_features)
        y: Binary labels (n_samples,)
        lr: Learning rate
        epochs: Number of passes over the data

    Returns:
        Tuple of (weights, bias)
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        for i in range(n_samples):
            z = np.dot(X[i], weights) + bias
            y_hat = 1.0 if z >= 0 else 0.0
            error = y[i] - y_hat
            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias


# ============================================================
# Approach 2: Vectorized Epoch with Convergence Check
# Time Complexity: O(epochs * n * d) but may stop early
# Space Complexity: O(n)
# ============================================================

def train_perceptron_early_stop(X: np.ndarray, y: np.ndarray,
                                lr: float = 0.1, max_epochs: int = 1000):
    """Train with early stopping when all predictions are correct."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    for epoch in range(max_epochs):
        errors = 0
        for i in range(n_samples):
            y_hat = 1.0 if (np.dot(X[i], weights) + bias) >= 0 else 0.0
            error = y[i] - y_hat
            if error != 0:
                weights += lr * error * X[i]
                bias += lr * error
                errors += 1
        if errors == 0:
            break

    return weights, bias, epoch + 1


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test AND gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_and = np.array([0, 0, 0, 1], dtype=np.float64)
    w, b = train_perceptron(X_and, y_and, lr=0.1, epochs=100)
    pred_and = perceptron_forward(X_and, w, b)
    print(f"AND gate:       {'PASS' if np.array_equal(pred_and, y_and) else 'FAIL'}")

    # Test OR gate
    y_or = np.array([0, 1, 1, 1], dtype=np.float64)
    w, b = train_perceptron(X_and, y_or, lr=0.1, epochs=100)
    pred_or = perceptron_forward(X_and, w, b)
    print(f"OR gate:        {'PASS' if np.array_equal(pred_or, y_or) else 'FAIL'}")

    # Test NAND gate
    y_nand = np.array([1, 1, 1, 0], dtype=np.float64)
    w, b = train_perceptron(X_and, y_nand, lr=0.1, epochs=100)
    pred_nand = perceptron_forward(X_and, w, b)
    print(f"NAND gate:      {'PASS' if np.array_equal(pred_nand, y_nand) else 'FAIL'}")

    # Test early stopping
    w2, b2, ep = train_perceptron_early_stop(X_and, y_and, lr=0.1, max_epochs=1000)
    pred2 = perceptron_forward(X_and, w2, b2)
    print(f"Early stop AND: {'PASS' if np.array_equal(pred2, y_and) else 'FAIL'} (converged in {ep} epochs)")

    # Test step activation
    z = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    expected = np.array([0, 0, 1, 1, 1], dtype=np.float64)
    print(f"Step activation: {'PASS' if np.array_equal(step_activation(z), expected) else 'FAIL'}")
