"""
Problem: Binary Cross-Entropy Loss
Category: Deep Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Vectorized BCE
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                         epsilon: float = 1e-15) -> np.ndarray:
    """
    Per-sample binary cross-entropy loss.

    Args:
        y_true: True labels (n,), values in {0, 1}
        y_pred: Predicted probabilities (n,), values in (0, 1)
        epsilon: Clipping constant for numerical stability

    Returns:
        Per-sample losses (n,)
    """
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def binary_cross_entropy_mean(y_true: np.ndarray, y_pred: np.ndarray,
                              epsilon: float = 1e-15) -> float:
    """Mean BCE loss over the batch."""
    return float(np.mean(binary_cross_entropy(y_true, y_pred, epsilon)))


def binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray,
                                  epsilon: float = 1e-15) -> np.ndarray:
    """
    Gradient of mean BCE loss w.r.t. y_pred.

    dL/dp = -(y/p - (1-y)/(1-p)) / N
    """
    n = len(y_true)
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -(y_true / p - (1 - y_true) / (1 - p)) / n


# ============================================================
# Approach 2: Logits-based BCE (numerically superior)
# ============================================================

def bce_with_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    BCE from logits without explicitly computing sigmoid.
    L = max(z, 0) - z*y + log(1 + exp(-|z|))

    This avoids numerical issues from log(sigmoid(z)).

    Args:
        y_true: True labels (n,)
        logits: Raw logits before sigmoid (n,)

    Returns:
        Mean loss
    """
    # Stable formula: max(z,0) - z*y + log(1 + exp(-|z|))
    relu_z = np.maximum(logits, 0)
    loss = relu_z - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def bce_with_logits_gradient(y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Gradient of logits-based BCE w.r.t. logits: (sigmoid(z) - y) / N."""
    sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    return (sigmoid_z - y_true) / len(y_true)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test basic BCE
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.3])

    losses = binary_cross_entropy(y_true, y_pred)
    mean_loss = binary_cross_entropy_mean(y_true, y_pred)
    print(f"Per-sample losses: {np.round(losses, 4)}")
    print(f"Mean loss:         {mean_loss:.4f}")
    print(f"Loss[0]=-log(0.9): {'PASS' if np.isclose(losses[0], -np.log(0.9), atol=1e-4) else 'FAIL'}")
    print(f"Loss[1]=-log(0.9): {'PASS' if np.isclose(losses[1], -np.log(0.9), atol=1e-4) else 'FAIL'}")

    # Test numerical stability
    y_edge = np.array([1.0])
    p_edge = np.array([0.0])
    loss_edge = binary_cross_entropy(y_edge, p_edge)
    print(f"Edge case finite: {'PASS' if np.isfinite(loss_edge[0]) else 'FAIL'}")

    # Gradient numerical check
    eps = 1e-5
    grad = binary_cross_entropy_gradient(y_true, y_pred)
    grad_num = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        p_plus = y_pred.copy()
        p_plus[i] += eps
        p_minus = y_pred.copy()
        p_minus[i] -= eps
        grad_num[i] = (binary_cross_entropy_mean(y_true, p_plus) -
                       binary_cross_entropy_mean(y_true, p_minus)) / (2 * eps)
    print(f"Gradient check:   {'PASS' if np.allclose(grad, grad_num, atol=1e-4) else 'FAIL'}")

    # Test logits-based BCE
    logits = np.array([2.0, -2.0, 1.5, -0.5])
    sigmoid_vals = 1.0 / (1.0 + np.exp(-logits))
    loss_logits = bce_with_logits(y_true, logits)
    loss_probs = binary_cross_entropy_mean(y_true, sigmoid_vals)
    print(f"Logits BCE match: {'PASS' if np.isclose(loss_logits, loss_probs, atol=1e-6) else 'FAIL'}")

    # Test logits gradient
    grad_logits = bce_with_logits_gradient(y_true, logits)
    grad_logits_num = np.zeros_like(logits)
    for i in range(len(logits)):
        l_plus = logits.copy()
        l_plus[i] += eps
        l_minus = logits.copy()
        l_minus[i] -= eps
        grad_logits_num[i] = (bce_with_logits(y_true, l_plus) -
                              bce_with_logits(y_true, l_minus)) / (2 * eps)
    print(f"Logits grad check:{'PASS' if np.allclose(grad_logits, grad_logits_num, atol=1e-4) else 'FAIL'}")

    # Test extreme logits stability
    extreme_logits = np.array([500.0, -500.0])
    extreme_y = np.array([1.0, 0.0])
    loss_extreme = bce_with_logits(extreme_y, extreme_logits)
    print(f"Extreme stability: {'PASS' if np.isfinite(loss_extreme) and loss_extreme < 1e-5 else 'FAIL'}")
