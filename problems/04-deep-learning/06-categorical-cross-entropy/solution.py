"""
Problem: Categorical Cross-Entropy Loss
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Vectorized Categorical Cross-Entropy
# Time Complexity: O(N * C)
# Space Complexity: O(N * C)
# ============================================================

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                              epsilon: float = 1e-15) -> np.ndarray:
    """
    Per-sample categorical cross-entropy loss.

    Args:
        y_true: One-hot labels (N, C)
        y_pred: Predicted probabilities (N, C)
        epsilon: Clipping for numerical stability

    Returns:
        Per-sample losses (N,)
    """
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(p), axis=1)


def categorical_cross_entropy_mean(y_true: np.ndarray, y_pred: np.ndarray,
                                   epsilon: float = 1e-15) -> float:
    """Mean categorical cross-entropy loss."""
    return float(np.mean(categorical_cross_entropy(y_true, y_pred, epsilon)))


def categorical_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray,
                                       epsilon: float = 1e-15) -> np.ndarray:
    """Gradient of mean CCE w.r.t. y_pred: -y_true / (N * p)."""
    n = y_true.shape[0]
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -y_true / (n * p)


# ============================================================
# Approach 2: Combined Softmax + Cross-Entropy (from logits)
# ============================================================

def softmax_cross_entropy(logits: np.ndarray, y_true: np.ndarray):
    """
    Numerically stable softmax + cross-entropy from logits.
    Uses log-sum-exp trick.

    Args:
        logits: Raw scores (N, C)
        y_true: One-hot labels (N, C)

    Returns:
        Tuple of (mean_loss, probabilities, gradient_wrt_logits)
    """
    # Log-sum-exp trick
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=1, keepdims=True))

    log_probs = shifted - log_sum_exp
    losses = -np.sum(y_true * log_probs, axis=1)
    mean_loss = float(np.mean(losses))

    probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    n = y_true.shape[0]
    gradient = (probabilities - y_true) / n

    return mean_loss, probabilities, gradient


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test basic CCE
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])

    losses = categorical_cross_entropy(y_true, y_pred)
    mean_loss = categorical_cross_entropy_mean(y_true, y_pred)
    print(f"Per-sample losses: {np.round(losses, 4)}")
    print(f"Mean loss:         {mean_loss:.4f}")
    print(f"Loss[0]=-log(0.9): {'PASS' if np.isclose(losses[0], -np.log(0.9), atol=1e-4) else 'FAIL'}")

    # Test numerical stability
    y_edge = np.array([[1, 0, 0]], dtype=np.float64)
    p_edge = np.array([[0.0, 0.5, 0.5]])
    loss_edge = categorical_cross_entropy(y_edge, p_edge)
    print(f"Edge case finite: {'PASS' if np.isfinite(loss_edge[0]) else 'FAIL'}")

    # Gradient check
    eps = 1e-5
    grad = categorical_cross_entropy_gradient(y_true, y_pred)
    grad_num = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            p_plus = y_pred.copy()
            p_plus[i, j] += eps
            p_minus = y_pred.copy()
            p_minus[i, j] -= eps
            grad_num[i, j] = (categorical_cross_entropy_mean(y_true, p_plus) -
                              categorical_cross_entropy_mean(y_true, p_minus)) / (2 * eps)
    max_err = np.max(np.abs(grad - grad_num) / (np.abs(grad) + 1e-8))
    print(f"Gradient check:    {'PASS' if max_err < 1e-4 else 'FAIL'} (max rel err: {max_err:.2e})")

    # Test softmax+CE
    np.random.seed(42)
    logits = np.random.randn(5, 3)
    labels = np.eye(3)[np.array([0, 1, 2, 0, 1])]
    loss_combined, probs, grad_logits = softmax_cross_entropy(logits, labels)

    # Compare with separate approach
    loss_separate = categorical_cross_entropy_mean(labels, probs)
    print(f"Combined vs separate: {'PASS' if np.isclose(loss_combined, loss_separate, atol=1e-6) else 'FAIL'}")

    # Gradient check for logits
    grad_logits_num = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            l_plus = logits.copy()
            l_plus[i, j] += eps
            l_minus = logits.copy()
            l_minus[i, j] -= eps
            loss_p, _, _ = softmax_cross_entropy(l_plus, labels)
            loss_m, _, _ = softmax_cross_entropy(l_minus, labels)
            grad_logits_num[i, j] = (loss_p - loss_m) / (2 * eps)
    logit_err = np.max(np.abs(grad_logits - grad_logits_num))
    print(f"Logits grad check: {'PASS' if logit_err < 1e-5 else 'FAIL'} (max err: {logit_err:.2e})")

    # Test large logits stability
    big_logits = np.array([[1000.0, 999.0, 998.0]])
    big_y = np.array([[1.0, 0.0, 0.0]])
    loss_big, probs_big, _ = softmax_cross_entropy(big_logits, big_y)
    print(f"Large logits:      {'PASS' if np.isfinite(loss_big) else 'FAIL'}")
