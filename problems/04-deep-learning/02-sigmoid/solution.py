"""
Problem: Sigmoid Activation Function
Category: Deep Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Numerically Stable Sigmoid
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Args:
        z: Input array of any shape

    Returns:
        sigmoid(z) in range (0, 1)
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid: s(z) * (1 - s(z)).

    Args:
        z: Input (pre-activation values)

    Returns:
        Derivative at each point
    """
    s = sigmoid(z)
    return s * (1.0 - s)


# ============================================================
# Approach 2: Sigmoid Neuron with Forward/Backward
# Time Complexity: O(n * d)
# Space Complexity: O(n)
# ============================================================

def sigmoid_neuron_forward(X: np.ndarray, weights: np.ndarray,
                           bias: float) -> np.ndarray:
    """
    Single neuron with sigmoid activation.

    Args:
        X: Input (n_samples, n_features)
        weights: (n_features,)
        bias: scalar

    Returns:
        Activations (n_samples,)
    """
    z = X @ weights + bias
    return sigmoid(z)


def sigmoid_neuron_backward(X: np.ndarray, weights: np.ndarray,
                            bias: float, y: np.ndarray):
    """
    Backward pass using binary cross-entropy loss.

    Args:
        X: Input (n_samples, n_features)
        weights: (n_features,)
        bias: scalar
        y: True labels (n_samples,)

    Returns:
        Tuple of (dw, db, loss)
    """
    n = X.shape[0]
    a = sigmoid_neuron_forward(X, weights, bias)

    # Binary cross-entropy loss
    eps = 1e-15
    a_clip = np.clip(a, eps, 1 - eps)
    loss = -np.mean(y * np.log(a_clip) + (1 - y) * np.log(1 - a_clip))

    # Gradients
    dz = a - y  # (n_samples,)
    dw = (X.T @ dz) / n  # (n_features,)
    db = np.mean(dz)

    return dw, db, loss


# ============================================================
# Approach 3: Piecewise Stable Sigmoid
# ============================================================

def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """
    Piecewise stable sigmoid that avoids overflow for both
    large positive and large negative values.
    """
    result = np.zeros_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask

    # For z >= 0: 1 / (1 + exp(-z))
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

    # For z < 0: exp(z) / (1 + exp(z)) to avoid exp of large positive
    ez = np.exp(z[neg_mask])
    result[neg_mask] = ez / (1.0 + ez)

    return result


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test basic sigmoid values
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    s = sigmoid(z)
    print(f"sigmoid(z):      {np.round(s, 4)}")
    print(f"sigmoid(0)=0.5:  {'PASS' if np.isclose(s[2], 0.5) else 'FAIL'}")
    print(f"Symmetry:        {'PASS' if np.allclose(s + sigmoid(-z), 1.0) else 'FAIL'}")

    # Test derivative
    d = sigmoid_derivative(np.array([0.0]))
    print(f"sigmoid'(0)=0.25: {'PASS' if np.isclose(d[0], 0.25) else 'FAIL'}")

    # Test numerical stability
    extreme = np.array([-1000, -500, 500, 1000])
    s_ext = sigmoid(extreme)
    print(f"Extreme values:  {'PASS' if np.all(np.isfinite(s_ext)) else 'FAIL'}")
    print(f"  sigmoid(-1000)={s_ext[0]:.6f}, sigmoid(1000)={s_ext[3]:.6f}")

    # Test piecewise stable
    s_stable = sigmoid_stable(z)
    print(f"Piecewise match: {'PASS' if np.allclose(s, s_stable) else 'FAIL'}")

    # Test neuron forward/backward
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.float64)
    w = np.zeros(2)
    b = 0.0

    # Train for a few steps
    for _ in range(1000):
        dw, db, loss = sigmoid_neuron_backward(X, w, b, y)
        w -= 0.5 * dw
        b -= 0.5 * db

    preds = (sigmoid_neuron_forward(X, w, b) >= 0.5).astype(float)
    print(f"Neuron AND gate: {'PASS' if np.array_equal(preds, y) else 'FAIL'}")
    print(f"Final loss:      {loss:.4f}")
