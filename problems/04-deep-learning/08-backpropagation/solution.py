"""
Problem: Backpropagation
Category: Deep Learning
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================
# Activation Functions and Derivatives
# ============================================================

def relu(z):
    return np.maximum(0, z)

def relu_backward(dA, Z):
    return dA * (Z > 0).astype(np.float64)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def tanh_act(z):
    return np.tanh(z)

def tanh_backward(dA, Z):
    t = np.tanh(Z)
    return dA * (1 - t**2)

def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def linear(z):
    return z

def linear_backward(dA, Z):
    return dA

FORWARD = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh_act,
           "softmax": softmax, "linear": linear}
BACKWARD = {"relu": relu_backward, "sigmoid": sigmoid_backward,
            "tanh": tanh_backward, "linear": linear_backward}


# ============================================================
# Approach 1: Full Forward + Backward Pass
# Time Complexity: O(N * sum(d_l * d_{l+1}))
# Space Complexity: O(N * sum(d_l))
# ============================================================

def forward(X: np.ndarray, weights: List[np.ndarray],
            biases: List[np.ndarray],
            activations: List[str]) -> Tuple[np.ndarray, Dict]:
    """Forward pass returning output and cache."""
    cache = {"Z": [], "A": [X.astype(np.float64)]}
    A = X.astype(np.float64)

    for l in range(len(weights)):
        Z = A @ weights[l] + biases[l]
        cache["Z"].append(Z)
        A = FORWARD[activations[l]](Z)
        cache["A"].append(A)

    return A, cache


def backward(y_true: np.ndarray, cache: Dict,
             weights: List[np.ndarray],
             activations: List[str]) -> Dict:
    """
    Backward pass computing gradients for all parameters.

    Assumes output layer uses softmax + cross-entropy.

    Args:
        y_true: One-hot true labels (N, C)
        cache: From forward pass
        weights: List of weight matrices
        activations: List of activation names

    Returns:
        Dict with "dW" and "db" lists
    """
    L = len(weights)
    N = y_true.shape[0]
    grads = {"dW": [None] * L, "db": [None] * L}

    # Output layer: softmax + cross-entropy combined gradient
    A_out = cache["A"][L]
    dZ = (A_out - y_true) / N

    for l in reversed(range(L)):
        A_prev = cache["A"][l]

        grads["dW"][l] = A_prev.T @ dZ
        grads["db"][l] = np.sum(dZ, axis=0)

        if l > 0:
            dA = dZ @ weights[l].T
            dZ = BACKWARD[activations[l - 1]](dA, cache["Z"][l - 1])

    return grads


def gradient_check(X: np.ndarray, y: np.ndarray,
                   weights: List[np.ndarray],
                   biases: List[np.ndarray],
                   activations: List[str],
                   epsilon: float = 1e-5) -> float:
    """
    Numerical gradient check.

    Returns max relative error across all parameters.
    """
    # Analytical gradients
    _, cache = forward(X, weights, biases, activations)
    grads = backward(y, cache, weights, activations)

    max_error = 0.0

    def compute_loss(w_list, b_list):
        out, _ = forward(X, w_list, b_list, activations)
        out_clip = np.clip(out, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y * np.log(out_clip), axis=1))

    # Check weight gradients
    for l in range(len(weights)):
        for i in range(weights[l].shape[0]):
            for j in range(weights[l].shape[1]):
                w_plus = [w.copy() for w in weights]
                w_minus = [w.copy() for w in weights]
                w_plus[l][i, j] += epsilon
                w_minus[l][i, j] -= epsilon

                loss_plus = compute_loss(w_plus, biases)
                loss_minus = compute_loss(w_minus, biases)
                grad_num = (loss_plus - loss_minus) / (2 * epsilon)
                grad_ana = grads["dW"][l][i, j]

                denom = max(abs(grad_ana) + abs(grad_num), 1e-8)
                error = abs(grad_ana - grad_num) / denom
                max_error = max(max_error, error)

    return max_error


# ============================================================
# Approach 2: Training Loop
# ============================================================

def train_step(X, y, weights, biases, activations, lr=0.01):
    """Single training step: forward, backward, update."""
    out, cache = forward(X, weights, biases, activations)
    grads = backward(y, cache, weights, activations)

    for l in range(len(weights)):
        weights[l] -= lr * grads["dW"][l]
        biases[l] -= lr * grads["db"][l]

    # Compute loss
    out_clip = np.clip(out, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y * np.log(out_clip), axis=1))
    return loss


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Gradient shapes
    X = np.random.randn(4, 2)
    y = np.eye(3)[[0, 1, 2, 0]]
    W = [np.random.randn(2, 5) * 0.1, np.random.randn(5, 3) * 0.1]
    b = [np.zeros(5), np.zeros(3)]
    acts = ["relu", "softmax"]

    out, cache = forward(X, W, b, acts)
    grads = backward(y, cache, W, acts)
    print(f"Output shape:     {'PASS' if out.shape == (4, 3) else 'FAIL'}")
    print(f"dW[0] shape:      {'PASS' if grads['dW'][0].shape == (2, 5) else 'FAIL'}")
    print(f"dW[1] shape:      {'PASS' if grads['dW'][1].shape == (5, 3) else 'FAIL'}")
    print(f"db[0] shape:      {'PASS' if grads['db'][0].shape == (5,) else 'FAIL'}")
    print(f"db[1] shape:      {'PASS' if grads['db'][1].shape == (3,) else 'FAIL'}")

    # Test 2: Gradient check
    X_check = np.random.randn(3, 2)
    y_check = np.eye(2)[[0, 1, 0]]
    W_check = [np.random.randn(2, 4) * 0.1, np.random.randn(4, 2) * 0.1]
    b_check = [np.zeros(4), np.zeros(2)]
    acts_check = ["relu", "softmax"]

    max_err = gradient_check(X_check, y_check, W_check, b_check, acts_check)
    print(f"Gradient check:   {'PASS' if max_err < 1e-4 else 'FAIL'} (max err: {max_err:.2e})")

    # Test 3: Training reduces loss
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float64)  # XOR-like
    W_t = [np.random.randn(2, 8) * 0.5, np.random.randn(8, 2) * 0.5]
    b_t = [np.zeros(8), np.zeros(2)]

    loss_start = train_step(X_train, y_train, W_t, b_t, acts, lr=0)
    for _ in range(500):
        loss = train_step(X_train, y_train, W_t, b_t, acts, lr=0.1)
    print(f"Training reduces: {'PASS' if loss < loss_start else 'FAIL'} ({loss_start:.4f} -> {loss:.4f})")

    # Test 4: Sigmoid hidden layer
    W_sig = [np.random.randn(2, 4) * 0.5, np.random.randn(4, 2) * 0.5]
    b_sig = [np.zeros(4), np.zeros(2)]
    acts_sig = ["sigmoid", "softmax"]
    max_err_sig = gradient_check(X_check, y_check, W_sig, b_sig, acts_sig)
    print(f"Sigmoid grad chk: {'PASS' if max_err_sig < 1e-4 else 'FAIL'} (max err: {max_err_sig:.2e})")
