"""
Problem: Forward Propagation
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================
# Approach 1: Layer-by-Layer Forward Pass
# Time Complexity: O(N * sum(d_l * d_{l+1}))
# Space Complexity: O(N * sum(d_l))
# ============================================================

def _relu(z):
    return np.maximum(0, z)

def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def _tanh(z):
    return np.tanh(z)

def _softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def _linear(z):
    return z

ACTIVATIONS = {
    "relu": _relu,
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "softmax": _softmax,
    "linear": _linear,
}


def forward_propagation(X: np.ndarray,
                        weights: List[np.ndarray],
                        biases: List[np.ndarray],
                        activations: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Forward pass through a multi-layer network.

    Args:
        X: Input data (N, d_in)
        weights: List of weight matrices
        biases: List of bias vectors
        activations: List of activation function names

    Returns:
        Tuple of (output, cache) where cache contains Z and A lists
    """
    L = len(weights)
    cache = {"Z": [], "A": [X.copy()]}

    A = X.astype(np.float64)

    for l in range(L):
        Z = A @ weights[l] + biases[l]
        cache["Z"].append(Z)

        act_fn = ACTIVATIONS[activations[l]]
        A = act_fn(Z)
        cache["A"].append(A)

    return A, cache


# ============================================================
# Approach 2: MLP Class with Layer Management
# ============================================================

class MLP:
    """Multi-layer perceptron with configurable architecture."""

    def __init__(self, layer_dims: List[int], activations: List[str],
                 seed: int = 42):
        """
        Args:
            layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
            activations: Activation for each layer (len = len(layer_dims) - 1)
        """
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        self.activations = activations

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            # He initialization
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return forward_propagation(X, self.weights, self.biases,
                                   self.activations)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Single layer, ReLU
    X1 = np.array([[1.0, 2.0]])
    W1 = [np.array([[0.1, 0.2], [0.3, 0.4]])]
    b1 = [np.array([0.5, 0.6])]
    out1, cache1 = forward_propagation(X1, W1, b1, ["relu"])
    expected1 = np.array([[1.2, 1.6]])
    print(f"Single ReLU layer: {'PASS' if np.allclose(out1, expected1, atol=1e-6) else 'FAIL'}")

    # Test 2: Two layers, relu -> sigmoid
    X2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    W2 = [np.array([[0.5, -0.5], [-0.5, 0.5]]),
          np.array([[1.0], [1.0]])]
    b2 = [np.zeros(2), np.zeros(1)]
    out2, cache2 = forward_propagation(X2, W2, b2, ["relu", "sigmoid"])
    print(f"Two-layer shape:   {'PASS' if out2.shape == (2, 1) else 'FAIL'}")
    print(f"Output in (0,1):   {'PASS' if np.all((out2 > 0) & (out2 < 1)) else 'FAIL'}")

    # Test 3: Softmax output
    X3 = np.array([[1.0, 2.0, 3.0]])
    W3 = [np.eye(3)]
    b3 = [np.zeros(3)]
    out3, _ = forward_propagation(X3, W3, b3, ["softmax"])
    print(f"Softmax sums to 1: {'PASS' if np.isclose(out3.sum(), 1.0) else 'FAIL'}")

    # Test 4: Cache structure
    print(f"Cache A length:    {'PASS' if len(cache2['A']) == 3 else 'FAIL'}")
    print(f"Cache Z length:    {'PASS' if len(cache2['Z']) == 2 else 'FAIL'}")
    print(f"Cache A[0] = X:    {'PASS' if np.allclose(cache2['A'][0], X2) else 'FAIL'}")

    # Test 5: MLP class
    mlp = MLP([2, 4, 3, 1], ["relu", "relu", "sigmoid"], seed=42)
    X5 = np.random.randn(10, 2)
    out5, cache5 = mlp.forward(X5)
    print(f"MLP output shape:  {'PASS' if out5.shape == (10, 1) else 'FAIL'}")
    print(f"MLP output range:  {'PASS' if np.all((out5 >= 0) & (out5 <= 1)) else 'FAIL'}")

    # Test 6: Linear activation (identity)
    out_lin, _ = forward_propagation(X1, W1, b1, ["linear"])
    expected_lin = X1 @ W1[0] + b1[0]
    print(f"Linear activation: {'PASS' if np.allclose(out_lin, expected_lin) else 'FAIL'}")
