"""
Problem: ReLU and Variants
Category: Deep Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Standard Implementations
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return (x > 0).astype(np.float64)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: x if x > 0, else alpha * x."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU."""
    return np.where(x > 0, 1.0, alpha)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU: x if x > 0, else alpha * (exp(x) - 1)."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU: 1 if x > 0, else alpha * exp(x)."""
    return np.where(x > 0, 1.0, alpha * np.exp(x))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU approximation using tanh."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Approximate derivative of GELU."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner**2
    d_inner = c * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner


# ============================================================
# Approach 2: Class-Based Activation Layers
# ============================================================

class ReLULayer:
    """ReLU as a layer with forward/backward."""

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(np.float64)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class LeakyReLULayer:
    """Leaky ReLU layer."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = np.where(x > 0, 1.0, self.alpha)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU
    r = relu(x)
    expected_r = np.array([0, 0, 0, 1, 2], dtype=np.float64)
    print(f"ReLU:            {'PASS' if np.allclose(r, expected_r) else 'FAIL'}")

    rd = relu_derivative(x)
    expected_rd = np.array([0, 0, 0, 1, 1], dtype=np.float64)
    print(f"ReLU deriv:      {'PASS' if np.allclose(rd, expected_rd) else 'FAIL'}")

    # Leaky ReLU
    lr = leaky_relu(x, alpha=0.01)
    expected_lr = np.array([-0.02, -0.01, 0, 1, 2])
    print(f"Leaky ReLU:      {'PASS' if np.allclose(lr, expected_lr) else 'FAIL'}")

    lrd = leaky_relu_derivative(x, alpha=0.01)
    expected_lrd = np.array([0.01, 0.01, 0.01, 1, 1])
    print(f"Leaky ReLU deriv:{'PASS' if np.allclose(lrd, expected_lrd) else 'FAIL'}")

    # ELU
    e = elu(x, alpha=1.0)
    print(f"ELU(0)=0:        {'PASS' if np.isclose(e[2], 0) else 'FAIL'}")
    print(f"ELU(1)=1:        {'PASS' if np.isclose(e[3], 1) else 'FAIL'}")
    print(f"ELU(-2)<0:       {'PASS' if e[0] < 0 and e[0] > -1 else 'FAIL'}")

    # GELU
    g = gelu(x)
    print(f"GELU(0)=0:       {'PASS' if np.isclose(g[2], 0, atol=1e-6) else 'FAIL'}")
    print(f"GELU(2)~1.95:    {'PASS' if np.isclose(g[4], 1.9545, atol=0.01) else 'FAIL'}")

    # Numerical gradient check for GELU
    eps = 1e-5
    x_test = np.array([0.5, -0.5, 1.0, -1.0])
    analytical = gelu_derivative(x_test)
    numerical = (gelu(x_test + eps) - gelu(x_test - eps)) / (2 * eps)
    print(f"GELU grad check: {'PASS' if np.allclose(analytical, numerical, atol=1e-4) else 'FAIL'}")

    # Layer test
    layer = ReLULayer()
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(x))
    print(f"ReLU layer fwd:  {'PASS' if np.allclose(out, expected_r) else 'FAIL'}")
    print(f"ReLU layer bwd:  {'PASS' if np.allclose(dx, expected_rd) else 'FAIL'}")
