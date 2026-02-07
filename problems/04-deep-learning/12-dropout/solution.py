"""
Problem: Dropout
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================
# Approach 1: Dropout Class
# Time Complexity: O(n)
# Space Complexity: O(n) for mask
# ============================================================

class Dropout:
    """Inverted dropout layer."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of DROPPING a neuron (0 <= p < 1)
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input array of any shape
            training: If True, apply dropout; else pass through

        Returns:
            Output with dropout applied (training) or identity (inference)
        """
        if not training or self.p == 0:
            return x.copy()

        if self.p >= 1.0:
            self.mask = np.zeros_like(x)
            return np.zeros_like(x)

        self.mask = (np.random.rand(*x.shape) >= self.p).astype(x.dtype)
        return x * self.mask / (1.0 - self.p)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass: same mask and scaling."""
        if self.p == 0:
            return dout.copy()
        if self.p >= 1.0:
            return np.zeros_like(dout)
        return dout * self.mask / (1.0 - self.p)


# ============================================================
# Approach 2: Functional Interface
# ============================================================

def dropout_forward(x: np.ndarray, p: float = 0.5,
                    training: bool = True,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, Optional[Tuple]]:
    """Functional dropout forward."""
    if not training or p == 0:
        return x.copy(), None

    if seed is not None:
        np.random.seed(seed)

    if p >= 1.0:
        return np.zeros_like(x), (np.zeros_like(x), p)

    mask = (np.random.rand(*x.shape) >= p).astype(x.dtype)
    out = x * mask / (1.0 - p)
    return out, (mask, p)


def dropout_backward(dout: np.ndarray, cache: Optional[Tuple]) -> np.ndarray:
    """Functional dropout backward."""
    if cache is None:
        return dout.copy()
    mask, p = cache
    if p >= 1.0:
        return np.zeros_like(dout)
    return dout * mask / (1.0 - p)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Training mode
    np.random.seed(42)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    drop = Dropout(p=0.5)
    out = drop.forward(x, training=True)
    print(f"Zeros where dropped: {'PASS' if np.all(out[drop.mask == 0] == 0) else 'FAIL'}")
    scaled = np.allclose(out[drop.mask == 1], x[drop.mask == 1] / 0.5)
    print(f"Scaled survivors:    {'PASS' if scaled else 'FAIL'}")

    # Test 2: Inference mode
    out_test = drop.forward(x, training=False)
    print(f"Inference identity:  {'PASS' if np.allclose(out_test, x) else 'FAIL'}")

    # Test 3: Expected value preserved
    np.random.seed(0)
    x_large = np.random.randn(1000, 500)
    drop3 = Dropout(p=0.3)
    outputs = [drop3.forward(x_large, training=True) for _ in range(50)]
    mean_out = np.mean(outputs, axis=0)
    rel_diff = abs(np.mean(mean_out) - np.mean(x_large)) / (abs(np.mean(x_large)) + 1e-8)
    print(f"E[out] ≈ E[x]:      {'PASS' if rel_diff < 0.1 else 'FAIL'} (rel diff: {rel_diff:.4f})")

    # Test 4: Backward
    np.random.seed(123)
    drop4 = Dropout(p=0.5)
    out4 = drop4.forward(x, training=True)
    dx = drop4.backward(np.ones_like(x))
    print(f"Grad 0 for dropped: {'PASS' if np.all(dx[drop4.mask == 0] == 0) else 'FAIL'}")
    print(f"Grad scaled for kept:{'PASS' if np.allclose(dx[drop4.mask == 1], 2.0) else 'FAIL'}")

    # Test 5: p=0 (no dropout)
    drop5 = Dropout(p=0.0)
    out5 = drop5.forward(x, training=True)
    print(f"p=0 identity:       {'PASS' if np.allclose(out5, x) else 'FAIL'}")

    # Test 6: p=1 (drop all)
    drop6 = Dropout(p=1.0)
    out6 = drop6.forward(x, training=True)
    print(f"p=1 all zeros:      {'PASS' if np.allclose(out6, 0) else 'FAIL'}")

    # Test 7: Functional API
    out_f, cache = dropout_forward(x, p=0.4, training=True, seed=99)
    dx_f = dropout_backward(np.ones_like(x), cache)
    print(f"Functional works:   {'PASS' if out_f.shape == x.shape else 'FAIL'}")
