"""
Problem: Adam Optimizer
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import List


# ============================================================
# Approach 1: Standard Adam
# Time Complexity: O(d) per step
# Space Complexity: O(2d) for first and second moments
# ============================================================

class Adam:
    """Adam optimizer with bias correction."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Adam parameter update.

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            params[i] = params[i] - self.lr * m_hat / (
                np.sqrt(v_hat) + self.epsilon)

        return params


# ============================================================
# Approach 2: AdamW (Adam with Weight Decay)
# ============================================================

class AdamW:
    """Adam with decoupled weight decay regularization."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        """AdamW update with decoupled weight decay."""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            # Weight decay (applied directly to params, not through gradient)
            params[i] = params[i] * (1 - self.lr * self.weight_decay)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            params[i] = params[i] - self.lr * m_hat / (
                np.sqrt(v_hat) + self.epsilon)

        return params


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Optimize f(x) = x^2
    print("=== Adam on f(x) = x^2 ===")
    x = np.array([5.0])
    adam = Adam(lr=0.1)
    for step in range(200):
        grad = 2 * x
        [x] = adam.step([x], [grad])
    print(f"x = {x[0]:.6f}: {'PASS' if abs(x[0]) < 0.01 else 'FAIL'}")

    # Test 2: Multi-dimensional
    print("\n=== Adam on f(x,y) = x^2 + 100*y^2 ===")
    xy = np.array([5.0, 5.0])
    adam2 = Adam(lr=0.01)
    for step in range(500):
        g = np.array([2 * xy[0], 200 * xy[1]])
        [xy] = adam2.step([xy], [g])
    print(f"x={xy[0]:.4f}, y={xy[1]:.4f}")
    print(f"Converged: {'PASS' if np.linalg.norm(xy) < 0.1 else 'FAIL'}")

    # Test 3: Bias correction matters early on
    print("\n=== Bias correction test ===")
    x_bc = np.array([10.0])
    adam_bc = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    grad_bc = np.array([1.0])  # constant gradient
    [x_bc] = adam_bc.step([x_bc], [grad_bc])
    # Without bias correction, first step would be tiny
    # With bias correction, m_hat = 1.0/(1-0.9) = 1.0, step is meaningful
    print(f"First step change: {10.0 - x_bc[0]:.4f}")
    print(f"Non-trivial step:  {'PASS' if abs(10.0 - x_bc[0]) > 0.05 else 'FAIL'}")

    # Test 4: Multiple parameter groups
    print("\n=== Multiple parameters ===")
    p1 = np.array([3.0, 4.0])
    p2 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    adam3 = Adam(lr=0.01)
    g1, g2 = np.ones_like(p1), np.ones_like(p2)
    [p1, p2] = adam3.step([p1, p2], [g1, g2])
    print(f"Shapes preserved: {'PASS' if p1.shape == (2,) and p2.shape == (2, 2) else 'FAIL'}")

    # Test 5: AdamW
    print("\n=== AdamW ===")
    x_w = np.array([5.0])
    adamw = AdamW(lr=0.1, weight_decay=0.01)
    for step in range(200):
        grad = 2 * x_w
        [x_w] = adamw.step([x_w], [grad])
    print(f"AdamW x={x_w[0]:.6f}: {'PASS' if abs(x_w[0]) < 0.01 else 'FAIL'}")

    # Test 6: Timestep tracking
    adam4 = Adam(lr=0.01)
    x_t = np.array([1.0])
    for _ in range(10):
        [x_t] = adam4.step([x_t], [np.array([0.1])])
    print(f"Timestep=10:      {'PASS' if adam4.t == 10 else 'FAIL'}")
