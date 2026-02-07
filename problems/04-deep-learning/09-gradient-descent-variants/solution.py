"""
Problem: Gradient Descent Variants
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# Approach 1: SGD with Momentum
# Time Complexity: O(d) per step
# Space Complexity: O(d) for velocity
# ============================================================

class SGDMomentum:
    """Stochastic Gradient Descent with Momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Update parameters using momentum.

        v = beta * v - lr * grad
        param = param + v
        """
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.velocities[i] = (self.momentum * self.velocities[i]
                                  - self.lr * grads[i])
            params[i] = params[i] + self.velocities[i]

        return params


# ============================================================
# Approach 2: RMSProp
# Time Complexity: O(d) per step
# Space Complexity: O(d) for cache
# ============================================================

class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, lr: float = 0.001, decay: float = 0.99,
                 epsilon: float = 1e-8):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Update parameters using RMSProp.

        s = decay * s + (1-decay) * grad^2
        param -= lr * grad / (sqrt(s) + eps)
        """
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.cache[i] = (self.decay * self.cache[i]
                             + (1 - self.decay) * grads[i]**2)
            params[i] = params[i] - self.lr * grads[i] / (
                np.sqrt(self.cache[i]) + self.epsilon)

        return params


# ============================================================
# Approach 3: Mini-batch SGD
# ============================================================

def create_mini_batches(X: np.ndarray, y: np.ndarray,
                        batch_size: int,
                        seed: Optional[int] = None) -> List[Tuple]:
    """
    Split data into mini-batches.

    Args:
        X: Input data (N, d)
        y: Labels (N, ...)
        batch_size: Size of each batch
        seed: Random seed for shuffling

    Returns:
        List of (X_batch, y_batch) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    N = X.shape[0]
    indices = np.random.permutation(N)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    batches = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batches.append((X_shuffled[start:end], y_shuffled[start:end]))

    return batches


class VanillaSGD:
    """Vanilla SGD (for comparison)."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        for i in range(len(params)):
            params[i] = params[i] - self.lr * grads[i]
        return params


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test on f(x) = x^2, gradient = 2x
    print("=== Optimizing f(x) = x^2 ===")

    # Vanilla SGD
    x_vanilla = np.array([5.0])
    sgd = VanillaSGD(lr=0.1)
    for step in range(20):
        grad = 2 * x_vanilla
        [x_vanilla] = sgd.step([x_vanilla], [grad])
    print(f"Vanilla SGD x={x_vanilla[0]:.6f}: {'PASS' if abs(x_vanilla[0]) < 0.1 else 'FAIL'}")

    # SGD + Momentum
    x_mom = np.array([5.0])
    opt_mom = SGDMomentum(lr=0.01, momentum=0.9)
    for step in range(50):
        grad = 2 * x_mom
        [x_mom] = opt_mom.step([x_mom], [grad])
    print(f"SGD+Momentum x={x_mom[0]:.6f}: {'PASS' if abs(x_mom[0]) < 0.1 else 'FAIL'}")

    # RMSProp
    x_rms = np.array([5.0])
    opt_rms = RMSProp(lr=0.1, decay=0.99)
    for step in range(50):
        grad = 2 * x_rms
        [x_rms] = opt_rms.step([x_rms], [grad])
    print(f"RMSProp x={x_rms[0]:.6f}: {'PASS' if abs(x_rms[0]) < 0.1 else 'FAIL'}")

    # Test on 2D: f(x,y) = x^2 + 100*y^2 (ill-conditioned)
    print("\n=== Ill-conditioned f(x,y) = x^2 + 100*y^2 ===")

    xy_rms = np.array([5.0, 5.0])
    opt_ill = RMSProp(lr=0.1, decay=0.9)
    for step in range(200):
        g = np.array([2 * xy_rms[0], 200 * xy_rms[1]])
        [xy_rms] = opt_ill.step([xy_rms], [g])
    print(f"RMSProp 2D: x={xy_rms[0]:.4f}, y={xy_rms[1]:.4f}")
    print(f"Converged:  {'PASS' if np.linalg.norm(xy_rms) < 0.5 else 'FAIL'}")

    # Test mini-batches
    print("\n=== Mini-batch creation ===")
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    batches = create_mini_batches(X, y, batch_size=3, seed=42)
    print(f"Num batches:  {'PASS' if len(batches) == 4 else 'FAIL'} ({len(batches)})")
    total = sum(b[0].shape[0] for b in batches)
    print(f"Total samples: {'PASS' if total == 10 else 'FAIL'} ({total})")
    print(f"Last batch sz: {'PASS' if batches[-1][0].shape[0] == 1 else 'FAIL'} ({batches[-1][0].shape[0]})")

    # Test multiple params
    print("\n=== Multiple parameters ===")
    p1 = np.array([3.0, 4.0])
    p2 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    opt = SGDMomentum(lr=0.01, momentum=0.9)
    g1, g2 = np.ones_like(p1), np.ones_like(p2)
    [p1, p2] = opt.step([p1, p2], [g1, g2])
    print(f"Multi-param: {'PASS' if p1.shape == (2,) and p2.shape == (2, 2) else 'FAIL'}")
