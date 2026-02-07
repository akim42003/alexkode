"""
Problem: Batch Normalization
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# Approach 1: BatchNorm Class
# Time Complexity: O(N * D)
# Space Complexity: O(N * D) for cache
# ============================================================

class BatchNorm:
    """Batch Normalization layer with forward/backward."""

    def __init__(self, D: int, epsilon: float = 1e-5, momentum: float = 0.9):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(D)
        self.running_var = np.ones(D)
        self.cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input (N, D)
            training: If True, use batch stats; else use running stats

        Returns:
            Normalized output (N, D)
        """
        if training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_hat = (x - mu) / np.sqrt(var + self.epsilon)
            out = self.gamma * x_hat + self.beta

            self.running_mean = (self.momentum * self.running_mean
                                 + (1 - self.momentum) * mu)
            self.running_var = (self.momentum * self.running_var
                                + (1 - self.momentum) * var)
            self.cache = (x, x_hat, mu, var)
        else:
            x_hat = ((x - self.running_mean)
                     / np.sqrt(self.running_var + self.epsilon))
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass.

        Args:
            dout: Upstream gradient (N, D)

        Returns:
            Tuple of (dx, dgamma, dbeta)
        """
        x, x_hat, mu, var = self.cache
        N = dout.shape[0]

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * x_hat, axis=0)

        dx_hat = dout * self.gamma
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * inv_std**3, axis=0)
        dmu = (np.sum(dx_hat * (-inv_std), axis=0)
               + dvar * np.mean(-2.0 * (x - mu), axis=0))
        dx = dx_hat * inv_std + dvar * 2.0 * (x - mu) / N + dmu / N

        return dx, dgamma, dbeta


# ============================================================
# Approach 2: Functional Interface
# ============================================================

def batchnorm_forward(x, gamma, beta, running_mean, running_var,
                      training=True, epsilon=1e-5, momentum=0.9):
    """Functional batchnorm forward."""
    if training:
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_hat = (x - mu) / np.sqrt(var + epsilon)
        out = gamma * x_hat + beta
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x, x_hat, mu, var, gamma, epsilon)
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + epsilon)
        out = gamma * x_hat + beta
        cache = None
    return out, cache, running_mean, running_var


def batchnorm_backward(dout, cache):
    """Functional batchnorm backward."""
    x, x_hat, mu, var, gamma, epsilon = cache
    N = dout.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    inv_std = 1.0 / np.sqrt(var + epsilon)
    dvar = np.sum(dx_hat * (x - mu) * (-0.5) * inv_std**3, axis=0)
    dmu = np.sum(dx_hat * (-inv_std), axis=0) + dvar * np.mean(-2.0 * (x - mu), axis=0)
    dx = dx_hat * inv_std + dvar * 2.0 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Forward normalizes to zero mean, unit std
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = BatchNorm(D=2)
    out = bn.forward(x, training=True)
    col_mean = np.mean(out, axis=0)
    col_std = np.std(out, axis=0)
    print(f"Mean ~0:          {'PASS' if np.allclose(col_mean, 0, atol=1e-6) else 'FAIL'}")
    print(f"Std  ~1:          {'PASS' if np.allclose(col_std, 1, atol=0.05) else 'FAIL'}")

    # Test 2: Backward
    dout = np.ones((3, 2))
    dx, dgamma, dbeta = bn.backward(dout)
    print(f"dbeta=[3,3]:      {'PASS' if np.allclose(dbeta, [3, 3]) else 'FAIL'}")
    print(f"dgamma=[0,0]:     {'PASS' if np.allclose(dgamma, 0, atol=1e-6) else 'FAIL'}")
    print(f"dx ~0:            {'PASS' if np.allclose(dx, 0, atol=1e-6) else 'FAIL'}")

    # Test 3: Numerical gradient check
    np.random.seed(42)
    N, D = 5, 4
    x = np.random.randn(N, D)
    dout = np.random.randn(N, D)
    bn2 = BatchNorm(D=D)
    bn2.gamma = np.random.randn(D)
    bn2.beta = np.random.randn(D)
    out = bn2.forward(x, training=True)
    dx_ana, _, _ = bn2.backward(dout)

    eps = 1e-5
    dx_num = np.zeros_like(x)
    for i in range(N):
        for j in range(D):
            x_p = x.copy(); x_p[i, j] += eps
            x_m = x.copy(); x_m[i, j] -= eps
            bn_p = BatchNorm(D=D)
            bn_p.gamma = bn2.gamma.copy()
            bn_p.beta = bn2.beta.copy()
            bn_m = BatchNorm(D=D)
            bn_m.gamma = bn2.gamma.copy()
            bn_m.beta = bn2.beta.copy()
            out_p = bn_p.forward(x_p, training=True)
            out_m = bn_m.forward(x_m, training=True)
            dx_num[i, j] = np.sum((out_p - out_m) * dout) / (2 * eps)

    rel_err = np.max(np.abs(dx_ana - dx_num) /
                     np.maximum(np.abs(dx_ana) + np.abs(dx_num), 1e-8))
    print(f"Gradient check:   {'PASS' if rel_err < 1e-4 else 'FAIL'} (rel err: {rel_err:.2e})")

    # Test 4: Inference mode uses running stats
    bn3 = BatchNorm(D=2)
    for _ in range(100):
        batch = np.random.randn(32, 2) * 3 + 5
        bn3.forward(batch, training=True)
    out_test = bn3.forward(np.array([[5.0, 5.0]]), training=False)
    print(f"Inference works:  {'PASS' if out_test.shape == (1, 2) else 'FAIL'}")
