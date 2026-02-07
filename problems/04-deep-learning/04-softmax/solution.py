"""
Problem: Softmax Function
Category: Deep Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Numerically Stable Softmax
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for a 1D vector.

    Args:
        z: Logits (n,)

    Returns:
        Probability distribution (n,) summing to 1.0
    """
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


def softmax_batch(Z: np.ndarray) -> np.ndarray:
    """
    Softmax applied row-wise to a 2D array.

    Args:
        Z: Logits (n_samples, n_classes)

    Returns:
        Probabilities (n_samples, n_classes)
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ============================================================
# Approach 2: Softmax Jacobian
# Time Complexity: O(n^2)
# Space Complexity: O(n^2)
# ============================================================

def softmax_jacobian(s: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix of softmax output.

    J[i,j] = s[i] * (delta_ij - s[j])
            = diag(s) - outer(s, s)

    Args:
        s: Softmax output vector (n,)

    Returns:
        Jacobian matrix (n, n)
    """
    return np.diag(s) - np.outer(s, s)


def softmax_vjp(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vector-Jacobian product: v @ J without forming full Jacobian.
    Efficient for backpropagation.

    Args:
        s: Softmax output (n,)
        v: Upstream gradient (n,)

    Returns:
        Gradient w.r.t. logits (n,)
    """
    # v @ (diag(s) - outer(s, s)) = v * s - s * (v @ s)
    return s * v - s * np.dot(v, s)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test basic softmax
    z = np.array([2.0, 1.0, 0.1])
    s = softmax(z)
    print(f"Softmax:         {np.round(s, 4)}")
    print(f"Sum=1:           {'PASS' if np.isclose(np.sum(s), 1.0) else 'FAIL'}")
    print(f"All positive:    {'PASS' if np.all(s > 0) else 'FAIL'}")

    # Test numerical stability
    z_large = np.array([1000.0, 1000.0, 1000.0])
    s_large = softmax(z_large)
    print(f"Large logits:    {'PASS' if np.allclose(s_large, 1/3) else 'FAIL'}")
    print(f"No NaN:          {'PASS' if np.all(np.isfinite(s_large)) else 'FAIL'}")

    # Test batch softmax
    Z = np.array([[1.0, 2.0, 3.0],
                  [1000.0, 1000.0, 1000.0]])
    S = softmax_batch(Z)
    print(f"Batch rows sum:  {'PASS' if np.allclose(S.sum(axis=1), 1.0) else 'FAIL'}")

    # Test Jacobian
    s = softmax(z)
    J = softmax_jacobian(s)
    print(f"Jacobian shape:  {'PASS' if J.shape == (3, 3) else 'FAIL'}")
    print(f"Row sums ~0:     {'PASS' if np.allclose(J.sum(axis=1), 0, atol=1e-10) else 'FAIL'}")

    # Numerical gradient check for Jacobian
    eps = 1e-5
    J_num = np.zeros((3, 3))
    for j in range(3):
        z_plus = z.copy()
        z_plus[j] += eps
        z_minus = z.copy()
        z_minus[j] -= eps
        J_num[:, j] = (softmax(z_plus) - softmax(z_minus)) / (2 * eps)
    print(f"Jacobian check:  {'PASS' if np.allclose(J, J_num, atol=1e-5) else 'FAIL'}")

    # Test VJP
    v = np.array([1.0, 0.0, 0.0])
    vjp_result = softmax_vjp(s, v)
    vjp_full = v @ J
    print(f"VJP matches:     {'PASS' if np.allclose(vjp_result, vjp_full) else 'FAIL'}")
