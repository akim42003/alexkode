"""
Problem: Implement a CNN Layer
Category: Computer Vision
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Dict


# ============================================================
# Approach 1: Loop-Based CNN Layer with Forward and Backward
# Time Complexity: O(batch * C_out * C_in * H_out * W_out * kH * kW)
# Space Complexity: O(batch * C_out * H_out * W_out)
# ============================================================

class Conv2DLayer:
    """Full convolutional layer with forward and backward pass."""

    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (C_in * kernel_size * kernel_size))

        self.W = rng.randn(C_out, C_in, kernel_size, kernel_size) * scale
        self.b = np.zeros(C_out, dtype=np.float64)
        self.stride = stride
        self.padding = padding

        # Cache for backward
        self.cache = None

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Apply zero padding to batch of images."""
        if self.padding == 0:
            return X
        p = self.padding
        return np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)),
                      mode='constant', constant_values=0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of convolution.

        Args:
            X: Input (batch, C_in, H, W)

        Returns:
            Output (batch, C_out, H_out, W_out)
        """
        X = X.astype(np.float64)
        batch, C_in, H, W_ = X.shape
        C_out, _, kH, kW = self.W.shape
        s = self.stride

        X_pad = self._pad(X)
        H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]

        H_out = (H_pad - kH) // s + 1
        W_out = (W_pad - kW) // s + 1

        output = np.zeros((batch, C_out, H_out, W_out), dtype=np.float64)

        for n in range(batch):
            for f in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        row, col = i * s, j * s
                        patch = X_pad[n, :, row:row+kH, col:col+kW]
                        output[n, f, i, j] = np.sum(patch * self.W[f]) + self.b[f]

        self.cache = (X, X_pad)
        return output

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass: compute gradients.

        Args:
            dout: Gradient of loss w.r.t. output (batch, C_out, H_out, W_out)

        Returns:
            Tuple of (dX, dW, db)
        """
        X, X_pad = self.cache
        batch, C_in, H, W_ = X.shape
        C_out, _, kH, kW = self.W.shape
        _, _, H_out, W_out = dout.shape
        s = self.stride

        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for n in range(batch):
            for f in range(C_out):
                db[f] += np.sum(dout[n, f])
                for i in range(H_out):
                    for j in range(W_out):
                        row, col = i * s, j * s
                        patch = X_pad[n, :, row:row+kH, col:col+kW]
                        dW[f] += dout[n, f, i, j] * patch
                        dX_pad[n, :, row:row+kH, col:col+kW] += dout[n, f, i, j] * self.W[f]

        # Remove padding from gradient
        if self.padding > 0:
            p = self.padding
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad

        return dX, dW, db


# ============================================================
# Approach 2: Functional Interface
# ============================================================

def conv2d_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                   stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Functional conv2d forward pass.

    Args:
        X: (batch, C_in, H, W)
        W: (C_out, C_in, kH, kW)
        b: (C_out,)
        stride: Stride
        padding: Zero-padding

    Returns:
        (batch, C_out, H_out, W_out)
    """
    layer = Conv2DLayer.__new__(Conv2DLayer)
    layer.W = W
    layer.b = b
    layer.stride = stride
    layer.padding = padding
    return layer.forward(X)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Example 1: Simple case
    X1 = np.random.randn(1, 1, 4, 4)
    conv1 = Conv2DLayer(C_in=1, C_out=2, kernel_size=3, stride=1, padding=0, seed=42)
    out1 = conv1.forward(X1)
    print(f"Test 1 shape:    {'PASS' if out1.shape == (1, 2, 2, 2) else 'FAIL'} ({out1.shape})")

    # Test backward
    dout1 = np.ones_like(out1)
    dX1, dW1, db1 = conv1.backward(dout1)
    print(f"dX shape:        {'PASS' if dX1.shape == X1.shape else 'FAIL'} ({dX1.shape})")
    print(f"dW shape:        {'PASS' if dW1.shape == conv1.W.shape else 'FAIL'} ({dW1.shape})")
    print(f"db shape:        {'PASS' if db1.shape == conv1.b.shape else 'FAIL'} ({db1.shape})")

    # Test Example 2: Batched, multi-channel with stride and padding
    X2 = np.random.randn(2, 3, 6, 6)
    conv2 = Conv2DLayer(C_in=3, C_out=8, kernel_size=3, stride=2, padding=1, seed=42)
    out2 = conv2.forward(X2)
    expected_shape = (2, 8, 3, 3)
    print(f"Test 2 shape:    {'PASS' if out2.shape == expected_shape else 'FAIL'} ({out2.shape})")

    # Gradient check (numerical)
    eps = 1e-5
    X_check = np.random.randn(1, 1, 4, 4)
    conv_check = Conv2DLayer(C_in=1, C_out=1, kernel_size=3, stride=1, padding=0, seed=0)

    out_check = conv_check.forward(X_check)
    dout_check = np.ones_like(out_check)
    dX_ana, _, _ = conv_check.backward(dout_check)

    # Numerical gradient for one element
    i, j = 1, 1
    X_plus = X_check.copy()
    X_plus[0, 0, i, j] += eps
    conv_check.forward(X_plus)
    out_plus = conv_check.forward(X_plus)

    X_minus = X_check.copy()
    X_minus[0, 0, i, j] -= eps
    out_minus = conv_check.forward(X_minus)

    dX_num = np.sum(out_plus - out_minus) / (2 * eps)
    print(f"Grad check:      {'PASS' if abs(dX_ana[0, 0, i, j] - dX_num) < 1e-4 else 'FAIL'} (ana={dX_ana[0,0,i,j]:.6f}, num={dX_num:.6f})")
