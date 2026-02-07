"""
Problem: Padding and Stride
Category: Computer Vision
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Union, Tuple


# ============================================================
# Approach 1: Zero Padding + Strided Convolution
# Time Complexity: O(out_H * out_W * kH * kW)
# Space Complexity: O(out_H * out_W)
# ============================================================

def zero_pad(image: np.ndarray, pad: int) -> np.ndarray:
    """
    Add zero padding around a 2D image.

    Args:
        image: Input (H, W)
        pad: Number of zero rows/cols to add on each side

    Returns:
        Padded image (H + 2*pad, W + 2*pad)
    """
    if pad == 0:
        return image.copy()
    return np.pad(image.astype(np.float64), ((pad, pad), (pad, pad)),
                  mode='constant', constant_values=0)


def compute_output_size(input_size: int, kernel_size: int, padding: int,
                        stride: int) -> int:
    """Compute convolution output dimension."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def conv2d_padded_strided(image: np.ndarray, kernel: np.ndarray,
                           padding: int = 0, stride: int = 1) -> np.ndarray:
    """
    2D convolution with padding and stride.

    Args:
        image: Input (H, W)
        kernel: Filter (kH, kW)
        padding: Zero-padding amount
        stride: Step size

    Returns:
        Output feature map
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Apply padding
    if padding > 0:
        image = zero_pad(image, padding)

    H, W = image.shape
    kH, kW = kernel.shape

    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            row = i * stride
            col = j * stride
            patch = image[row:row+kH, col:col+kW]
            output[i, j] = np.sum(patch * kernel)

    return output


# ============================================================
# Approach 2: "Same" Padding (Auto-Calculate)
# ============================================================

def compute_same_padding(input_size: int, kernel_size: int, stride: int = 1) -> int:
    """
    Compute padding needed for 'same' output size.
    For stride=1: pad = (kernel_size - 1) / 2

    Args:
        input_size: Input dimension
        kernel_size: Kernel dimension
        stride: Stride value

    Returns:
        Padding amount
    """
    if stride == 1:
        return (kernel_size - 1) // 2
    # For stride > 1, "same" means output = ceil(input / stride)
    out_size = (input_size + stride - 1) // stride
    pad_needed = max(0, (out_size - 1) * stride + kernel_size - input_size)
    return pad_needed // 2


def conv2d_same(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Convolution with 'same' padding (preserves spatial dimensions for stride=1).

    Args:
        image: Input (H, W)
        kernel: Filter (kH, kW)
        stride: Step size

    Returns:
        Output with same spatial size as input (when stride=1)
    """
    H, W = image.shape
    kH, kW = kernel.shape

    pad_h = compute_same_padding(H, kH, stride)
    pad_w = compute_same_padding(W, kW, stride)

    # Asymmetric padding if needed
    image_padded = np.pad(image.astype(np.float64),
                          ((pad_h, pad_h), (pad_w, pad_w)),
                          mode='constant')

    pH, pW = image_padded.shape
    out_h = (pH - kH) // stride + 1
    out_w = (pW - kW) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            row, col = i * stride, j * stride
            output[i, j] = np.sum(image_padded[row:row+kH, col:col+kW] * kernel)

    return output


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    kernel = np.array([[1, 0], [0, -1]], dtype=np.float64)

    # Test Example 1: Padding=1, Stride=1
    result1 = conv2d_padded_strided(img, kernel, padding=1, stride=1)
    expected_shape1 = (4, 4)
    print(f"Pad=1 Stride=1 shape: {'PASS' if result1.shape == expected_shape1 else 'FAIL'} ({result1.shape})")

    # Test output size formula
    out_h = compute_output_size(3, 2, 1, 1)
    print(f"Output size formula:  {'PASS' if out_h == 4 else 'FAIL'} ({out_h})")

    # Test Example 2: 6x6, kernel 3x3, stride=2
    img2 = np.arange(36, dtype=np.float64).reshape(6, 6)
    kernel2 = np.ones((3, 3)) / 9
    result2 = conv2d_padded_strided(img2, kernel2, padding=0, stride=2)
    expected_shape2 = (2, 2)
    print(f"6x6 stride=2 shape:   {'PASS' if result2.shape == expected_shape2 else 'FAIL'} ({result2.shape})")

    # Test: Same padding preserves size (stride=1)
    result_same = conv2d_same(img, np.ones((3, 3))/9, stride=1)
    print(f"Same padding shape:   {'PASS' if result_same.shape == img.shape else 'FAIL'} ({result_same.shape})")

    # Test: Zero padding
    padded = zero_pad(img, 1)
    print(f"Padding shape:        {'PASS' if padded.shape == (5, 5) else 'FAIL'} ({padded.shape})")
    print(f"Corners are zero:     {'PASS' if padded[0, 0] == 0 and padded[4, 4] == 0 else 'FAIL'}")

    # Test: No padding, stride=1 = valid convolution
    result_valid = conv2d_padded_strided(img, kernel, padding=0, stride=1)
    print(f"Valid conv shape:     {'PASS' if result_valid.shape == (2, 2) else 'FAIL'} ({result_valid.shape})")
