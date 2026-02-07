"""
Problem: 2D Convolution Operation
Category: Computer Vision
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Sliding Window Convolution
# Time Complexity: O(H * W * kH * kW)
# Space Complexity: O(H * W)
# ============================================================

def conv2d(image: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """
    2D convolution (cross-correlation) of image with kernel.

    Args:
        image: 2D array (H, W) or 3D array (H, W, C)
        kernel: 2D array (kH, kW)
        mode: 'valid' (no padding) or 'same' (zero-pad to preserve size)

    Returns:
        Convolved image
    """
    # Handle multi-channel
    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(conv2d(image[:, :, c], kernel, mode))
        return np.stack(channels, axis=-1)

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    H, W = image.shape
    kH, kW = kernel.shape

    if mode == 'same':
        pad_h = kH // 2
        pad_w = kW // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        H, W = image.shape

    out_h = H - kH + 1
    out_w = W - kW + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)

    return output


# ============================================================
# Common Kernels
# ============================================================

def sobel_x() -> np.ndarray:
    """Sobel kernel for horizontal edge detection."""
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def sobel_y() -> np.ndarray:
    """Sobel kernel for vertical edge detection."""
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def gaussian_blur(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur kernel."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def sharpen_kernel() -> np.ndarray:
    """Sharpening kernel."""
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)


# ============================================================
# Approach 2: Vectorized with im2col
# Time Complexity: O(H * W * kH * kW)
# Space Complexity: O(H * W * kH * kW)
# ============================================================

def im2col(image: np.ndarray, kH: int, kW: int) -> np.ndarray:
    """Extract image patches as columns."""
    H, W = image.shape
    out_h = H - kH + 1
    out_w = W - kW + 1

    cols = np.zeros((kH * kW, out_h * out_w), dtype=np.float64)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            cols[:, idx] = image[i:i+kH, j:j+kW].ravel()
            idx += 1
    return cols


def conv2d_im2col(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution using im2col for valid mode."""
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    H, W = image.shape
    kH, kW = kernel.shape
    out_h = H - kH + 1
    out_w = W - kW + 1

    cols = im2col(image, kH, kW)
    kernel_flat = kernel.ravel()
    result = kernel_flat @ cols
    return result.reshape(out_h, out_w)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    kernel1 = sharpen_kernel()
    result1 = conv2d(img1, kernel1, mode='valid')
    print(f"Test 1 (sharpen):  {'PASS' if result1.shape == (1, 1) and result1[0, 0] == 5 else 'FAIL'} ({result1})")

    # Test Example 2
    img2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    kernel2 = np.array([[1, 0], [0, -1]], dtype=np.float64)
    result2 = conv2d(img2, kernel2, mode='valid')
    expected2 = np.array([[-5, -5, -5], [-5, -5, -5]])
    print(f"Test 2:            {'PASS' if np.allclose(result2, expected2) else 'FAIL'}")

    # Test: Same mode preserves size
    result_same = conv2d(img1, np.ones((3, 3))/9, mode='same')
    print(f"Same mode shape:   {'PASS' if result_same.shape == img1.shape else 'FAIL'} ({result_same.shape})")

    # Test: im2col matches
    result_im2col = conv2d_im2col(img1, kernel1)
    result_normal = conv2d(img1, kernel1, mode='valid')
    print(f"im2col matches:    {'PASS' if np.allclose(result_im2col, result_normal) else 'FAIL'}")

    # Test: Gaussian blur
    gb = gaussian_blur(3, 1.0)
    print(f"Gauss kernel sum:  {'PASS' if abs(np.sum(gb) - 1.0) < 1e-10 else 'FAIL'}")

    # Test: Multi-channel
    img_rgb = np.random.RandomState(42).randint(0, 256, (5, 5, 3)).astype(np.float64)
    result_rgb = conv2d(img_rgb, sobel_x(), mode='valid')
    print(f"RGB conv shape:    {'PASS' if result_rgb.shape == (3, 3, 3) else 'FAIL'} ({result_rgb.shape})")
