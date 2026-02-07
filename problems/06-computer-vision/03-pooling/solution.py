"""
Problem: Max/Average Pooling
Category: Computer Vision
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# Approach 1: Sliding Window Pooling
# Time Complexity: O(H * W)
# Space Complexity: O(out_H * out_W)
# ============================================================

def max_pool2d(image: np.ndarray, pool_size: Tuple[int, int] = (2, 2),
               stride: Optional[int] = None) -> np.ndarray:
    """
    Max pooling over 2D image.

    Args:
        image: Input array (H, W) or (H, W, C)
        pool_size: (pool_h, pool_w)
        stride: Step size (defaults to pool_size[0])

    Returns:
        Pooled output
    """
    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(max_pool2d(image[:, :, c], pool_size, stride))
        return np.stack(channels, axis=-1)

    image = image.astype(np.float64)
    H, W = image.shape
    pH, pW = pool_size
    s = stride if stride is not None else pH

    out_h = (H - pH) // s + 1
    out_w = (W - pW) // s + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            window = image[i*s:i*s+pH, j*s:j*s+pW]
            output[i, j] = np.max(window)

    return output


def avg_pool2d(image: np.ndarray, pool_size: Tuple[int, int] = (2, 2),
               stride: Optional[int] = None) -> np.ndarray:
    """
    Average pooling over 2D image.

    Args:
        image: Input array (H, W) or (H, W, C)
        pool_size: (pool_h, pool_w)
        stride: Step size

    Returns:
        Pooled output
    """
    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(avg_pool2d(image[:, :, c], pool_size, stride))
        return np.stack(channels, axis=-1)

    image = image.astype(np.float64)
    H, W = image.shape
    pH, pW = pool_size
    s = stride if stride is not None else pH

    out_h = (H - pH) // s + 1
    out_w = (W - pW) // s + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            window = image[i*s:i*s+pH, j*s:j*s+pW]
            output[i, j] = np.mean(window)

    return output


def global_avg_pool(image: np.ndarray) -> np.ndarray:
    """
    Global average pooling: average across all spatial dimensions.

    Args:
        image: Input array (H, W) or (H, W, C)

    Returns:
        Scalar (for 2D) or 1D array of length C (for 3D)
    """
    image = image.astype(np.float64)
    if image.ndim == 2:
        return np.mean(image)
    elif image.ndim == 3:
        return np.mean(image, axis=(0, 1))
    raise ValueError(f"Expected 2D or 3D, got {image.ndim}D")


# ============================================================
# Approach 2: Reshape-based (Non-overlapping Only)
# Time Complexity: O(H * W)
# Space Complexity: O(out_H * out_W)
# ============================================================

def max_pool2d_reshape(image: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    Max pooling using reshape trick (non-overlapping, square pools only).

    Args:
        image: (H, W) where H and W are divisible by pool_size

    Returns:
        Pooled output
    """
    image = image.astype(np.float64)
    H, W = image.shape
    p = pool_size

    # Reshape to (out_h, p, out_w, p) then max over pool dims
    reshaped = image.reshape(H // p, p, W // p, p)
    return reshaped.max(axis=(1, 3))


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Max pooling
    img = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]], dtype=np.float64)

    max_result = max_pool2d(img, (2, 2), stride=2)
    expected_max = np.array([[6, 8], [14, 16]], dtype=np.float64)
    print(f"Max Pool:          {'PASS' if np.allclose(max_result, expected_max) else 'FAIL'}")

    # Test Example 2: Average pooling
    avg_result = avg_pool2d(img, (2, 2), stride=2)
    expected_avg = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=np.float64)
    print(f"Avg Pool:          {'PASS' if np.allclose(avg_result, expected_avg) else 'FAIL'}")

    # Test: Reshape-based matches
    max_reshape = max_pool2d_reshape(img, 2)
    print(f"Reshape matches:   {'PASS' if np.allclose(max_reshape, expected_max) else 'FAIL'}")

    # Test: Global average pooling
    gap = global_avg_pool(img)
    print(f"Global Avg Pool:   {'PASS' if abs(gap - 8.5) < 1e-10 else 'FAIL'} ({gap})")

    # Test: Multi-channel
    img_3ch = np.stack([img, img * 2, img * 3], axis=-1)
    max_3ch = max_pool2d(img_3ch, (2, 2), stride=2)
    print(f"3-channel shape:   {'PASS' if max_3ch.shape == (2, 2, 3) else 'FAIL'} ({max_3ch.shape})")
    print(f"Channel 0 correct: {'PASS' if np.allclose(max_3ch[:,:,0], expected_max) else 'FAIL'}")

    # Test: Stride 1 (overlapping)
    max_s1 = max_pool2d(img, (2, 2), stride=1)
    print(f"Stride 1 shape:    {'PASS' if max_s1.shape == (3, 3) else 'FAIL'} ({max_s1.shape})")

    # Test: Global avg pool multi-channel
    gap_3ch = global_avg_pool(img_3ch)
    print(f"GAP 3ch shape:     {'PASS' if gap_3ch.shape == (3,) else 'FAIL'}")
