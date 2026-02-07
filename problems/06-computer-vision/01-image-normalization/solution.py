"""
Problem: Image Normalization
Category: Computer Vision
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Min-Max Normalization
# Time Complexity: O(H * W * C)
# Space Complexity: O(H * W * C)
# ============================================================

def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """
    Scale pixel values to [0, 1] range.

    Args:
        image: Input image (H, W) or (H, W, C)

    Returns:
        Normalized image with values in [0, 1]
    """
    image = image.astype(np.float64)
    img_min = np.min(image)
    img_max = np.max(image)

    if img_max - img_min < 1e-12:
        return np.zeros_like(image)

    return (image - img_min) / (img_max - img_min)


# ============================================================
# Approach 2: Mean Subtraction (Per-Channel)
# Time Complexity: O(H * W * C)
# Space Complexity: O(H * W * C)
# ============================================================

def mean_subtraction(image: np.ndarray) -> np.ndarray:
    """
    Subtract per-channel mean from the image.

    Args:
        image: Input image (H, W) or (H, W, C)

    Returns:
        Mean-centered image
    """
    image = image.astype(np.float64)

    if image.ndim == 2:
        return image - np.mean(image)
    elif image.ndim == 3:
        result = image.copy()
        for c in range(image.shape[2]):
            result[:, :, c] -= np.mean(image[:, :, c])
        return result
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


# ============================================================
# Approach 3: Standardization (Per-Channel)
# Time Complexity: O(H * W * C)
# Space Complexity: O(H * W * C)
# ============================================================

def standardize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize image to zero mean and unit variance per channel.

    Args:
        image: Input image (H, W) or (H, W, C)
        eps: Small constant to avoid division by zero

    Returns:
        Standardized image
    """
    image = image.astype(np.float64)

    if image.ndim == 2:
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + eps)
    elif image.ndim == 3:
        result = image.copy()
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            result[:, :, c] = (channel - mean) / (std + eps)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Grayscale
    img1 = np.array([[0, 128], [64, 255]], dtype=np.uint8)

    norm1 = minmax_normalize(img1)
    print(f"Min-Max range [0,1]: {'PASS' if norm1.min() >= 0 and norm1.max() <= 1.0 else 'FAIL'}")
    print(f"Min=0, Max=1:        {'PASS' if abs(norm1.min()) < 1e-10 and abs(norm1.max() - 1.0) < 1e-10 else 'FAIL'}")

    # Test mean subtraction
    centered = mean_subtraction(img1)
    print(f"Mean ≈ 0:            {'PASS' if abs(np.mean(centered)) < 1e-10 else 'FAIL'}")

    # Test standardization
    std_img = standardize(img1)
    print(f"Std mean ≈ 0:        {'PASS' if abs(np.mean(std_img)) < 1e-6 else 'FAIL'}")
    print(f"Std std ≈ 1:         {'PASS' if abs(np.std(std_img) - 1.0) < 1e-6 else 'FAIL'}")

    # Test Example 2: RGB image
    np.random.seed(42)
    img_rgb = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)

    norm_rgb = minmax_normalize(img_rgb)
    print(f"\nRGB Min-Max shape:   {'PASS' if norm_rgb.shape == (4, 4, 3) else 'FAIL'}")

    centered_rgb = mean_subtraction(img_rgb)
    # Per-channel mean should be ~0
    for c in range(3):
        ch_mean = np.mean(centered_rgb[:, :, c])
        print(f"Channel {c} mean ≈ 0:  {'PASS' if abs(ch_mean) < 1e-10 else 'FAIL'}")

    std_rgb = standardize(img_rgb)
    for c in range(3):
        ch_std = np.std(std_rgb[:, :, c])
        print(f"Channel {c} std ≈ 1:   {'PASS' if abs(ch_std - 1.0) < 1e-4 else 'FAIL'}")

    # Test: All zeros image
    zeros = np.zeros((3, 3), dtype=np.uint8)
    norm_zeros = minmax_normalize(zeros)
    print(f"\nZeros handled:       {'PASS' if np.allclose(norm_zeros, 0) else 'FAIL'}")
