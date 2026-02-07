"""
Problem: Data Augmentation
Category: Computer Vision
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# Approach 1: Flip Operations
# Time Complexity: O(H * W)
# Space Complexity: O(H * W)
# ============================================================

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Flip image horizontally (mirror left-right)."""
    return image[:, ::-1].copy()


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Flip image vertically (mirror top-bottom)."""
    return image[::-1, :].copy()


# ============================================================
# Approach 2: Rotation with Bilinear Interpolation
# Time Complexity: O(H * W)
# Space Complexity: O(H * W)
# ============================================================

def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> float:
    """
    Bilinear interpolation at sub-pixel coordinates.

    Args:
        image: 2D array (H, W)
        x: Row coordinate (can be fractional)
        y: Column coordinate (can be fractional)

    Returns:
        Interpolated pixel value
    """
    H, W = image.shape[:2]

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    # Clamp to image bounds
    x0 = max(0, min(x0, H - 1))
    x1 = max(0, min(x1, H - 1))
    y0 = max(0, min(y0, W - 1))
    y1 = max(0, min(y1, W - 1))

    dx = x - int(np.floor(x))
    dy = y - int(np.floor(y))

    if image.ndim == 2:
        val = (image[x0, y0] * (1 - dx) * (1 - dy) +
               image[x1, y0] * dx * (1 - dy) +
               image[x0, y1] * (1 - dx) * dy +
               image[x1, y1] * dx * dy)
        return val
    else:
        val = (image[x0, y0] * (1 - dx) * (1 - dy) +
               image[x1, y0] * dx * (1 - dy) +
               image[x0, y1] * (1 - dx) * dy +
               image[x1, y1] * dx * dy)
        return val


def rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate image by given angle using inverse mapping and bilinear interpolation.

    Args:
        image: Input (H, W) or (H, W, C)
        angle_degrees: Rotation angle in degrees (counterclockwise)

    Returns:
        Rotated image (same size, with black fill for out-of-bounds)
    """
    image = image.astype(np.float64)
    is_color = image.ndim == 3

    H, W = image.shape[:2]
    cx, cy = H / 2, W / 2
    angle_rad = np.radians(angle_degrees)

    # Inverse rotation matrix (to map output -> input coordinates)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if is_color:
        output = np.zeros_like(image)
    else:
        output = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            # Translate to center
            x = i - cx
            y = j - cy

            # Inverse rotation
            src_x = x * cos_a + y * sin_a + cx
            src_y = -x * sin_a + y * cos_a + cy

            # Check bounds
            if 0 <= src_x < H - 1 and 0 <= src_y < W - 1:
                output[i, j] = bilinear_interpolate(image, src_x, src_y)

    return output


# ============================================================
# Random Crop and Resize
# ============================================================

def random_crop_resize(image: np.ndarray, crop_ratio: float = 0.8,
                       seed: Optional[int] = None) -> np.ndarray:
    """
    Random crop and resize back to original size using nearest neighbor.

    Args:
        image: Input (H, W) or (H, W, C)
        crop_ratio: Fraction of original size to crop (0, 1]
        seed: Random seed

    Returns:
        Cropped and resized image
    """
    rng = np.random.RandomState(seed)
    H, W = image.shape[:2]

    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)

    top = rng.randint(0, H - crop_h + 1)
    left = rng.randint(0, W - crop_w + 1)

    if image.ndim == 3:
        crop = image[top:top+crop_h, left:left+crop_w, :]
    else:
        crop = image[top:top+crop_h, left:left+crop_w]

    # Nearest-neighbor resize
    row_indices = (np.arange(H) * crop_h / H).astype(int)
    col_indices = (np.arange(W) * crop_w / W).astype(int)
    row_indices = np.clip(row_indices, 0, crop_h - 1)
    col_indices = np.clip(col_indices, 0, crop_w - 1)

    return crop[np.ix_(row_indices, col_indices)]


# ============================================================
# Brightness and Contrast
# ============================================================

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by multiplying pixel values. factor > 1 = brighter."""
    result = image.astype(np.float64) * factor
    return np.clip(result, 0, 255).astype(image.dtype)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast. factor > 1 = more contrast, < 1 = less."""
    img = image.astype(np.float64)
    mean = np.mean(img)
    result = mean + factor * (img - mean)
    return np.clip(result, 0, 255).astype(image.dtype)


# ============================================================
# Gaussian Noise
# ============================================================

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0,
                       std: float = 25.0, seed: Optional[int] = None) -> np.ndarray:
    """Add Gaussian noise to image."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, std, image.shape)
    result = image.astype(np.float64) + noise
    return np.clip(result, 0, 255).astype(image.dtype)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Horizontal Flip
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    flipped_h = horizontal_flip(img)
    expected_h = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    print(f"H-Flip:            {'PASS' if np.array_equal(flipped_h, expected_h) else 'FAIL'}")

    # Test Vertical Flip
    flipped_v = vertical_flip(img)
    expected_v = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    print(f"V-Flip:            {'PASS' if np.array_equal(flipped_v, expected_v) else 'FAIL'}")

    # Test: Double flip = original
    double_h = horizontal_flip(horizontal_flip(img))
    print(f"Double H-flip:     {'PASS' if np.array_equal(double_h, img) else 'FAIL'}")

    # Test: Rotation by 0 degrees
    rotated_0 = rotate(img.astype(np.float64), 0)
    print(f"Rotate 0°:         {'PASS' if np.allclose(rotated_0, img, atol=1) else 'FAIL'}")

    # Test: Rotation preserves shape
    big_img = np.random.randint(0, 256, (10, 10)).astype(np.float64)
    rotated_45 = rotate(big_img, 45)
    print(f"Rotate shape:      {'PASS' if rotated_45.shape == big_img.shape else 'FAIL'}")

    # Test: Random crop resize
    crop_result = random_crop_resize(big_img, crop_ratio=0.8, seed=42)
    print(f"Crop-resize shape: {'PASS' if crop_result.shape == big_img.shape else 'FAIL'}")

    # Test: Brightness
    bright = adjust_brightness(np.array([[100, 150]], dtype=np.uint8), 1.5)
    print(f"Brightness:        {'PASS' if bright[0, 0] == 150 else 'FAIL'} ({bright[0, 0]})")
    print(f"Brightness clamp:  {'PASS' if bright[0, 1] == 225 else 'FAIL'} ({bright[0, 1]})")

    # Test: Gaussian noise changes values
    noisy = add_gaussian_noise(big_img.astype(np.uint8), std=10, seed=42)
    print(f"Noise changes img: {'PASS' if not np.array_equal(noisy, big_img.astype(np.uint8)) else 'FAIL'}")
    print(f"Noise shape:       {'PASS' if noisy.shape == big_img.shape else 'FAIL'}")

    # Test: Contrast
    contrast = adjust_contrast(np.array([[50, 200]], dtype=np.uint8), 2.0)
    print(f"Contrast shape:    {'PASS' if contrast.shape == (1, 2) else 'FAIL'}")
