# Data Augmentation

**Difficulty:** Hard
**Category:** Computer Vision

---

## Description

Implement common image data augmentation techniques manually using NumPy:

1. **Horizontal/Vertical Flip:** Mirror the image along an axis
2. **Rotation:** Rotate by arbitrary angle using rotation matrix and bilinear interpolation
3. **Random Crop and Resize:** Extract a random sub-region and resize to original dimensions
4. **Brightness/Contrast Adjustment:** Modify pixel intensity distributions
5. **Gaussian Noise:** Add random noise sampled from a normal distribution

All implementations must use NumPy only (no cv2 for transforms).

### Constraints

- Input: NumPy array (H, W) for grayscale or (H, W, C) for color
- Output: Augmented image with same dtype as input
- Rotation uses bilinear interpolation to avoid jagged edges
- Pixel values should be clipped to valid range [0, 255] for uint8

---

## Examples

### Example 1

**Input:**
```
image = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
```

**Output (Horizontal Flip):**
```
[[3, 2, 1],
 [6, 5, 4],
 [9, 8, 7]]
```

**Explanation:** Each row is reversed.

### Example 2

**Input:**
```
image = 4×4 matrix, angle = 90°
```

**Output:**
```
Rotated 90° counterclockwise
```

**Explanation:** Rotation matrix maps each pixel to its new position.

---

## Approach Hints

1. **Flips:** Simple array indexing with reversed slices
2. **Rotation:** Apply inverse rotation matrix to output coordinates to find source coordinates, then interpolate

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Flip | O(H × W) | O(H × W) |
| Rotation | O(H × W) | O(H × W) |
