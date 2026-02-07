# Image Normalization

**Difficulty:** Easy
**Category:** Computer Vision

---

## Description

Implement image normalization techniques:

1. **Min-Max Normalization:** Scale pixel values to [0, 1]: `x_norm = (x - min) / (max - min)`
2. **Mean Subtraction:** Subtract the per-channel mean: `x_norm = x - mean_per_channel`
3. **Standardization:** Zero mean and unit variance per channel: `x_norm = (x - mean) / std`

All operations should handle both grayscale (H, W) and RGB (H, W, C) images.

### Constraints

- Input is a NumPy array with uint8 values [0, 255] or float values
- Output is a float64 array
- Handle grayscale (H, W) and color (H, W, C) images
- For per-channel operations, compute statistics independently per channel

---

## Examples

### Example 1

**Input:**
```
image = [[0, 128], [64, 255]]  (2x2 grayscale, uint8)
```

**Output (Min-Max):**
```
[[0.0, 0.502], [0.251, 1.0]]
```

**Explanation:** Values scaled to [0, 1] using min=0, max=255.

### Example 2

**Input:**
```
image = 3x3 RGB image with shape (3, 3, 3)
```

**Output (Standardization):**
```
Each channel has mean ≈ 0 and std ≈ 1
```

**Explanation:** Statistics computed independently per R, G, B channel.

---

## Approach Hints

1. **Global normalization:** Use min/max or mean/std across all pixels
2. **Per-channel:** Compute statistics for each color channel separately

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| All methods | O(H × W × C) | O(H × W × C) |
