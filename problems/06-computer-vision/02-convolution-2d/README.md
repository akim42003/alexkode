# 2D Convolution Operation

**Difficulty:** Easy
**Category:** Computer Vision

---

## Description

Implement 2D convolution from scratch. Slide a kernel over the image and compute the sum of element-wise products at each position.

Support two modes:
1. **Valid:** No padding, output is smaller: `(H-kH+1, W-kW+1)`
2. **Same:** Pad so output has same size as input

Demonstrate with common kernels: edge detection (Sobel), blur (Gaussian), sharpen.

### Constraints

- Image: 2D array (H, W) for grayscale or (H, W, C) for color
- Kernel: 2D array (kH, kW), typically square and odd-sized
- For multi-channel: convolve each channel independently
- Implement cross-correlation (the standard "convolution" in deep learning)

---

## Examples

### Example 1

**Input:**
```
image = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
kernel = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]  (sharpen kernel)
mode = "valid"
```

**Output:**
```
[[5]]  (single value for 3x3 image with 3x3 kernel in valid mode)
```

**Explanation:** 1×0+2×(-1)+3×0+4×(-1)+5×5+6×(-1)+7×0+8×(-1)+9×0 = -2-4-6-8+25 = 5.

### Example 2

**Input:**
```
image = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]
kernel = [[1, 0], [0, -1]]
mode = "valid"
```

**Output:**
```
[[-5, -5, -5],
 [-5, -5, -5]]
```

**Explanation:** Output shape = (3-2+1, 4-2+1) = (2, 3). Each element is top-left minus bottom-right.

---

## Approach Hints

1. **Sliding Window:** Iterate over valid positions, compute element-wise product and sum
2. **im2col:** Reshape image patches into columns for matrix multiplication (used in practice)

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Sliding Window | O(H × W × kH × kW) | O(H × W) |
