# Padding and Stride

**Difficulty:** Medium
**Category:** Computer Vision

---

## Description

Implement convolution with configurable padding and stride:

1. **Zero Padding:** Add rows/columns of zeros around the image border
2. **Strided Convolution:** Skip positions when sliding the kernel
3. **"Same" Padding:** Auto-calculate padding to preserve spatial dimensions

Output size formula: `out = floor((input + 2*pad - kernel) / stride) + 1`

### Constraints

- Input: 2D array (H, W)
- Padding: integer ≥ 0
- Stride: integer ≥ 1
- Kernel: 2D array (kH, kW)

---

## Examples

### Example 1

**Input:**
```
image = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
kernel = [[1, 0], [0, -1]]
padding = 1, stride = 1
```

**Output:**
```
shape = (4, 4)  (3 + 2*1 - 2)/1 + 1 = 4
```

**Explanation:** One pixel of zero-padding on all sides expands the 3×3 image to 5×5, then convolving with 2×2 kernel gives 4×4.

### Example 2

**Input:**
```
image = 6×6 matrix, kernel = 3×3
padding = 0, stride = 2
```

**Output:**
```
shape = (2, 2)  (6 + 0 - 3)/2 + 1 = 2
```

**Explanation:** Stride of 2 skips every other position, producing a smaller output.

---

## Approach Hints

1. **Pad then Convolve:** Apply zero padding first, then use standard convolution with stride
2. **Output Size Formula:** Always compute expected output size first to validate inputs

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Padded Strided Conv | O(out_H × out_W × kH × kW) | O(out_H × out_W) |
