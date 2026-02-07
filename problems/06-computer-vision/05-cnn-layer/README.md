# Implement a CNN Layer

**Difficulty:** Hard
**Category:** Computer Vision

---

## Description

Implement a complete convolutional neural network layer with forward and backward passes.

**Forward:** Apply multiple filters to a multi-channel input with bias, stride, and padding.

**Backward:** Compute gradients for:
- Filters (dW)
- Biases (db)
- Input (dX) for backpropagation to previous layers

### Constraints

- Input shape: (batch, C_in, H, W)
- Filter shape: (C_out, C_in, kH, kW)
- Bias shape: (C_out,)
- Output shape: (batch, C_out, H_out, W_out)
- Support configurable stride and padding

---

## Examples

### Example 1

**Input:**
```
X shape: (1, 1, 4, 4)   # 1 image, 1 channel, 4x4
W shape: (2, 1, 3, 3)   # 2 filters, 1 input channel, 3x3
stride = 1, padding = 0
```

**Output:**
```
shape: (1, 2, 2, 2)  # 1 image, 2 output channels, 2x2
```

**Explanation:** Each 3×3 filter slides over the 4×4 input producing a 2×2 feature map.

### Example 2

**Input:**
```
X shape: (2, 3, 6, 6)   # batch of 2, 3 channels (RGB), 6x6
W shape: (8, 3, 3, 3)   # 8 filters
stride = 2, padding = 1
```

**Output:**
```
shape: (2, 8, 3, 3)  # out = (6+2-3)/2+1 = 3
```

**Explanation:** 8 filters produce 8 feature maps. Stride 2 reduces spatial size.

---

## Approach Hints

1. **Loop-based:** Iterate over batch, filters, and spatial positions
2. **im2col:** Unfold input patches into a matrix for efficient matrix multiplication

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Loop-based | O(batch × C_out × C_in × H_out × W_out × kH × kW) | O(batch × C_out × H_out × W_out) |
