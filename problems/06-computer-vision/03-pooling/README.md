# Max/Average Pooling

**Difficulty:** Medium
**Category:** Computer Vision

---

## Description

Implement pooling operations commonly used in CNNs:

1. **Max Pooling:** Take the maximum value in each pooling window
2. **Average Pooling:** Take the average value in each pooling window
3. **Global Average Pooling:** Average across the entire spatial dimensions

### Constraints

- Input: 2D (H, W) or 3D (H, W, C) array
- pool_size: (pH, pW) — size of the pooling window
- stride: step size (default = pool_size)
- Output size: ((H - pH) / stride + 1, (W - pW) / stride + 1)

---

## Examples

### Example 1

**Input:**
```
image = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
pool_size = (2, 2), stride = 2
```

**Output (Max Pooling):**
```
[[6, 8],
 [14, 16]]
```

**Explanation:** Non-overlapping 2×2 windows, taking max of each.

### Example 2

**Input:**
```
Same as above
```

**Output (Average Pooling):**
```
[[3.5, 5.5],
 [11.5, 13.5]]
```

**Explanation:** Average of each 2×2 window: (1+2+5+6)/4=3.5, etc.

---

## Approach Hints

1. **Sliding Window:** Iterate over non-overlapping (or overlapping) windows and apply the operation
2. **Reshape Trick:** For non-overlapping pools, reshape the array to apply operations along axes

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Pooling | O(H × W) | O(out_H × out_W) |
