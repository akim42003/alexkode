# Feature Scaling (Min-Max, Standard)

**Difficulty:** Easy
**Category:** Machine Learning

---

## Description

Implement two common feature scaling methods:

1. **Min-Max Scaling:** Scale features to [0, 1]:
   ```
   x_scaled = (x - x_min) / (x_max - x_min)
   ```

2. **Standard Scaling (Z-score):** Scale features to zero mean and unit variance:
   ```
   x_scaled = (x - mean) / std
   ```

Both should work on 2D data (rows=samples, columns=features), scaling each feature independently.

### Constraints

- Input is a 2D NumPy array of shape (n_samples, n_features)
- n_samples ≥ 1, n_features ≥ 1
- Handle edge case: if a feature has zero range/variance, set scaled values to 0

---

## Examples

### Example 1

**Input:**
```
X = [[1, 10],
     [2, 20],
     [3, 30]]
```

**Output (Min-Max):**
```
[[0.0, 0.0],
 [0.5, 0.5],
 [1.0, 1.0]]
```

**Explanation:** Feature 0: min=1, max=3. Feature 1: min=10, max=30. Both scale linearly to [0,1].

### Example 2

**Input:**
```
X = [[1, 10],
     [2, 20],
     [3, 30]]
```

**Output (Standard Scaling):**
```
[[-1.2247, -1.2247],
 [ 0.0,     0.0   ],
 [ 1.2247,  1.2247]]
```

**Explanation:** Both features have the same relative distribution, so z-scores are identical.

---

## Approach Hints

1. **Column-wise:** Compute statistics per column, then apply the formula
2. **Fit-Transform Pattern:** Separate fitting (computing stats) from transforming (applying scaling) to handle train/test properly

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Column-wise | O(n × d) | O(n × d) |
