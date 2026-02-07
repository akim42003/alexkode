# Binary Cross-Entropy Loss

**Difficulty:** Easy

**Category:** Deep Learning

---

## Description

Implement the binary cross-entropy (BCE) loss function used for binary classification. Given true labels and predicted probabilities, compute the loss and its gradient.

Implement:

1. **`binary_cross_entropy(y_true, y_pred)`** -- Compute per-sample BCE loss with numerical clipping.
2. **`binary_cross_entropy_mean(y_true, y_pred)`** -- Mean BCE over the batch.
3. **`binary_cross_entropy_gradient(y_true, y_pred)`** -- Gradient of mean BCE w.r.t. predictions.

### Constraints

- `y_true` is a 1D array with values in {0, 1}.
- `y_pred` is a 1D array with values in (0, 1).
- Must clip predictions to avoid log(0).
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3])
```

**Output:**
```
per-sample losses: [0.1054, 0.1054, 0.2231, 0.3567]
mean loss: 0.1976
```

**Explanation:** Low loss when predictions match labels closely. Higher loss for y_pred=0.3 when y_true=0.

### Example 2

**Input:**
```python
y_true = np.array([1])
y_pred = np.array([0.0])  # completely wrong
```

**Output:**
```
loss ≈ 34.54 (clipped)
```

**Explanation:** Without clipping, log(0) = -inf. With clipping at 1e-15, loss = -log(1e-15) ≈ 34.54.

---

## Approach Hints

1. BCE formula: `L = -(y * log(p) + (1-y) * log(1-p))`.
2. Clip predictions: `p = clip(p, epsilon, 1 - epsilon)`.
3. Gradient: `dL/dp = -(y/p - (1-y)/(1-p)) / N`.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Vectorized | O(n) | O(n) |
