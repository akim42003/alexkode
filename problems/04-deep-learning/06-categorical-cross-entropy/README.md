# Categorical Cross-Entropy Loss

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement the categorical cross-entropy loss function for multi-class classification. Given one-hot encoded true labels and predicted probability distributions, compute the loss and gradient.

Implement:

1. **`categorical_cross_entropy(y_true, y_pred)`** -- Per-sample losses with clipping.
2. **`categorical_cross_entropy_mean(y_true, y_pred)`** -- Mean loss over the batch.
3. **`softmax_cross_entropy(logits, y_true)`** -- Combined softmax + cross-entropy from logits (numerically superior).

### Constraints

- `y_true` shape: `(N, C)` one-hot encoded.
- `y_pred` shape: `(N, C)` predicted probabilities.
- Must handle edge cases where predictions are near 0 or 1.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
```

**Output:**
```
per-sample losses: [0.1054, 0.2231, 0.1054]
mean loss: 0.1446
```

**Explanation:** Loss = -log(predicted probability of true class).

### Example 2

**Input:**
```python
y_true = np.array([[1, 0, 0]])
y_pred = np.array([[0.0, 0.5, 0.5]])
```

**Output:**
```
loss ≈ 34.54 (clipped at epsilon)
```

**Explanation:** Completely wrong prediction; clipping prevents -inf.

---

## Approach Hints

1. Clip predictions: `p = clip(p, eps, 1-eps)` before log.
2. Per-sample: `L_i = -sum(y_true_i * log(p_i))`.
3. For softmax+CE from logits: use log-sum-exp trick for stability.
4. Gradient of softmax+CE w.r.t. logits is simply `(softmax(z) - y) / N`.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Vectorized | O(N * C) | O(N * C) |
