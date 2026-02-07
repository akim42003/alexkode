# Batch Normalization

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement batch normalization with forward and backward passes.

**Training forward:**
1. Compute batch mean and variance.
2. Normalize: `x_hat = (x - mean) / sqrt(var + eps)`.
3. Scale and shift: `out = gamma * x_hat + beta`.
4. Update running statistics with exponential moving average.

**Inference forward:** Use running mean/variance instead of batch statistics.

**Backward:** Compute `dx`, `dgamma`, `dbeta` from upstream gradient.

### Constraints

- Input shape: `(N, D)`.
- `gamma`, `beta` are learnable vectors of shape `(D,)`.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
gamma = np.ones(2), beta = np.zeros(2)
```

**Output:**
```
out = [[-1.2247, -1.2247], [0, 0], [1.2247, 1.2247]]
```

**Explanation:** Each feature column is normalized to zero mean and unit variance.

### Example 2

**Input:**
```python
dout = np.ones((3, 2))
```

**Output:**
```
dgamma = [0, 0], dbeta = [3, 3]
dx ≈ [[0, 0], [0, 0], [0, 0]]
```

**Explanation:** Uniform upstream gradient produces zero dx (mean component is removed).

---

## Approach Hints

1. Store cache (x, x_hat, mean, var) for backward pass.
2. Backward: `dbeta = sum(dout)`, `dgamma = sum(dout * x_hat)`.
3. `dx` requires careful chain rule through normalization.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Forward | O(N * D) | O(N * D) for cache |
| Backward | O(N * D) | O(N * D) |
