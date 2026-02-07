# Softmax Function

**Difficulty:** Easy

**Category:** Deep Learning

---

## Description

Implement the softmax function and its Jacobian. Softmax converts a vector of raw scores (logits) into a probability distribution.

Implement:

1. **`softmax(z)`** -- Compute softmax with numerical stability (subtract max before exp).
2. **`softmax_batch(Z)`** -- Apply softmax row-wise on a 2D array.
3. **`softmax_jacobian(s)`** -- Compute the full Jacobian matrix for a single softmax output vector.

### Constraints

- Must be numerically stable (subtract max per sample).
- Each row of output sums to 1.0.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
z = np.array([2.0, 1.0, 0.1])
```

**Output:**
```
softmax(z) = [0.6590, 0.2424, 0.0986]
```

**Explanation:** Largest logit (2.0) gets highest probability. Sum = 1.0.

### Example 2

**Input:**
```python
z = np.array([1000, 1000, 1000])
```

**Output:**
```
softmax(z) = [0.3333, 0.3333, 0.3333]
```

**Explanation:** Equal logits produce uniform distribution. Subtracting max prevents overflow.

---

## Approach Hints

1. Subtract `max(z)` before `exp` to prevent overflow.
2. The Jacobian of softmax `s` is: `J[i,j] = s[i]*(1-s[j])` if `i==j`, else `-s[i]*s[j]`.
3. This is equivalently `diag(s) - s @ s.T`.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Softmax | O(n) | O(n) |
| Jacobian | O(n^2) | O(n^2) |
