# ReLU and Variants

**Difficulty:** Easy

**Category:** Deep Learning

---

## Description

Implement ReLU and its popular variants with both forward and backward passes:

1. **ReLU:** `max(0, x)`
2. **Leaky ReLU:** `x if x > 0 else alpha * x` (default alpha=0.01)
3. **ELU:** `x if x > 0 else alpha * (exp(x) - 1)`
4. **GELU (approximation):** `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

Each function needs a corresponding derivative for the backward pass.

### Constraints

- Use only NumPy.
- Must work element-wise on arrays of any shape.
- Derivatives should handle the non-differentiable point at x=0 (use right derivative: f'(0) = 0 for ReLU).

---

## Examples

### Example 1

**Input:**
```python
x = np.array([-2, -1, 0, 1, 2])
```

**Output (ReLU):**
```
[0, 0, 0, 1, 2]
```

**Explanation:** Negative values are zeroed out, positive values pass through.

### Example 2

**Input:**
```python
x = np.array([-2, -1, 0, 1, 2])
alpha = 0.01
```

**Output (Leaky ReLU):**
```
[-0.02, -0.01, 0, 1, 2]
```

**Explanation:** Negative values are scaled by alpha instead of zeroed.

---

## Approach Hints

1. Use `np.maximum(0, x)` for ReLU forward, `(x > 0).astype(float)` for backward.
2. For ELU backward on the negative region: derivative is `alpha * exp(x)` = `output + alpha`.
3. GELU approximation uses the tanh-based formula from the original paper.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| All activations | O(n) | O(n) |
