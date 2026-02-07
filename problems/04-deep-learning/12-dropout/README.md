# Dropout

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement inverted dropout regularization with forward and backward passes.

**Training forward:**
1. Generate binary mask: each element kept with probability `(1 - p)`.
2. Apply mask and scale survivors by `1 / (1 - p)`.

**Inference forward:** Return input unchanged.

**Backward:** Multiply upstream gradient by the same scaled mask.

### Constraints

- `p` is drop probability (0 <= p < 1).
- Input can be any shape.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
p = 0.5
```

**Output (training):**
```
out = [[2.0, 0.0, 6.0], [0.0, 10.0, 12.0]]  # survivors scaled by 2x
```

**Explanation:** Each neuron dropped with p=0.5, survivors scaled by 1/(1-0.5)=2.

### Example 2

**Input:**
```python
# Inference mode
```

**Output:**
```
out = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # unchanged
```

**Explanation:** Inverted dropout means no scaling needed at inference time.

---

## Approach Hints

1. Mask: `(np.random.rand(*shape) >= p)`.
2. Store mask for backward pass.
3. Expected value is preserved: `E[out] = E[x]` because surviving neurons are scaled up.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Forward | O(n) | O(n) for mask |
| Backward | O(n) | O(1) extra |
