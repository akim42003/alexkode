# Gradient Descent Variants

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement three gradient descent variants used in neural network training:

1. **SGD with Momentum:** `v = beta * v - lr * grad`, `param += v`
2. **RMSProp:** `s = decay * s + (1-decay) * grad^2`, `param -= lr * grad / (sqrt(s) + eps)`
3. **Mini-batch SGD:** Split data into batches, apply SGD to each batch.

### Constraints

- Each optimizer maintains its own state (velocity, cache).
- Must work with arbitrary parameter shapes.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
# Optimize f(x) = x^2, starting at x=5
x = 5.0, lr = 0.1, momentum = 0.9
```

**Output (SGD with Momentum):**
```
Step 0: x=5.0
Step 1: x=4.0
Step 2: x=2.18
...converges to x≈0
```

**Explanation:** Momentum accumulates velocity, accelerating convergence.

### Example 2

**Input:**
```python
# RMSProp on f(x,y) = x^2 + 100*y^2
lr = 0.1, decay = 0.99
```

**Output:**
```
Converges faster than vanilla SGD on ill-conditioned functions.
```

**Explanation:** RMSProp adapts learning rates per-parameter, handling different scales.

---

## Approach Hints

1. SGD+Momentum: velocity accumulates gradient history, beta controls decay.
2. RMSProp: running average of squared gradients normalizes the update.
3. Both methods converge faster than vanilla SGD on most problems.

---

## Expected Complexity

| Approach | Time per step | Space |
|----------|---------------|-------|
| SGD + Momentum | O(d) | O(d) for velocity |
| RMSProp | O(d) | O(d) for cache |
