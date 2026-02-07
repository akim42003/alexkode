# Adam Optimizer

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement the Adam (Adaptive Moment Estimation) optimizer, which combines momentum and RMSProp with bias correction.

Implement:

1. **`Adam` class** with `step()` method that updates parameters.
2. Maintain first moment (mean) and second moment (variance) estimates.
3. Apply bias correction: `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`.

### Constraints

- Default hyperparameters: `lr=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
- Must handle parameter lists of any shape.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
x = 5.0, optimize f(x) = x^2
adam = Adam(lr=0.01)
```

**Output:**
```
Converges to x≈0 within 100-200 steps
```

**Explanation:** Adam adapts learning rates per-parameter and handles sparse/noisy gradients well.

### Example 2

**Input:**
```python
# Compare Adam vs vanilla SGD on Rosenbrock function
```

**Output:**
```
Adam converges faster and more reliably than vanilla SGD.
```

**Explanation:** Adam's adaptive learning rate handles the curved valley of the Rosenbrock function better.

---

## Approach Hints

1. `m = beta1 * m + (1 - beta1) * g` (first moment)
2. `v = beta2 * v + (1 - beta2) * g^2` (second moment)
3. Bias correction is crucial in early steps when moments are biased toward zero.
4. Update: `param -= lr * m_hat / (sqrt(v_hat) + eps)`.

---

## Expected Complexity

| Approach | Time per step | Space |
|----------|---------------|-------|
| Adam | O(d) | O(2d) for m and v |
