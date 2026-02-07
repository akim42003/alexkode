# Sigmoid Activation Function

**Difficulty:** Easy

**Category:** Deep Learning

---

## Description

Implement the sigmoid activation function and its derivative from scratch. The sigmoid function squashes any real number to the range (0, 1) and is fundamental to neural networks and logistic regression.

Implement:

1. **`sigmoid(z)`** -- Compute `1 / (1 + exp(-z))` with numerical stability.
2. **`sigmoid_derivative(z)`** -- Compute `sigmoid(z) * (1 - sigmoid(z))`.
3. **`sigmoid_neuron_forward(X, weights, bias)`** -- Single neuron with sigmoid activation.
4. **`sigmoid_neuron_backward(X, weights, bias, y)`** -- Compute gradients using binary cross-entropy loss.

### Constraints

- Must be numerically stable (handle very large positive/negative inputs).
- Works element-wise on arrays of any shape.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
z = np.array([-2, -1, 0, 1, 2])
```

**Output:**
```
sigmoid(z) = [0.1192, 0.2689, 0.5, 0.7311, 0.8808]
```

**Explanation:** sigmoid(0) = 0.5, negative inputs map below 0.5, positive above.

### Example 2

**Input:**
```python
z = np.array([0.0])
```

**Output:**
```
sigmoid_derivative(z) = [0.25]
```

**Explanation:** At z=0, sigmoid=0.5, so derivative = 0.5 * (1 - 0.5) = 0.25 (maximum).

---

## Approach Hints

1. For numerical stability, use `np.clip(z, -500, 500)` before computing `exp(-z)`.
2. The derivative can be computed from the output: `s' = s * (1 - s)`.
3. For binary cross-entropy gradient: `dz = (a - y)` where `a = sigmoid(z)`.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Sigmoid forward | O(n) | O(n) |
| Sigmoid derivative | O(n) | O(n) |
