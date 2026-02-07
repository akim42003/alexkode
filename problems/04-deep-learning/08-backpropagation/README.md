# Backpropagation

**Difficulty:** Hard

**Category:** Deep Learning

---

## Description

Implement backpropagation for a multi-layer perceptron. Given the forward pass cache, compute gradients of the loss with respect to all weights and biases.

Implement:

1. Activation derivatives: `relu_backward`, `sigmoid_backward`, `tanh_backward`.
2. **`backward_propagation(y_true, cache, weights, activations)`** -- Full backward pass returning gradients for all parameters.
3. **`gradient_check(X, y, weights, biases, activations)`** -- Numerical gradient verification.

### Constraints

- Works with any number of layers.
- Supports ReLU, sigmoid, and tanh activations.
- Output layer uses softmax + cross-entropy (combined gradient).
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
# 2-layer network: 2 -> 3 -> 2 (softmax)
X = np.array([[1.0, 0.0]])
y = np.array([[1, 0]])  # one-hot
```

**Output:**
```
gradients = {"dW": [dW1, dW2], "db": [db1, db2]}
dW1 shape: (2, 3), dW2 shape: (3, 2)
```

**Explanation:** Chain rule propagates the error from output to each layer's parameters.

### Example 2

**Input:**
```python
# Gradient check: numerical vs analytical
epsilon = 1e-5
```

**Output:**
```
max relative error < 1e-5 for all parameters
```

**Explanation:** Finite differences should match analytical gradients.

---

## Approach Hints

1. Start from output: `dZ_L = A_L - y` (softmax + CE combined gradient).
2. For each layer l going backwards: `dW[l] = A[l-1].T @ dZ[l] / N`, `db[l] = mean(dZ[l])`, `dA[l-1] = dZ[l] @ W[l].T`.
3. Apply activation derivative: `dZ[l-1] = dA[l-1] * act_derivative(Z[l-1])`.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Backward pass | O(N * sum(d_l * d_{l+1})) | O(N * sum(d_l)) |
