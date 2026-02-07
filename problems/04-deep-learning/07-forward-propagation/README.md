# Forward Propagation

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement forward propagation for a fully-connected multi-layer perceptron (MLP). Given input data, weight matrices, bias vectors, and activation functions, compute the output and cache intermediate values.

Implement:

1. Activation functions: `relu`, `sigmoid`, `tanh`, `softmax`, `linear`.
2. **`forward_propagation(X, weights, biases, activations)`** -- Full forward pass returning output and cache.

### Constraints

- `X` shape: `(N, d_in)`.
- `weights[l]` shape: `(d_l, d_{l+1})`.
- `biases[l]` shape: `(d_{l+1},)`.
- `activations[l]` in `{"relu", "sigmoid", "tanh", "softmax", "linear"}`.
- Return cache with all pre-activation (Z) and post-activation (A) values.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
X = np.array([[1.0, 2.0]])           # (1, 2)
weights = [np.array([[0.1, 0.2],
                     [0.3, 0.4]])]   # (2, 2)
biases = [np.array([0.5, 0.6])]
activations = ["relu"]
```

**Output:**
```
output = [[1.2, 1.6]]
```

**Explanation:** Z = X @ W + b = [0.7, 1.0] + [0.5, 0.6] = [1.2, 1.6]. ReLU keeps positives.

### Example 2

**Input:**
```python
# 2-layer network: 2 -> 3 -> 1
X shape (4, 2), hidden layer relu, output sigmoid
```

**Output:**
```
output shape (4, 1), values in (0, 1)
```

**Explanation:** Hidden layer extracts features, output layer produces probability.

---

## Approach Hints

1. At each layer: `Z[l] = A[l-1] @ W[l] + b[l]`, then `A[l] = activation(Z[l])`.
2. Set `A[0] = X`, iterate through layers.
3. For softmax, subtract max per row before exp for numerical stability.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Sequential layers | O(N * sum(d_l * d_{l+1})) | O(N * sum(d_l)) |
