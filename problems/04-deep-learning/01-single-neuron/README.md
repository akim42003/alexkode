# Single Neuron (Perceptron)

**Difficulty:** Easy

**Category:** Deep Learning

---

## Description

Implement a single neuron (perceptron) from scratch. The perceptron takes weighted inputs, sums them, applies a step activation function, and produces a binary output.

Implement:

1. **`step_activation(z)`** -- Returns 1 if `z >= 0`, else 0. Works element-wise on arrays.
2. **`perceptron_forward(X, weights, bias)`** -- Forward pass: `z = X @ weights + bias`, then apply step activation.
3. **`train_perceptron(X, y, lr, epochs)`** -- Train using the perceptron learning rule: for each sample, update `weights += lr * error * x_i` and `bias += lr * error`.

### Constraints

- Use only NumPy.
- `X` shape `(n_samples, n_features)`, `y` shape `(n_samples,)` with binary values.
- Converges for linearly separable data.

---

## Examples

### Example 1: AND Gate

**Input:**
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
weights, bias = train_perceptron(X, y, lr=0.1, epochs=100)
predictions = perceptron_forward(X, weights, bias)
```

**Output:**
```
predictions = [0, 0, 0, 1]
```

**Explanation:** The perceptron learns the AND function. Only when both inputs are 1 does the output become 1.

### Example 2: OR Gate

**Input:**
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
weights, bias = train_perceptron(X, y, lr=0.1, epochs=100)
predictions = perceptron_forward(X, weights, bias)
```

**Output:**
```
predictions = [0, 1, 1, 1]
```

**Explanation:** The perceptron learns the OR function.

---

## Approach Hints

1. The step activation is a threshold at zero.
2. The perceptron learning rule adjusts weights only when there is an error.
3. XOR is NOT linearly separable -- a single perceptron cannot learn it.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Forward pass | O(n * d) | O(n) |
| Full training | O(epochs * n * d) | O(d) |
