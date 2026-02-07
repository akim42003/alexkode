# Two-Layer Neural Network

**Difficulty:** Hard

**Category:** Deep Learning

---

## Description

Implement a complete two-layer neural network (one hidden layer) with training, including forward pass, backward pass, loss computation, and gradient descent training loop.

Architecture: `Input -> Linear -> ReLU -> Linear -> Softmax -> Cross-Entropy Loss`

Implement:

1. **`TwoLayerNN` class** with `forward`, `backward`, `compute_loss`, and `train` methods.
2. Support configurable hidden size, learning rate, and regularization.
3. Training loop with loss history.

### Constraints

- Input shape: `(N, D)`, output classes: `C`.
- Hidden layer uses ReLU activation.
- Output uses softmax + cross-entropy.
- Optional L2 regularization on weights.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
nn = TwoLayerNN(input_dim=2, hidden_dim=10, output_dim=2)
# XOR-like problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
nn.train(X, y, epochs=1000, lr=0.1)
```

**Output:**
```
Epoch 0: loss=0.693
Epoch 500: loss=0.05
Epoch 1000: loss=0.01
Accuracy: 100%
```

**Explanation:** Two-layer network can learn XOR (not linearly separable).

### Example 2

**Input:**
```python
# Spiral dataset with 3 classes
nn = TwoLayerNN(input_dim=2, hidden_dim=100, output_dim=3)
```

**Output:**
```
Training accuracy > 95% after sufficient epochs.
```

**Explanation:** The hidden layer learns non-linear decision boundaries.

---

## Approach Hints

1. Initialize weights with He initialization for the ReLU layer.
2. Forward: `Z1 = X@W1+b1`, `A1 = relu(Z1)`, `Z2 = A1@W2+b2`, `probs = softmax(Z2)`.
3. Backward: Start with `dZ2 = probs - y_onehot`, then chain rule back through ReLU.
4. L2 regularization adds `0.5 * reg * (||W1||^2 + ||W2||^2)` to the loss.

---

## Expected Complexity

| Approach | Time per epoch | Space |
|----------|----------------|-------|
| Full batch | O(N * D * H + N * H * C) | O(N * H + H * C) |
