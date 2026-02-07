# Linear Regression (Gradient Descent)

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement linear regression using gradient descent optimization. Instead of the closed-form normal equation, iteratively update weights to minimize the mean squared error loss.

**Gradient of MSE loss:**
```
∂L/∂θ = (2/n) × X^T × (Xθ - y)
```

**Update rule:**
```
θ = θ - learning_rate × ∂L/∂θ
```

Implement both batch gradient descent (use all data each step) and mini-batch gradient descent.

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,)
- learning_rate > 0 (typically 0.01)
- n_iterations > 0
- Return both final weights and loss history

---

## Examples

### Example 1

**Input:**
```
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]
learning_rate = 0.1
n_iterations = 1000
```

**Output:**
```
weights ≈ [0.0, 2.0]  (converges to y = 2x)
final_loss ≈ 0.0
```

**Explanation:** Gradient descent converges to the same solution as the normal equation.

### Example 2

**Input:**
```
X = [[1, 2], [3, 4], [5, 6]]
y = [5, 11, 17]
learning_rate = 0.01
n_iterations = 5000
```

**Output:**
```
weights ≈ [0.0, 1.0, 2.0]  (y ≈ x1 + 2*x2)
```

**Explanation:** The loss decreases each iteration and converges to the minimum.

---

## Approach Hints

1. **Batch GD:** Use full dataset gradient at each step — stable but can be slow
2. **Mini-Batch GD:** Use random subsets — faster per iteration, noisy gradients

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Batch GD | O(n × d × iterations) | O(d) |
| Mini-Batch GD | O(batch × d × iterations) | O(d) |
