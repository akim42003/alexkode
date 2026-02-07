# Logistic Regression

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement binary logistic regression using gradient descent.

**Model:** `P(y=1|x) = σ(x·w + b)` where σ is the sigmoid function.

**Binary Cross-Entropy Loss:**
```
L = -(1/n) × Σ[y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```

**Gradient:**
```
∂L/∂w = (1/n) × X^T × (σ(Xw) - y)
```

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,) with binary labels {0, 1}
- Clip sigmoid output to [ε, 1-ε] to avoid log(0)
- Return weights, loss history, and predictions

---

## Examples

### Example 1

**Input:**
```
X = [[0.5], [1.5], [2.5], [3.5]]
y = [0, 0, 1, 1]
```

**Output:**
```
Decision boundary at x ≈ 2.0
Predictions: [0, 0, 1, 1]
```

**Explanation:** Logistic regression finds a threshold separating the two classes.

### Example 2

**Input:**
```
X = [[1, 1], [1, 2], [2, 1], [3, 3], [3, 4], [4, 3]]
y = [0, 0, 0, 1, 1, 1]
```

**Output:**
```
accuracy = 1.0 (linearly separable data)
```

**Explanation:** The model finds a linear decision boundary separating the two clusters.

---

## Approach Hints

1. **Gradient Descent:** Iteratively update weights using the gradient of binary cross-entropy
2. **Newton's Method:** Use second-order information (Hessian) for faster convergence

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Gradient Descent | O(n × d × iterations) | O(d) |
