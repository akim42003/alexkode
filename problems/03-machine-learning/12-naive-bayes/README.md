# Naive Bayes Classifier

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement a Gaussian Naive Bayes classifier. The model assumes:
- Features are conditionally independent given the class
- Each feature follows a Gaussian distribution within each class

**Training:** For each class, compute the prior P(C), mean, and variance of each feature.

**Prediction:** For a new sample x, compute:
```
P(C|x) ∝ P(C) × Π P(x_i | C)
```
where P(x_i | C) is the Gaussian PDF with class-specific mean and variance.

### Constraints

- X has shape (n_samples, n_features) with continuous features
- y has shape (n_samples,) with discrete class labels
- Add a small epsilon to variance to avoid division by zero

---

## Examples

### Example 1

**Input:**
```
X_train = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[0, 0]]
```

**Output:**
```
prediction = [0] or [1]  (near decision boundary)
```

**Explanation:** The test point is at the origin, equidistant from both class centers.

### Example 2

**Input:**
```
X_train = [[1, 1], [1, 2], [2, 1], [5, 5], [6, 5], [5, 6]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[1.5, 1.5], [5.5, 5.5]]
```

**Output:**
```
predictions = [0, 1]
```

**Explanation:** Each test point is clearly closer to one cluster.

---

## Approach Hints

1. **Gaussian NB:** Compute per-class statistics, then evaluate Gaussian PDF for each feature
2. **Log-Space:** Work with log probabilities to avoid numerical underflow

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Gaussian NB | O(n × d) train, O(c × d) predict per sample | O(c × d) |
