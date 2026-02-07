# Gradient Boosting (Basic)

**Difficulty:** Hard
**Category:** Machine Learning

---

## Description

Implement Gradient Boosting for regression. Build an ensemble of shallow decision trees sequentially, where each tree fits the residuals (negative gradients) of the previous ensemble's predictions.

**Algorithm:**
1. Initialize prediction with the mean of y
2. For each iteration:
   - Compute residuals: r = y - current_prediction
   - Fit a shallow tree to the residuals
   - Update: prediction += learning_rate × tree_prediction
3. Final prediction is the sum of all tree predictions

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,) with continuous values (regression)
- n_estimators ≥ 1, learning_rate in (0, 1]
- max_depth for individual trees typically 1-5 (shallow)

---

## Examples

### Example 1

**Input:**
```
X = [[1], [2], [3], [4], [5]]
y = [1.0, 2.1, 2.9, 4.2, 4.8]
n_estimators = 50, learning_rate = 0.1
```

**Output:**
```
MSE < 0.1 on training data
```

**Explanation:** The ensemble of trees progressively reduces the residuals.

### Example 2

**Input:**
```
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [3, 7, 11, 15]
n_estimators = 100
```

**Output:**
```
predictions ≈ [3, 7, 11, 15]  (close fit to y = x1 + x2)
```

**Explanation:** The boosted ensemble learns the linear relationship.

---

## Approach Hints

1. **Basic Gradient Boosting:** Sequential tree fitting on residuals with a learning rate
2. **With Line Search:** Optimize the step size for each tree

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Gradient Boosting | O(n_estimators × n × d × depth) | O(n_estimators × nodes) |
