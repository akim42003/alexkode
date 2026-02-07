# Linear Regression (Normal Equation)

**Difficulty:** Easy
**Category:** Machine Learning

---

## Description

Implement linear regression using the Normal Equation (closed-form solution):

```
θ = (X^T X)^(-1) X^T y
```

This gives the optimal weights that minimize the mean squared error without iteration. Include automatic bias term handling by prepending a column of ones to X.

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,)
- X^T X must be invertible (n_samples ≥ n_features)
- Works for single and multiple features

---

## Examples

### Example 1

**Input:**
```
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]
```

**Output:**
```
weights: [0.0, 2.0]  (bias ≈ 0, slope = 2)
predictions for X=[[5]]: [10.0]
```

**Explanation:** The data follows y = 2x perfectly. The normal equation finds slope=2, intercept=0.

### Example 2

**Input:**
```
X = [[1, 1], [1, 2], [2, 2], [2, 3]]
y = [6, 8, 9, 11]
```

**Output:**
```
weights: [3.0, 1.0, 2.0]  (bias=3, w1=1, w2=2)
```

**Explanation:** y = 3 + 1×x1 + 2×x2. The normal equation recovers the exact coefficients.

---

## Approach Hints

1. **Normal Equation:** Prepend ones column, compute θ = (X^T X)^(-1) X^T y
2. **Pseudoinverse:** Use the Moore-Penrose pseudoinverse for better numerical stability

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Normal Equation | O(n×d² + d³) | O(d²) |
