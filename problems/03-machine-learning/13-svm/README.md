# Support Vector Machine (SVM)

**Difficulty:** Hard
**Category:** Machine Learning

---

## Description

Implement a Support Vector Machine for binary classification using a simplified SMO (Sequential Minimal Optimization) algorithm.

The SVM finds the hyperplane that maximizes the margin between two classes. The optimization problem:
```
minimize (1/2)||w||² + C × Σ max(0, 1 - y_i(w·x_i + b))
```

Support both linear and RBF kernels.

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,) with labels in {-1, +1}
- C > 0 is the regularization parameter
- For RBF kernel: K(x, y) = exp(-γ||x-y||²)

---

## Examples

### Example 1

**Input:**
```
X = [[1, 1], [2, 2], [1, 2], [4, 4], [5, 5], [4, 5]]
y = [-1, -1, -1, 1, 1, 1]
C = 1.0
kernel = 'linear'
```

**Output:**
```
predictions = [-1, -1, -1, 1, 1, 1]  (100% accuracy)
```

**Explanation:** The data is linearly separable.

### Example 2

**Input (XOR-like):**
```
X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [-1, -1, 1, 1]
kernel = 'rbf', gamma = 1.0
```

**Output:**
```
RBF kernel can separate non-linear data
```

**Explanation:** Linear kernel fails on XOR, but RBF kernel can find a non-linear boundary.

---

## Approach Hints

1. **Simplified SMO:** Iterate over pairs of Lagrange multipliers and optimize them jointly
2. **Kernel Trick:** Replace dot products with kernel function for non-linear boundaries

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Simplified SMO | O(n² × iterations) | O(n²) for kernel matrix |
