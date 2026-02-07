# Principal Component Analysis (PCA)

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement PCA for dimensionality reduction:

1. Center the data (subtract column means)
2. Compute the covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort by eigenvalue magnitude (descending)
5. Project data onto the top n_components eigenvectors

Return the transformed data, principal components, and explained variance ratio.

### Constraints

- X has shape (n_samples, n_features)
- 1 ≤ n_components ≤ n_features
- n_samples ≥ 2

---

## Examples

### Example 1

**Input:**
```
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
n_components = 1
```

**Output:**
```
X_transformed shape: (4, 1)
explained_variance_ratio ≈ [1.0]  (features are perfectly correlated)
```

**Explanation:** Both features increase linearly, so one component captures ~100% variance.

### Example 2

**Input:**
```
X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]]
n_components = 2
```

**Output:**
```
explained_variance_ratio ≈ [0.96, 0.04]
```

**Explanation:** First component captures ~96% of the variance.

---

## Approach Hints

1. **Eigendecomposition of Covariance:** Direct and interpretable approach
2. **SVD-based:** More numerically stable — compute SVD of centered X directly

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Covariance + Eigen | O(n×d² + d³) | O(d²) |
| SVD-based | O(n×d²) | O(n×d) |
