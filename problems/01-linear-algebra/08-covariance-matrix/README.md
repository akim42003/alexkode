# Covariance Matrix

**Difficulty:** Medium
**Category:** Linear Algebra

---

## Description

Compute the covariance matrix of a dataset. Given a data matrix X where each row is a sample and each column is a feature, compute the covariance matrix C where C[i][j] represents the covariance between feature i and feature j.

The formula for sample covariance is:
```
Cov(X_i, X_j) = (1/(n-1)) * Σ(X_i_k - mean_i)(X_j_k - mean_j)
```

Do **not** use `np.cov`.

### Constraints

- Input X has shape (n_samples, n_features)
- n_samples ≥ 2, n_features ≥ 1
- Output covariance matrix has shape (n_features, n_features)
- Use sample covariance (divide by n-1, Bessel's correction)

---

## Examples

### Example 1

**Input:**
```
X = [[1, 2],
     [3, 4],
     [5, 6]]
```

**Output:**
```
[[4.0, 4.0],
 [4.0, 4.0]]
```

**Explanation:** Both features increase together linearly. Var(X0) = ((1-3)²+(3-3)²+(5-3)²)/2 = 4. Cov(X0,X1) = ((1-3)(2-4)+(3-3)(4-4)+(5-3)(6-4))/2 = 4.

### Example 2

**Input:**
```
X = [[1, 5],
     [2, 3],
     [3, 1]]
```

**Output:**
```
[[ 1.0, -2.0],
 [-2.0,  4.0]]
```

**Explanation:** Feature 0 increases while feature 1 decreases, producing negative covariance. Var(X0)=1, Var(X1)=4, Cov=-2.

---

## Approach Hints

1. **Direct Formula:** Center the data (subtract means), then compute C = X_centered^T × X_centered / (n-1)
2. **Element-wise:** Compute each Cov(i,j) individually using the definition

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Matrix Formula | O(n × d²) | O(d²) |
| Element-wise | O(d² × n) | O(d²) |
