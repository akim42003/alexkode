# Singular Value Decomposition (SVD)

**Difficulty:** Hard
**Category:** Linear Algebra

---

## Description

Implement Singular Value Decomposition (SVD). Given a matrix A of shape (m, n), decompose it into three matrices:

```
A = U × Σ × V^T
```

Where:
- U is an (m, m) orthogonal matrix (left singular vectors)
- Σ is an (m, n) diagonal matrix (singular values, non-negative, in decreasing order)
- V^T is an (n, n) orthogonal matrix (right singular vectors transposed)

The singular values are the square roots of the eigenvalues of A^T A (or A A^T).

### Constraints

- A is a 2D NumPy array of shape (m, n)
- 1 ≤ m, n ≤ 50
- You may use your own eigenvalue decomposition as a subroutine (or np.linalg.eigh for the eigenvalue step)

---

## Examples

### Example 1

**Input:**
```
A = [[3, 0],
     [0, 2]]
```

**Output:**
```
U = [[1, 0], [0, 1]]
Σ = [[3, 0], [0, 2]]
V^T = [[1, 0], [0, 1]]
```

**Explanation:** A is already diagonal, so singular values are 3 and 2 (the diagonal elements). U and V are identity.

### Example 2

**Input:**
```
A = [[1, 2],
     [3, 4],
     [5, 6]]
```

**Output:**
```
Singular values ≈ [9.526, 0.514]
```

**Explanation:** The two singular values of this 3×2 matrix. Verify: U @ Σ @ V^T ≈ A.

---

## Approach Hints

1. **Via Eigendecomposition:** Compute eigenvalues/vectors of A^T A to get V and σ², then compute U = A V Σ⁻¹
2. **One-Sided Jacobi:** Iteratively apply plane rotations to orthogonalize the columns of A

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Eigendecomposition | O(n³ + m×n²) | O(m×n) |
