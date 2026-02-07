# Solve Linear Systems (Jacobi Method)

**Difficulty:** Medium
**Category:** Linear Algebra

---

## Description

Solve a system of linear equations Ax = b using the Jacobi iterative method. The Jacobi method decomposes A into its diagonal (D) and off-diagonal (R = A - D) components, then iteratively computes:

```
x_new[i] = (1/A[i][i]) * (b[i] - Σ(A[i][j] * x_old[j]) for j ≠ i)
```

Also implement the Gauss-Seidel method as a comparison, which uses updated values immediately within the same iteration.

### Constraints

- A is a square matrix of shape (n, n)
- b is a vector of shape (n,)
- A must be diagonally dominant for guaranteed convergence
- 2 ≤ n ≤ 100
- Maximum iterations: 1000, tolerance: 1e-8

---

## Examples

### Example 1

**Input:**
```
A = [[4, 1, 0],
     [1, 4, 1],
     [0, 1, 4]]
b = [5, 10, 5]
```

**Output:**
```
x ≈ [0.625, 2.125, 0.625]
```

**Explanation:** This diagonally dominant system converges. The exact solution can be verified: A × x ≈ b.

### Example 2

**Input:**
```
A = [[10, -1, 2],
     [-1, 11, -1],
     [2, -1, 10]]
b = [6, 25, -11]
```

**Output:**
```
x ≈ [1.0, 2.0, -1.0]
```

**Explanation:** The system has exact solution [1, 2, -1]. Jacobi converges quickly due to strong diagonal dominance.

---

## Approach Hints

1. **Jacobi Method:** Update all variables simultaneously using only values from the previous iteration
2. **Gauss-Seidel Method:** Update variables sequentially, using the most recent values — typically converges faster

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Jacobi | O(n² × iterations) | O(n) |
| Gauss-Seidel | O(n² × iterations) | O(n) |
