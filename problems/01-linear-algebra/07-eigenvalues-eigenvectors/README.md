# Eigenvalues and Eigenvectors

**Difficulty:** Medium
**Category:** Linear Algebra

---

## Description

Implement methods to find eigenvalues and eigenvectors of a square matrix:

1. **Power Iteration:** Find the dominant (largest magnitude) eigenvalue and its corresponding eigenvector by repeatedly multiplying by A and normalizing.
2. **QR Algorithm:** Find all eigenvalues by repeatedly decomposing A = QR and forming A' = RQ until convergence.

### Constraints

- Input is a square 2D NumPy array of shape (n, n)
- 2 ≤ n ≤ 50
- For power iteration: the matrix should have a unique dominant eigenvalue
- Convergence tolerance: 1e-8

---

## Examples

### Example 1

**Input:**
```
A = [[2, 1],
     [1, 3]]
```

**Output (Power Iteration):**
```
eigenvalue ≈ 3.618
eigenvector ≈ [0.526, 0.851] (normalized)
```

**Explanation:** The dominant eigenvalue of this symmetric matrix is (5+√5)/2 ≈ 3.618. Power iteration converges to it.

### Example 2

**Input:**
```
A = [[4, 1],
     [2, 3]]
```

**Output (QR Algorithm):**
```
eigenvalues ≈ [5.0, 2.0]
```

**Explanation:** The eigenvalues of this matrix are 5 and 2 (characteristic polynomial: λ²-7λ+10 = (λ-5)(λ-2)).

---

## Approach Hints

1. **Power Iteration:** Start with a random vector, repeatedly compute v = Av / ||Av||. The eigenvalue is estimated via Rayleigh quotient: λ = (v^T A v) / (v^T v)
2. **QR Algorithm:** Decompose A into Q and R using Gram-Schmidt, then update A = RQ. Diagonal converges to eigenvalues

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Power Iteration | O(n² × iterations) | O(n) |
| QR Algorithm | O(n³ × iterations) | O(n²) |
