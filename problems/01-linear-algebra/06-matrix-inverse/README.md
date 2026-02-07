# Matrix Inverse (2x2 and NxN)

**Difficulty:** Medium
**Category:** Linear Algebra

---

## Description

Implement matrix inversion:

1. **2×2 Inverse:** Use the closed-form formula: for matrix `[[a,b],[c,d]]`, the inverse is `(1/det) * [[d,-b],[-c,a]]` where `det = ad - bc`.
2. **NxN Inverse:** Use Gauss-Jordan elimination — augment the matrix with the identity, then row-reduce to transform the left side into the identity. The right side becomes the inverse.

If the matrix is singular (determinant is zero or a pivot is zero), raise a `ValueError`.

### Constraints

- Input is a square 2D NumPy array of shape (n, n)
- 1 ≤ n ≤ 100
- Matrix elements are floats

---

## Examples

### Example 1

**Input:**
```
[[4, 7],
 [2, 6]]
```

**Output:**
```
[[ 0.6, -0.7],
 [-0.2,  0.4]]
```

**Explanation:** det = 4×6 - 7×2 = 10. Inverse = (1/10) × [[6,-7],[-2,4]].

### Example 2

**Input:**
```
[[1, 2, 3],
 [0, 1, 4],
 [5, 6, 0]]
```

**Output:**
```
[[-24,  18,  5],
 [ 20, -15, -4],
 [ -5,   4,  1]]
```

**Explanation:** The inverse computed via Gauss-Jordan elimination. Verify: A × A⁻¹ = I.

---

## Approach Hints

1. **2×2 Formula:** Direct formula using determinant — fast and simple for 2×2
2. **Gauss-Jordan Elimination:** Augment [A|I], apply row operations to get [I|A⁻¹] — works for any size

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| 2×2 Formula | O(1) | O(1) |
| Gauss-Jordan | O(n³) | O(n²) |
