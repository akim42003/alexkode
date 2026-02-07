# Matrix Multiplication

**Difficulty:** Easy
**Category:** Linear Algebra

---

## Description

Implement matrix multiplication from scratch. Given two matrices A (shape m×n) and B (shape n×p), compute their product C = A × B where each element C[i][j] is the dot product of row i of A and column j of B.

You must implement this **without** using `np.matmul`, `np.dot`, or the `@` operator.

### Constraints

- A has shape (m, n) and B has shape (n, p) — inner dimensions must match
- 1 ≤ m, n, p ≤ 500
- If dimensions are incompatible, raise a `ValueError`

---

## Examples

### Example 1

**Input:**
```
A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]
```

**Output:**
```
[[19, 22],
 [43, 50]]
```

**Explanation:** C[0][0] = 1×5 + 2×7 = 19, C[0][1] = 1×6 + 2×8 = 22, C[1][0] = 3×5 + 4×7 = 43, C[1][1] = 3×6 + 4×8 = 50.

### Example 2

**Input:**
```
A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]
```

**Output:**
```
[[58, 64],
 [139, 154]]
```

**Explanation:** A is 2×3 and B is 3×2, resulting in a 2×2 matrix. C[0][0] = 1×7 + 2×9 + 3×11 = 58.

---

## Approach Hints

1. **Triple Nested Loop:** Iterate over rows of A, columns of B, and the shared dimension to accumulate dot products
2. **Row-Column Dot Products:** For each (i, j), compute the sum of element-wise products of row i and column j

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Triple Loop | O(m × n × p) | O(m × p) |
