# Matrix Transpose

**Difficulty:** Easy
**Category:** Linear Algebra

---

## Description

Implement a function that computes the transpose of a given matrix. The transpose of a matrix is obtained by swapping its rows and columns — the element at position (i, j) moves to position (j, i).

You must implement this **without** using `np.transpose`, `.T`, or `np.swapaxes`.

### Constraints

- Input is a 2D NumPy array (matrix) with shape (m, n)
- 1 ≤ m, n ≤ 1000
- Matrix elements can be integers or floats

---

## Examples

### Example 1

**Input:**
```
[[1, 2, 3],
 [4, 5, 6]]
```

**Output:**
```
[[1, 4],
 [2, 5],
 [3, 6]]
```

**Explanation:** The 2×3 matrix becomes a 3×2 matrix. Row 0 `[1,2,3]` becomes column 0, and row 1 `[4,5,6]` becomes column 1.

### Example 2

**Input:**
```
[[1, 2],
 [3, 4],
 [5, 6],
 [7, 8]]
```

**Output:**
```
[[1, 3, 5, 7],
 [2, 4, 6, 8]]
```

**Explanation:** The 4×2 matrix becomes a 2×4 matrix. Each row in the input becomes a column in the output.

---

## Approach Hints

1. **Index Swapping:** Create an output matrix of shape (n, m) and fill `out[j][i] = input[i][j]`
2. **List Comprehension / Stacking:** Build each row of the result by collecting column elements from the original

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Index Swapping | O(m × n) | O(m × n) |
