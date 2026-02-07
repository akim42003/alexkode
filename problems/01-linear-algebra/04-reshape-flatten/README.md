# Reshape and Flatten

**Difficulty:** Easy
**Category:** Linear Algebra

---

## Description

Implement two operations:

1. **Reshape:** Transform a matrix from shape (m, n) to a target shape (p, q) where m×n = p×q. Elements are taken in row-major order.
2. **Flatten:** Convert a 2D matrix into a 1D array by concatenating rows.

Do **not** use `np.reshape`, `.reshape()`, `.flatten()`, or `.ravel()`.

### Constraints

- Input is a 2D NumPy array
- For reshape: total elements must match (m × n = p × q), otherwise raise `ValueError`
- 1 ≤ m, n, p, q ≤ 1000

---

## Examples

### Example 1

**Input:**
```
matrix = [[1, 2, 3, 4],
          [5, 6, 7, 8]]
target_shape = (4, 2)
```

**Output (Reshape):**
```
[[1, 2],
 [3, 4],
 [5, 6],
 [7, 8]]
```

**Explanation:** The 8 elements in row-major order [1,2,3,4,5,6,7,8] are rearranged into a 4×2 matrix.

### Example 2

**Input:**
```
matrix = [[1, 2, 3],
          [4, 5, 6]]
```

**Output (Flatten):**
```
[1, 2, 3, 4, 5, 6]
```

**Explanation:** Rows are concatenated: [1,2,3] + [4,5,6] = [1,2,3,4,5,6].

---

## Approach Hints

1. **Index Mapping:** Flatten to 1D conceptually, then map linear index k to (k // q, k % q) for the target shape
2. **Iterative Collection:** Collect all elements row by row, then distribute into the new shape

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Index Mapping | O(m × n) | O(m × n) |
