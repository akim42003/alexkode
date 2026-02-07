# Scalar and Element-wise Operations

**Difficulty:** Easy
**Category:** Linear Algebra

---

## Description

Implement the following matrix operations from scratch:

1. **Scalar Multiplication:** Multiply every element of a matrix by a scalar value
2. **Element-wise Addition:** Add two matrices of the same shape element by element
3. **Element-wise Subtraction:** Subtract two matrices of the same shape element by element
4. **Hadamard Product:** Multiply two matrices of the same shape element by element

Do **not** use NumPy's built-in arithmetic operators (`+`, `-`, `*`) directly on arrays. Implement using explicit loops or index-based operations.

### Constraints

- All input matrices are 2D NumPy arrays
- For binary operations, both matrices must have the same shape
- 1 ≤ m, n ≤ 1000

---

## Examples

### Example 1

**Input:**
```
A = [[1, 2], [3, 4]]
scalar = 3
```

**Output (Scalar Multiplication):**
```
[[3, 6], [9, 12]]
```

**Explanation:** Each element is multiplied by 3.

### Example 2

**Input:**
```
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
```

**Output (Hadamard Product):**
```
[[5, 12], [21, 32]]
```

**Explanation:** Element-wise: 1×5=5, 2×6=12, 3×7=21, 4×8=32.

---

## Approach Hints

1. **Loop-based:** Iterate over every (i, j) and apply the operation
2. **Flat Iteration:** Flatten, apply operation, reshape back

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Loop-based | O(m × n) | O(m × n) |
