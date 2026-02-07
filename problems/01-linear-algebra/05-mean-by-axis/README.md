# Calculate Mean by Row/Column

**Difficulty:** Easy
**Category:** Linear Algebra

---

## Description

Implement a function that calculates the mean of a matrix along a specified axis:

- **axis=0:** Compute the mean of each column (collapse rows)
- **axis=1:** Compute the mean of each row (collapse columns)
- **axis=None:** Compute the overall mean of all elements

Do **not** use `np.mean`, `.mean()`, or `np.average`.

### Constraints

- Input is a 2D NumPy array of shape (m, n)
- axis is one of: 0, 1, or None
- 1 ≤ m, n ≤ 1000

---

## Examples

### Example 1

**Input:**
```
matrix = [[1, 2, 3],
          [4, 5, 6]]
axis = 0
```

**Output:**
```
[2.5, 3.5, 4.5]
```

**Explanation:** Column means: (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5.

### Example 2

**Input:**
```
matrix = [[1, 2, 3],
          [4, 5, 6]]
axis = 1
```

**Output:**
```
[2.0, 5.0]
```

**Explanation:** Row means: (1+2+3)/3=2.0, (4+5+6)/3=5.0.

---

## Approach Hints

1. **Loop and Sum:** Iterate along the appropriate axis, summing elements and dividing by count
2. **Reduction:** Sum all elements along the axis, then divide by the axis length

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Loop and Sum | O(m × n) | O(m) or O(n) |
