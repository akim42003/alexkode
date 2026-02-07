# Z-Score Normalization

**Difficulty:** Easy
**Category:** Statistics & Probability

---

## Description

Implement Z-score normalization (standardization). Transform each value in a dataset to represent how many standard deviations it is from the mean:

```
z_i = (x_i - μ) / σ
```

After normalization, the data will have mean ≈ 0 and standard deviation ≈ 1.

Do **not** use `sklearn.preprocessing` or any built-in normalization functions.

### Constraints

- Input is a 1D NumPy array of length n
- n ≥ 2
- If standard deviation is 0 (all values identical), return an array of zeros

---

## Examples

### Example 1

**Input:**
```
data = [1, 2, 3, 4, 5]
```

**Output:**
```
[-1.4142, -0.7071, 0.0, 0.7071, 1.4142]
```

**Explanation:** Mean=3, Std=√2≈1.4142. z = (x-3)/1.4142. The middle value maps to 0.

### Example 2

**Input:**
```
data = [10, 10, 10]
```

**Output:**
```
[0.0, 0.0, 0.0]
```

**Explanation:** All values are identical so std=0. Return zeros to avoid division by zero.

---

## Approach Hints

1. **Standard Formula:** Compute mean and std, then apply z = (x - mean) / std for each element
2. **Vectorized:** Subtract mean array and divide by std in one operation

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Standard | O(n) | O(n) |
