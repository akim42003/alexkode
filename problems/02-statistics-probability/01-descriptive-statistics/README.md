# Calculate Mean, Variance, Std Dev

**Difficulty:** Easy
**Category:** Statistics & Probability

---

## Description

Implement functions to compute basic descriptive statistics from scratch:

1. **Mean:** The average of all values: μ = (1/n) × Σx_i
2. **Variance (Population):** σ² = (1/n) × Σ(x_i - μ)²
3. **Variance (Sample):** s² = (1/(n-1)) × Σ(x_i - μ)² (Bessel's correction)
4. **Standard Deviation:** Square root of variance (both population and sample)

Do **not** use `np.mean`, `np.var`, `np.std`, or any built-in statistics functions.

### Constraints

- Input is a 1D NumPy array or list of numbers
- n ≥ 1 for mean, n ≥ 2 for sample variance
- Elements can be integers or floats

---

## Examples

### Example 1

**Input:**
```
data = [2, 4, 4, 4, 5, 5, 7, 9]
```

**Output:**
```
mean = 5.0
population_variance = 4.0
sample_variance = 4.571428571428571
population_std = 2.0
sample_std = 2.138089935299395
```

**Explanation:** Mean = 40/8 = 5.0. Pop variance: sum of squared deviations / 8 = 32/8 = 4.0. Sample variance: 32/7 ≈ 4.571.

### Example 2

**Input:**
```
data = [10, 20, 30]
```

**Output:**
```
mean = 20.0
population_variance = 66.66666666666667
sample_variance = 100.0
population_std = 8.16496580927726
sample_std = 10.0
```

**Explanation:** Mean = 60/3 = 20. Deviations: [-10, 0, 10]. Sum of squares = 200. Pop var = 200/3, Sample var = 200/2 = 100.

---

## Approach Hints

1. **Two-Pass:** First pass to compute mean, second pass to compute variance
2. **Single-Pass (Welford's):** Online algorithm that computes mean and variance in one pass — numerically more stable

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Two-Pass | O(n) | O(1) |
| Welford's | O(n) | O(1) |
