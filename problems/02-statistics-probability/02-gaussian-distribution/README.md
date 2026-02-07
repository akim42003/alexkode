# Probability Distributions (Gaussian)

**Difficulty:** Easy
**Category:** Statistics & Probability

---

## Description

Implement the Gaussian (Normal) distribution:

1. **PDF (Probability Density Function):**
   ```
   f(x) = (1 / (σ√(2π))) × exp(-(x - μ)² / (2σ²))
   ```
2. **CDF (Cumulative Distribution Function):** Using numerical integration (trapezoidal rule) of the PDF from -∞ to x.

### Constraints

- μ (mean) can be any real number
- σ (standard deviation) must be > 0
- For CDF, approximate -∞ as μ - 10σ with at least 1000 integration steps

---

## Examples

### Example 1

**Input:**
```
x = 0, mu = 0, sigma = 1
```

**Output:**
```
PDF ≈ 0.3989 (peak of standard normal)
CDF ≈ 0.5
```

**Explanation:** At x=0 for the standard normal, the PDF is at its maximum and the CDF is exactly 0.5 (by symmetry).

### Example 2

**Input:**
```
x = 1, mu = 0, sigma = 1
```

**Output:**
```
PDF ≈ 0.2420
CDF ≈ 0.8413
```

**Explanation:** One standard deviation above the mean, ~84.13% of the distribution is below this point.

---

## Approach Hints

1. **Direct Formula:** Implement the PDF formula directly, handle numerical stability with large exponents
2. **Trapezoidal Rule for CDF:** Integrate the PDF numerically from a low bound to x

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| PDF | O(1) | O(1) |
| CDF (Trapezoidal) | O(n_steps) | O(1) |
