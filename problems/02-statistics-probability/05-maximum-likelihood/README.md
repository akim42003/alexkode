# Maximum Likelihood Estimation

**Difficulty:** Medium
**Category:** Statistics & Probability

---

## Description

Implement Maximum Likelihood Estimation (MLE) to estimate the parameters of a Gaussian distribution from data.

Given n data samples assumed to come from a normal distribution N(μ, σ²), find the parameters μ and σ that maximize the likelihood of observing the data.

**Closed-form MLE solutions for Gaussian:**
```
μ_MLE = (1/n) × Σx_i
σ²_MLE = (1/n) × Σ(x_i - μ_MLE)²
```

Also implement MLE using numerical optimization (gradient ascent on the log-likelihood).

### Constraints

- Input is a 1D array of data samples
- n ≥ 2
- Data is assumed to come from a Gaussian distribution

---

## Examples

### Example 1

**Input:**
```
data = [2.1, 2.5, 3.6, 4.0, 4.5, 5.2]
```

**Output:**
```
mu_mle ≈ 3.65
sigma_mle ≈ 1.059
```

**Explanation:** MLE for mean is the sample mean. MLE for σ uses population variance (divide by n, not n-1).

### Example 2

**Input:**
```
data = [10.2, 9.8, 10.1, 10.0, 9.9]
```

**Output:**
```
mu_mle ≈ 10.0
sigma_mle ≈ 0.1414
```

**Explanation:** Data is tightly clustered around 10, giving small σ.

---

## Approach Hints

1. **Closed-Form:** Directly compute μ and σ using the MLE formulas
2. **Gradient Ascent:** Maximize the log-likelihood function numerically using derivatives

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Closed-Form | O(n) | O(1) |
| Gradient Ascent | O(n × iterations) | O(1) |
