# Monte Carlo Sampling

**Difficulty:** Hard
**Category:** Statistics & Probability

---

## Description

Implement Monte Carlo methods for estimation:

1. **Estimate Pi:** Use random sampling to estimate π by checking if random points fall inside a unit circle inscribed in a square.
2. **Monte Carlo Integration:** Estimate the integral of an arbitrary function f(x) over [a, b] by sampling random points.
3. **Importance Sampling:** Improve estimation accuracy by sampling from a proposal distribution that better covers high-value regions.

### Constraints

- Number of samples: n ≥ 100
- Functions for integration should be provided as callables
- Importance sampling requires a known proposal distribution

---

## Examples

### Example 1

**Input:**
```
n_samples = 100000
```

**Output:**
```
pi_estimate ≈ 3.14159 (varies due to randomness)
```

**Explanation:** Generate random (x,y) in [0,1]². Count points where x²+y² ≤ 1. Ratio × 4 ≈ π.

### Example 2

**Input:**
```
f(x) = x^2, interval [0, 1], n_samples = 100000
```

**Output:**
```
integral ≈ 0.3333 (true value = 1/3)
```

**Explanation:** Expected value of f(x) over [a,b] times interval length: E[f(x)] × (b-a).

---

## Approach Hints

1. **Basic Monte Carlo:** Average f(x) for uniform random x, multiply by interval length
2. **Importance Sampling:** Sample from proposal q(x), weight by f(x)×p(x)/q(x)

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Basic MC | O(n) | O(1) |
| Importance Sampling | O(n) | O(n) |
