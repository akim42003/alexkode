# Hypothesis Testing (t-test)

**Difficulty:** Medium
**Category:** Statistics & Probability

---

## Description

Implement a two-sample t-test to determine if two groups have significantly different means.

1. **Equal Variance (Student's t-test):**
   ```
   t = (x̄₁ - x̄₂) / (s_p × √(1/n₁ + 1/n₂))
   where s_p = √(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2))
   ```

2. **Unequal Variance (Welch's t-test):**
   ```
   t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
   ```

Compute the t-statistic and approximate the p-value. Determine whether to reject the null hypothesis (H₀: μ₁ = μ₂) at a given significance level.

### Constraints

- Both samples have at least 2 elements
- Significance level α is typically 0.05
- For p-value approximation, use numerical integration of the t-distribution PDF

---

## Examples

### Example 1

**Input:**
```
group1 = [5.1, 4.9, 5.0, 5.2, 4.8]
group2 = [5.5, 5.7, 5.6, 5.8, 5.4]
alpha = 0.05
```

**Output:**
```
t_statistic ≈ -5.48
reject_null = True
```

**Explanation:** Large |t| indicates the means are significantly different. Group 2 has a noticeably higher mean.

### Example 2

**Input:**
```
group1 = [3.0, 3.1, 2.9, 3.0]
group2 = [3.1, 2.9, 3.0, 3.0]
alpha = 0.05
```

**Output:**
```
t_statistic ≈ 0.0
reject_null = False
```

**Explanation:** The groups have nearly identical distributions, so we fail to reject H₀.

---

## Approach Hints

1. **Student's t-test:** Assumes equal variance — uses pooled standard deviation
2. **Welch's t-test:** Does not assume equal variance — more robust in general

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| t-test | O(n₁ + n₂) | O(1) |
