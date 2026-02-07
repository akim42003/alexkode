# Bayes' Theorem Implementation

**Difficulty:** Medium
**Category:** Statistics & Probability

---

## Description

Implement Bayes' Theorem to compute posterior probabilities:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

Implement two versions:

1. **Simple Bayes:** Given prior P(A), likelihood P(B|A), and evidence P(B), compute posterior P(A|B)
2. **Multiple Hypotheses:** Given n hypotheses with priors P(H_i) and likelihoods P(B|H_i), compute all posteriors P(H_i|B)

For multiple hypotheses, P(B) is computed via the law of total probability:
```
P(B) = Σ P(B|H_i) × P(H_i)
```

### Constraints

- All probabilities are in [0, 1]
- Priors must sum to 1 (for multiple hypotheses)
- P(B) must not be 0

---

## Examples

### Example 1

**Input:**
```
prior P(Disease) = 0.001
likelihood P(Positive|Disease) = 0.99
evidence P(Positive) = 0.05
```

**Output:**
```
P(Disease|Positive) = 0.0198
```

**Explanation:** Despite a positive test, the disease is rare so the posterior is still low (~2%). This illustrates the base rate fallacy.

### Example 2

**Input:**
```
priors = [0.6, 0.3, 0.1]  (3 hypotheses)
likelihoods = [0.2, 0.5, 0.9]  (P(evidence|H_i))
```

**Output:**
```
posteriors = [0.3529, 0.4412, 0.2647]  (sums to ≈ 1.0)
```

**Explanation:** P(B) = 0.6×0.2 + 0.3×0.5 + 0.1×0.9 = 0.36. Posterior_i = likelihood_i × prior_i / P(B).

---

## Approach Hints

1. **Direct Formula:** Apply Bayes' formula directly
2. **Log-Space:** For numerical stability with very small probabilities, compute in log space

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Direct | O(n) for n hypotheses | O(n) |
