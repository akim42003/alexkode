# KL Divergence

**Difficulty:** Medium
**Category:** Statistics & Probability

---

## Description

Implement the Kullback-Leibler (KL) Divergence between two discrete probability distributions P and Q:

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

KL divergence measures how one probability distribution diverges from a second, expected distribution. It is **not symmetric**: KL(P||Q) ≠ KL(Q||P).

Also implement Jensen-Shannon Divergence (symmetric version):
```
JS(P, Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
where M = 0.5 × (P + Q)
```

### Constraints

- P and Q are 1D arrays representing probability distributions (non-negative, sum to 1)
- Both must have the same length
- Handle edge cases: where P(x)=0, that term contributes 0. Where Q(x)=0 but P(x)>0, KL is +∞

---

## Examples

### Example 1

**Input:**
```
P = [0.4, 0.6]
Q = [0.5, 0.5]
```

**Output:**
```
KL(P||Q) ≈ 0.0204
```

**Explanation:** P is slightly different from uniform Q. KL(P||Q) = 0.4×log(0.4/0.5) + 0.6×log(0.6/0.5) ≈ 0.0204.

### Example 2

**Input:**
```
P = [0.1, 0.2, 0.3, 0.4]
Q = [0.25, 0.25, 0.25, 0.25]
```

**Output:**
```
KL(P||Q) ≈ 0.1067
```

**Explanation:** P is compared against the uniform distribution Q.

---

## Approach Hints

1. **Direct Summation:** Iterate and compute each term, handling zeros carefully
2. **Vectorized with Masking:** Use numpy to compute all terms at once, masking zeros

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Direct | O(n) | O(1) |
