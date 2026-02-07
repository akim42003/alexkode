# Self-Attention Mechanism

**Difficulty:** Medium
**Category:** NLP

---

## Description

Implement scaled dot-product self-attention and multi-head attention.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Multi-Head Attention:**
1. Project input into h separate Q, K, V heads
2. Compute attention for each head independently
3. Concatenate heads and project back

### Constraints

- Input: sequence of embeddings with shape (seq_len, d_model)
- Q, K, V are computed via linear projections: Q = X @ W_Q, etc.
- d_k = d_model / n_heads for multi-head attention
- Softmax is applied row-wise over the key dimension

---

## Examples

### Example 1

**Input:**
```
X = [[1, 0, 1, 0],   # Token 1 embedding (d_model=4)
     [0, 1, 0, 1],   # Token 2 embedding
     [1, 1, 0, 0]]   # Token 3 embedding
W_Q = W_K = W_V = eye(4)  (identity for simplicity)
```

**Output:**
```
attention_weights: (3, 3) matrix showing how each token attends to others
output: (3, 4) attended representations
```

**Explanation:** With identity projections, attention weights are based on dot-product similarity of the embeddings themselves.

### Example 2

**Input:**
```
X shape: (5, 8), n_heads = 2
```

**Output:**
```
Multi-head output shape: (5, 8)
Each head has d_k = 4
```

**Explanation:** Input split into 2 heads of dimension 4, computed independently, then concatenated.

---

## Approach Hints

1. **Single Head:** Matrix multiply Q×K^T, scale, softmax, multiply by V
2. **Multi-Head:** Split embedding dimension across heads, compute in parallel, concatenate

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Single Head Attention | O(seq² × d) | O(seq² + seq × d) |
| Multi-Head Attention | O(seq² × d) | O(h × seq² + seq × d) |
