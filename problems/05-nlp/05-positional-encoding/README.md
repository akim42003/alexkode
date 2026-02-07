# Positional Encoding

**Difficulty:** Hard
**Category:** NLP

---

## Description

Implement positional encoding as described in "Attention is All You Need":

**Sinusoidal Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where `pos` is the position in the sequence and `i` is the dimension index.

Also implement learned positional embeddings (a trainable matrix).

### Constraints

- max_seq_len: maximum sequence length
- d_model: embedding dimension
- Output shape: (max_seq_len, d_model)
- Add positional encoding to input embeddings: X_out = X + PE

---

## Examples

### Example 1

**Input:**
```
max_seq_len = 4, d_model = 6
```

**Output:**
```
PE shape: (4, 6)
PE[0, :] = [sin(0), cos(0), sin(0), cos(0), sin(0), cos(0)]
         = [0, 1, 0, 1, 0, 1]
```

**Explanation:** At position 0, all sine terms are sin(0)=0 and cosine terms are cos(0)=1.

### Example 2

**Input:**
```
X = random embeddings (10, 64), max_seq_len = 10, d_model = 64
```

**Output:**
```
X_with_pos = X + PE[:10, :]  (same shape as X)
```

**Explanation:** Positional information is added to each token's embedding.

---

## Approach Hints

1. **Sinusoidal:** Pre-compute the full PE matrix using the formula — no training needed
2. **Learned:** Initialize a random matrix of shape (max_seq_len, d_model) and treat as parameters

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Sinusoidal | O(seq × d) | O(seq × d) |
| Learned | O(seq × d) | O(seq × d) |
