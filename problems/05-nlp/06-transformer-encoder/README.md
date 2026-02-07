# Transformer Encoder Block

**Difficulty:** Hard
**Category:** NLP

---

## Description

Implement a complete Transformer encoder block as in "Attention is All You Need":

1. **Multi-Head Self-Attention** (with residual connection)
2. **Layer Normalization** (after attention)
3. **Feed-Forward Network** (two linear layers with ReLU)
4. **Layer Normalization** (after FFN)

Architecture: `X → MHA → Add & LayerNorm → FFN → Add & LayerNorm → Output`

Implement Layer Normalization from scratch: `LN(x) = γ × (x - μ) / (σ + ε) + β`

### Constraints

- Input shape: (seq_len, d_model)
- FFN hidden dimension typically d_ff = 4 × d_model
- Layer norm has learnable parameters γ (scale) and β (shift)
- Include residual connections (skip connections)

---

## Examples

### Example 1

**Input:**
```
X = random (5, 64), d_model = 64, n_heads = 8, d_ff = 256
```

**Output:**
```
output shape: (5, 64)  (same as input)
```

**Explanation:** The encoder block transforms representations while preserving shape.

### Example 2

**Input:**
```
X = random (3, 16), d_model = 16, n_heads = 4, d_ff = 64
```

**Output:**
```
Layer norm ensures output has approximately zero mean and unit variance per position
```

**Explanation:** Layer normalization stabilizes the values at each layer.

---

## Approach Hints

1. **Component-wise:** Implement each component separately (MHA, LayerNorm, FFN), then compose
2. **Pre-norm vs Post-norm:** Original paper uses post-norm; pre-norm is an alternative

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Transformer Encoder | O(seq² × d + seq × d × d_ff) | O(seq² + seq × d_ff) |
