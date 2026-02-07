# Word2Vec (Skip-gram)

**Difficulty:** Medium
**Category:** NLP

---

## Description

Implement the Word2Vec Skip-gram model with negative sampling.

**Skip-gram Objective:** Given a center word, predict its context words.

**Steps:**
1. Build vocabulary and word-index mappings from a corpus
2. Generate training pairs: (center_word, context_word) from sliding windows
3. Train with negative sampling: for each positive pair, sample k negative words
4. Learn two embedding matrices: W_in (center embeddings) and W_out (context embeddings)

### Constraints

- embedding_dim typically 50-300 (use small values for testing)
- window_size: number of words on each side of the center word
- n_negative: number of negative samples per positive pair (typically 5-20)
- Learning via gradient descent on the negative sampling objective

---

## Examples

### Example 1

**Input:**
```
corpus = "the king loves the queen and the queen loves the king"
embedding_dim = 10, window_size = 2
```

**Output:**
```
Word embeddings of shape (vocab_size, embedding_dim)
Similar words have closer embeddings (cosine similarity)
```

**Explanation:** "king" and "queen" should have similar embeddings since they appear in similar contexts.

### Example 2

**Input:**
```
words = ["king", "queen"]
```

**Output:**
```
cosine_similarity(embed("king"), embed("queen")) > 0  (positive similarity)
```

**Explanation:** Words in similar contexts get similar vector representations.

---

## Approach Hints

1. **Negative Sampling Loss:** For positive pair: log σ(v_c · v_w), for negatives: Σ log σ(-v_c · v_n)
2. **Subsampling:** Downsample frequent words to improve quality

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Skip-gram + Neg Sampling | O(corpus_len × window × (1 + n_neg) × d) | O(vocab × d) |
