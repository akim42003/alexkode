# Bag of Words / TF-IDF

**Difficulty:** Medium
**Category:** NLP

---

## Description

Implement text vectorization methods:

1. **Bag of Words (BoW):** Represent each document as a vector of word counts based on a vocabulary.
2. **TF-IDF (Term Frequency - Inverse Document Frequency):**
   - TF(t, d) = count(t in d) / total_words(d)
   - IDF(t) = log(N / (1 + df(t))) where N = total documents, df(t) = documents containing t
   - TF-IDF(t, d) = TF(t, d) × IDF(t)

### Constraints

- Input is a list of document strings
- Build vocabulary from training documents
- Output is a matrix of shape (n_documents, vocab_size)
- Handle unseen words in new documents (ignore them)

---

## Examples

### Example 1

**Input:**
```
documents = ["the cat sat on the mat",
             "the dog sat on the log"]
```

**Output (BoW):**
```
vocab: {cat:0, dog:1, log:2, mat:3, on:4, sat:5, the:6}
[[1, 0, 0, 1, 1, 1, 2],
 [0, 1, 1, 0, 1, 1, 2]]
```

**Explanation:** Each column counts occurrences of that word. "the" appears twice in each doc.

### Example 2

**Input:**
```
Same documents as above
```

**Output (TF-IDF):**
```
Words unique to one doc ("cat", "mat", "dog", "log") get higher TF-IDF scores.
Common words ("the", "sat", "on") get lower TF-IDF scores.
```

**Explanation:** TF-IDF downweights words that appear in many documents.

---

## Approach Hints

1. **BoW:** Build vocabulary, then count word occurrences per document
2. **TF-IDF:** Compute TF and IDF separately, multiply to get final scores

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| BoW | O(n_docs × doc_length) | O(n_docs × vocab) |
| TF-IDF | O(n_docs × doc_length) | O(n_docs × vocab) |
