# Tokenization (Word/Character)

**Difficulty:** Easy
**Category:** NLP

---

## Description

Implement text tokenization at two levels:

1. **Word Tokenization:** Split text into words. Handle punctuation (separate it from words), whitespace, and basic contractions.
2. **Character Tokenization:** Split text into individual characters.
3. **Vocabulary Builder:** Create a mapping from tokens to integer indices.

Also implement basic text cleaning: lowercase conversion, removal of special characters.

### Constraints

- Input is a string of text
- Handle punctuation marks: `.`, `,`, `!`, `?`, `;`, `:` (separate them as tokens)
- Vocabulary indices start at 0
- Include special tokens: `<PAD>` (index 0), `<UNK>` (index 1)

---

## Examples

### Example 1

**Input:**
```
text = "Hello, world! This is NLP."
```

**Output (Word Tokenization):**
```
["hello", ",", "world", "!", "this", "is", "nlp", "."]
```

**Explanation:** Text is lowercased, punctuation is separated as its own tokens.

### Example 2

**Input:**
```
texts = ["the cat sat", "the dog ran"]
```

**Output (Vocabulary):**
```
{"<PAD>": 0, "<UNK>": 1, "the": 2, "cat": 3, "sat": 4, "dog": 5, "ran": 6}
```

**Explanation:** Vocabulary built from all unique tokens across documents.

---

## Approach Hints

1. **Regex-based:** Use character checks to split on whitespace and punctuation boundaries
2. **Character-by-character:** Iterate and group alphanumeric runs vs punctuation

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Word Tokenization | O(n) | O(n) |
| Vocabulary Build | O(n × docs) | O(vocab_size) |
