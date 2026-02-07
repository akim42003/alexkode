"""
Problem: Tokenization (Word/Character)
Category: NLP
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================
# Approach 1: Manual Tokenization
# Time Complexity: O(n) per string
# Space Complexity: O(n)
# ============================================================

PUNCTUATION = set('.,!?;:()[]{}"\'-')


def clean_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters except punctuation."""
    return text.lower().strip()


def word_tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, separating punctuation as individual tokens.

    Args:
        text: Input string

    Returns:
        List of tokens
    """
    text = clean_text(text)
    tokens = []
    current = []

    for char in text:
        if char in PUNCTUATION:
            if current:
                tokens.append(''.join(current))
                current = []
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)

    if current:
        tokens.append(''.join(current))

    return tokens


def char_tokenize(text: str) -> List[str]:
    """
    Tokenize text into individual characters.

    Args:
        text: Input string

    Returns:
        List of characters
    """
    return list(text)


# ============================================================
# Approach 2: Vocabulary Builder
# Time Complexity: O(total_tokens)
# Space Complexity: O(vocab_size)
# ============================================================

class Vocabulary:
    """Token-to-index vocabulary mapping."""

    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>"]
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

        for i, token in enumerate(self.special_tokens):
            self.token_to_idx[token] = i
            self.idx_to_token[i] = token

    def build(self, texts: List[str]) -> 'Vocabulary':
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings

        Returns:
            self
        """
        for text in texts:
            tokens = word_tokenize(text)
            for token in tokens:
                if token not in self.token_to_idx:
                    idx = len(self.token_to_idx)
                    self.token_to_idx[token] = idx
                    self.idx_to_token[idx] = token
        return self

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        tokens = word_tokenize(text)
        unk_idx = self.token_to_idx.get("<UNK>", 1)
        return [self.token_to_idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens."""
        return [self.idx_to_token.get(i, "<UNK>") for i in indices]

    def __len__(self) -> int:
        return len(self.token_to_idx)


def texts_to_matrix(texts: List[str], vocab: Vocabulary,
                    max_len: int = None) -> np.ndarray:
    """
    Convert texts to padded integer matrix.

    Args:
        texts: List of strings
        vocab: Vocabulary object
        max_len: Maximum sequence length (pad/truncate)

    Returns:
        Integer matrix of shape (n_texts, max_len)
    """
    encoded = [vocab.encode(t) for t in texts]
    if max_len is None:
        max_len = max(len(e) for e in encoded)

    pad_idx = vocab.token_to_idx.get("<PAD>", 0)
    matrix = np.full((len(texts), max_len), pad_idx, dtype=np.int64)

    for i, enc in enumerate(encoded):
        length = min(len(enc), max_len)
        matrix[i, :length] = enc[:length]

    return matrix


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Word tokenization
    text1 = "Hello, world! This is NLP."
    tokens1 = word_tokenize(text1)
    expected1 = ["hello", ",", "world", "!", "this", "is", "nlp", "."]
    print(f"Word tokenize:    {'PASS' if tokens1 == expected1 else 'FAIL'}")
    print(f"  Tokens: {tokens1}")

    # Test: Character tokenization
    chars = char_tokenize("Hi!")
    print(f"Char tokenize:    {'PASS' if chars == ['H', 'i', '!'] else 'FAIL'}")

    # Test Example 2: Vocabulary
    texts = ["the cat sat", "the dog ran"]
    vocab = Vocabulary().build(texts)
    print(f"Vocab size:       {'PASS' if len(vocab) == 7 else 'FAIL'} ({len(vocab)})")
    print(f"PAD=0:            {'PASS' if vocab.token_to_idx['<PAD>'] == 0 else 'FAIL'}")
    print(f"UNK=1:            {'PASS' if vocab.token_to_idx['<UNK>'] == 1 else 'FAIL'}")

    # Test: Encode/Decode
    encoded = vocab.encode("the cat ran")
    decoded = vocab.decode(encoded)
    print(f"Encode/Decode:    {'PASS' if decoded == ['the', 'cat', 'ran'] else 'FAIL'}")

    # Test: Unknown token
    encoded_unk = vocab.encode("the bird flew")
    print(f"UNK handling:     {'PASS' if 1 in encoded_unk else 'FAIL'}")

    # Test: Text to matrix
    matrix = texts_to_matrix(texts, vocab, max_len=4)
    print(f"Matrix shape:     {'PASS' if matrix.shape == (2, 4) else 'FAIL'} ({matrix.shape})")
    print(f"Padding works:    {'PASS' if matrix[0, 3] == 0 else 'FAIL'}")
