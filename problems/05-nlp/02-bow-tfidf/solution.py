"""
Problem: Bag of Words / TF-IDF
Category: NLP
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


# ============================================================
# Helper: Simple tokenizer
# ============================================================

def simple_tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace."""
    return text.lower().split()


# ============================================================
# Approach 1: Bag of Words
# Time Complexity: O(n_docs * avg_doc_length)
# Space Complexity: O(n_docs * vocab_size)
# ============================================================

class BagOfWords:
    """Bag of Words vectorizer."""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}

    def fit(self, documents: List[str]) -> 'BagOfWords':
        """Build vocabulary from documents."""
        word_set = set()
        for doc in documents:
            tokens = simple_tokenize(doc)
            word_set.update(tokens)

        # Sort for deterministic ordering
        for idx, word in enumerate(sorted(word_set)):
            self.vocabulary[word] = idx

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Convert documents to count vectors."""
        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = simple_tokenize(doc)
            counts = Counter(tokens)
            for word, count in counts.items():
                if word in self.vocabulary:
                    matrix[i, self.vocabulary[word]] = count

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


# ============================================================
# Approach 2: TF-IDF
# Time Complexity: O(n_docs * avg_doc_length)
# Space Complexity: O(n_docs * vocab_size)
# ============================================================

class TfIdf:
    """TF-IDF vectorizer."""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = None

    def fit(self, documents: List[str]) -> 'TfIdf':
        """Build vocabulary and compute IDF."""
        word_set = set()
        for doc in documents:
            word_set.update(simple_tokenize(doc))

        for idx, word in enumerate(sorted(word_set)):
            self.vocabulary[word] = idx

        N = len(documents)
        vocab_size = len(self.vocabulary)
        df = np.zeros(vocab_size)

        # Document frequency: how many docs contain each word
        for doc in documents:
            unique_words = set(simple_tokenize(doc))
            for word in unique_words:
                if word in self.vocabulary:
                    df[self.vocabulary[word]] += 1

        # IDF = log(N / (1 + df))
        self.idf = np.log(N / (1 + df))

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Convert documents to TF-IDF vectors."""
        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = simple_tokenize(doc)
            n_tokens = len(tokens)
            if n_tokens == 0:
                continue

            counts = Counter(tokens)
            for word, count in counts.items():
                if word in self.vocabulary:
                    j = self.vocabulary[word]
                    tf = count / n_tokens
                    matrix[i, j] = tf * self.idf[j]

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    docs = ["the cat sat on the mat", "the dog sat on the log"]

    # Test BoW
    bow = BagOfWords()
    bow_matrix = bow.fit_transform(docs)
    vocab = bow.vocabulary
    print(f"Vocabulary: {vocab}")
    print(f"BoW shape:     {'PASS' if bow_matrix.shape == (2, len(vocab)) else 'FAIL'} ({bow_matrix.shape})")

    # "the" appears twice in each doc
    the_idx = vocab["the"]
    print(f"'the' count:   {'PASS' if bow_matrix[0, the_idx] == 2 and bow_matrix[1, the_idx] == 2 else 'FAIL'}")

    # "cat" only in doc 0
    cat_idx = vocab["cat"]
    print(f"'cat' in doc0: {'PASS' if bow_matrix[0, cat_idx] == 1 and bow_matrix[1, cat_idx] == 0 else 'FAIL'}")

    # Test TF-IDF
    tfidf = TfIdf()
    tfidf_matrix = tfidf.fit_transform(docs)
    print(f"\nTF-IDF shape:  {'PASS' if tfidf_matrix.shape == bow_matrix.shape else 'FAIL'}")

    # Common words should have lower TF-IDF than unique words
    sat_idx = vocab["sat"]
    tfidf_cat = tfidf_matrix[0, cat_idx]
    tfidf_sat = tfidf_matrix[0, sat_idx]
    print(f"Unique > common: {'PASS' if tfidf_cat > tfidf_sat else 'FAIL'} (cat={tfidf_cat:.4f}, sat={tfidf_sat:.4f})")

    # Test: Non-negative
    print(f"Non-negative:  {'PASS' if np.all(tfidf_matrix >= 0) else 'FAIL'}")

    # Test: Transform new document
    new_doc = ["the cat ran"]
    new_bow = bow.transform(new_doc)
    print(f"New doc shape: {'PASS' if new_bow.shape == (1, len(vocab)) else 'FAIL'}")
    # "ran" is unknown, should be ignored
    print(f"Cat counted:   {'PASS' if new_bow[0, cat_idx] == 1 else 'FAIL'}")
