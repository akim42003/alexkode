"""
Problem: Word2Vec (Skip-gram)
Category: NLP
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


# ============================================================
# Approach 1: Skip-gram with Negative Sampling
# Time Complexity: O(corpus_len * window * (1 + n_neg) * d)
# Space Complexity: O(vocab * d)
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    """Word2Vec Skip-gram with negative sampling."""

    def __init__(self, embedding_dim: int = 50, window_size: int = 2,
                 n_negative: int = 5, lr: float = 0.01, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.n_negative = n_negative
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.W_in = None   # Center word embeddings
        self.W_out = None  # Context word embeddings
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0

    def _build_vocab(self, tokens: List[str]) -> None:
        """Build vocabulary from tokens."""
        word_counts = Counter(tokens)
        for idx, (word, _) in enumerate(word_counts.most_common()):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.word_to_idx)

    def _generate_pairs(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """Generate (center, context) training pairs from token sequence."""
        pairs = []
        token_ids = [self.word_to_idx[t] for t in tokens if t in self.word_to_idx]

        for i in range(len(token_ids)):
            for j in range(max(0, i - self.window_size),
                           min(len(token_ids), i + self.window_size + 1)):
                if i != j:
                    pairs.append((token_ids[i], token_ids[j]))
        return pairs

    def _sample_negatives(self, positive_word: int, k: int) -> List[int]:
        """Sample k negative word indices (not equal to positive_word)."""
        negatives = []
        while len(negatives) < k:
            neg = self.rng.randint(0, self.vocab_size)
            if neg != positive_word:
                negatives.append(neg)
        return negatives

    def fit(self, corpus: str, n_epochs: int = 50) -> 'Word2Vec':
        """
        Train Word2Vec on a text corpus.

        Args:
            corpus: Text string
            n_epochs: Number of training epochs

        Returns:
            self
        """
        tokens = corpus.lower().split()
        self._build_vocab(tokens)

        # Initialize embeddings
        scale = 0.5 / self.embedding_dim
        self.W_in = self.rng.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_out = self.rng.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))

        # Generate training pairs
        pairs = self._generate_pairs(tokens)

        for epoch in range(n_epochs):
            total_loss = 0.0
            self.rng.shuffle(pairs)

            for center_idx, context_idx in pairs:
                # Positive sample
                v_center = self.W_in[center_idx]
                v_context = self.W_out[context_idx]

                score = np.dot(v_center, v_context)
                sig = sigmoid(score)
                loss = -np.log(sig + 1e-10)

                # Gradient for positive pair
                grad_in = (sig - 1) * v_context
                grad_out = (sig - 1) * v_center

                self.W_in[center_idx] -= self.lr * grad_in
                self.W_out[context_idx] -= self.lr * grad_out

                # Negative samples
                negatives = self._sample_negatives(context_idx, self.n_negative)
                for neg_idx in negatives:
                    v_neg = self.W_out[neg_idx]
                    score_neg = np.dot(v_center, v_neg)
                    sig_neg = sigmoid(score_neg)
                    loss += -np.log(1 - sig_neg + 1e-10)

                    grad_in_neg = sig_neg * v_neg
                    grad_out_neg = sig_neg * v_center

                    self.W_in[center_idx] -= self.lr * grad_in_neg
                    self.W_out[neg_idx] -= self.lr * grad_out_neg

                total_loss += loss

        return self

    def get_embedding(self, word: str) -> np.ndarray:
        """Get the embedding vector for a word."""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.W_in[self.word_to_idx[word]]

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar words by cosine similarity."""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")

        target = self.get_embedding(word)
        target_norm = target / (np.linalg.norm(target) + 1e-10)

        similarities = []
        for idx in range(self.vocab_size):
            if self.idx_to_word[idx] == word:
                continue
            vec = self.W_in[idx]
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            sim = np.dot(target_norm, vec_norm)
            similarities.append((self.idx_to_word[idx], sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    corpus = "the king loves the queen and the queen loves the king " * 20
    corpus += "the man saw the woman and the woman saw the man " * 20

    w2v = Word2Vec(embedding_dim=20, window_size=2, n_negative=5, lr=0.025, seed=42)
    w2v.fit(corpus, n_epochs=30)

    print(f"Vocab size:        {w2v.vocab_size}")
    print(f"Embedding shape:   {'PASS' if w2v.W_in.shape == (w2v.vocab_size, 20) else 'FAIL'}")

    # Test embeddings exist
    king_emb = w2v.get_embedding("king")
    queen_emb = w2v.get_embedding("queen")
    print(f"Embedding dim:     {'PASS' if len(king_emb) == 20 else 'FAIL'}")

    # Test cosine similarity
    sim_kq = cosine_similarity(king_emb, queen_emb)
    print(f"king-queen sim:    {sim_kq:.4f}")

    # Test most similar
    similar = w2v.most_similar("king", top_k=3)
    print(f"Similar to 'king': {similar}")
    print(f"Returns list:      {'PASS' if len(similar) > 0 else 'FAIL'}")

    # Test unknown word
    try:
        w2v.get_embedding("xyz")
        print("Unknown word:      FAIL - no exception")
    except KeyError:
        print("Unknown word:      PASS")
