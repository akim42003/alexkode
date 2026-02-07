"""
Problem: Positional Encoding
Category: NLP
Difficulty: Hard

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Sinusoidal Positional Encoding
# Time Complexity: O(seq_len * d_model)
# Space Complexity: O(seq_len * d_model)
# ============================================================

def sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encoding as in "Attention is All You Need."
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model/embedding dimension

    Returns:
        Positional encoding matrix of shape (max_seq_len, d_model)
    """
    PE = np.zeros((max_seq_len, d_model), dtype=np.float64)

    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            denominator = 10000.0 ** (i / d_model)
            PE[pos, i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denominator)

    return PE


def sinusoidal_positional_encoding_vectorized(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Vectorized version of sinusoidal positional encoding.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix of shape (max_seq_len, d_model)
    """
    PE = np.zeros((max_seq_len, d_model), dtype=np.float64)

    positions = np.arange(max_seq_len)[:, np.newaxis]  # (seq, 1)
    dim_indices = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d/2)

    angles = positions / (10000.0 ** (dim_indices / d_model))

    PE[:, 0::2] = np.sin(angles)
    if d_model > 1:
        PE[:, 1::2] = np.cos(angles[:, :d_model // 2 + d_model % 2 - (1 if d_model % 2 == 0 else 0)])

    return PE


# ============================================================
# Approach 2: Learned Positional Embeddings
# Time Complexity: O(seq_len * d_model)
# Space Complexity: O(max_seq_len * d_model)
# ============================================================

class LearnedPositionalEncoding:
    """Trainable positional embeddings."""

    def __init__(self, max_seq_len: int, d_model: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(max_seq_len, d_model) * 0.02

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional embeddings for given sequence length.

        Args:
            seq_len: Actual sequence length (≤ max_seq_len)

        Returns:
            Positional embeddings of shape (seq_len, d_model)
        """
        return self.embeddings[:seq_len]


def add_positional_encoding(X: np.ndarray, PE: np.ndarray) -> np.ndarray:
    """
    Add positional encoding to input embeddings.

    Args:
        X: Input embeddings (seq_len, d_model)
        PE: Positional encoding matrix (max_seq_len, d_model)

    Returns:
        X + PE[:seq_len] of shape (seq_len, d_model)
    """
    seq_len = X.shape[0]
    return X + PE[:seq_len]


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Small PE
    PE1 = sinusoidal_positional_encoding(4, 6)
    print(f"PE shape:          {'PASS' if PE1.shape == (4, 6) else 'FAIL'} ({PE1.shape})")

    # At position 0: sin(0)=0, cos(0)=1
    print(f"PE[0] even=0:      {'PASS' if np.allclose(PE1[0, 0::2], 0) else 'FAIL'}")
    print(f"PE[0] odd=1:       {'PASS' if np.allclose(PE1[0, 1::2], 1) else 'FAIL'}")

    # Test: Vectorized matches loop version
    PE_vec = sinusoidal_positional_encoding_vectorized(4, 6)
    PE_loop = sinusoidal_positional_encoding(4, 6)
    print(f"Vec matches loop:  {'PASS' if np.allclose(PE_vec, PE_loop) else 'FAIL'}")

    # Test: Different positions have different encodings
    print(f"Positions differ:  {'PASS' if not np.allclose(PE1[0], PE1[1]) else 'FAIL'}")

    # Test: Larger dimension
    PE2 = sinusoidal_positional_encoding(100, 64)
    print(f"Large PE shape:    {'PASS' if PE2.shape == (100, 64) else 'FAIL'}")

    # Values should be in [-1, 1]
    print(f"Values in [-1,1]:  {'PASS' if np.all(PE2 >= -1) and np.all(PE2 <= 1) else 'FAIL'}")

    # Test: Adding to embeddings
    np.random.seed(42)
    X = np.random.randn(10, 64)
    X_pos = add_positional_encoding(X, PE2)
    print(f"Add PE shape:      {'PASS' if X_pos.shape == X.shape else 'FAIL'}")
    print(f"Values changed:    {'PASS' if not np.allclose(X, X_pos) else 'FAIL'}")

    # Test: Learned positional encoding
    lpe = LearnedPositionalEncoding(100, 64)
    lpe_out = lpe.forward(10)
    print(f"Learned PE shape:  {'PASS' if lpe_out.shape == (10, 64) else 'FAIL'}")
