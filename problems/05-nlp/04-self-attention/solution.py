"""
Problem: Self-Attention Mechanism
Category: NLP
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Helper: Softmax
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================
# Approach 1: Scaled Dot-Product Attention
# Time Complexity: O(seq^2 * d)
# Space Complexity: O(seq^2 + seq * d)
# ============================================================

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                  V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)

    Returns:
        Tuple of (output, attention_weights)
        output: (seq_len, d_v)
        attention_weights: (seq_len, seq_len)
    """
    d_k = K.shape[-1]

    # QK^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)

    # Softmax over keys dimension (last axis)
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights


def self_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray,
                   W_V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-attention: compute Q, K, V from input X and apply attention.

    Args:
        X: Input embeddings (seq_len, d_model)
        W_Q: Query projection (d_model, d_k)
        W_K: Key projection (d_model, d_k)
        W_V: Value projection (d_model, d_v)

    Returns:
        Tuple of (output, attention_weights)
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    return scaled_dot_product_attention(Q, K, V)


# ============================================================
# Approach 2: Multi-Head Attention
# Time Complexity: O(seq^2 * d)
# Space Complexity: O(h * seq^2 + seq * d)
# ============================================================

class MultiHeadAttention:
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, seed: int = 42):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)

        # Projection matrices for each head
        self.W_Q = rng.randn(d_model, d_model) * scale
        self.W_K = rng.randn(d_model, d_model) * scale
        self.W_V = rng.randn(d_model, d_model) * scale
        self.W_O = rng.randn(d_model, d_model) * scale

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.

        Args:
            X: Input (seq_len, d_model)

        Returns:
            Tuple of (output, attention_weights per head)
        """
        seq_len = X.shape[0]

        # Project to Q, K, V
        Q = X @ self.W_Q  # (seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into heads: (seq_len, d_model) -> (n_heads, seq_len, d_k)
        Q_heads = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Attention per head
        all_outputs = []
        all_weights = []

        for h in range(self.n_heads):
            output_h, weights_h = scaled_dot_product_attention(
                Q_heads[h], K_heads[h], V_heads[h]
            )
            all_outputs.append(output_h)
            all_weights.append(weights_h)

        # Concatenate heads: list of (seq_len, d_k) -> (seq_len, d_model)
        concat = np.concatenate(all_outputs, axis=-1)

        # Final linear projection
        output = concat @ self.W_O

        return output, np.array(all_weights)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Simple self-attention with identity projections
    X1 = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 1, 0, 0]], dtype=np.float64)
    W_I = np.eye(4)

    output1, weights1 = self_attention(X1, W_I, W_I, W_I)
    print(f"Output shape:      {'PASS' if output1.shape == (3, 4) else 'FAIL'} ({output1.shape})")
    print(f"Weights shape:     {'PASS' if weights1.shape == (3, 3) else 'FAIL'} ({weights1.shape})")

    # Attention weights should sum to 1 per row
    row_sums = np.sum(weights1, axis=1)
    print(f"Weights sum to 1:  {'PASS' if np.allclose(row_sums, 1.0) else 'FAIL'}")

    # Weights should be non-negative
    print(f"Weights >= 0:      {'PASS' if np.all(weights1 >= 0) else 'FAIL'}")

    # Test Example 2: Multi-head attention
    X2 = np.random.RandomState(42).randn(5, 8)
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=42)
    output2, weights2 = mha.forward(X2)

    print(f"\nMHA output shape:  {'PASS' if output2.shape == (5, 8) else 'FAIL'} ({output2.shape})")
    print(f"MHA weights shape: {'PASS' if weights2.shape == (2, 5, 5) else 'FAIL'} ({weights2.shape})")

    # Each head's weights should sum to 1
    for h in range(2):
        sums = np.sum(weights2[h], axis=1)
        print(f"Head {h} sums to 1:  {'PASS' if np.allclose(sums, 1.0) else 'FAIL'}")

    # Test: 4 heads
    mha4 = MultiHeadAttention(d_model=8, n_heads=4, seed=0)
    output4, weights4 = mha4.forward(X2)
    print(f"4-head shape:      {'PASS' if output4.shape == (5, 8) else 'FAIL'}")
