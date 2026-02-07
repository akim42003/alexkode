"""
Problem: Transformer Encoder Block
Category: NLP
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Helper Components
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


# ============================================================
# Layer Normalization
# Time Complexity: O(seq * d)
# Space Complexity: O(d)
# ============================================================

class LayerNorm:
    """Layer Normalization: normalizes across the feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model, dtype=np.float64)
        self.beta = np.zeros(d_model, dtype=np.float64)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        LN(x) = gamma * (x - mean) / (std + eps) + beta

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Normalized output of same shape
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ============================================================
# Multi-Head Attention (for encoder)
# ============================================================

class MultiHeadAttention:
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.RandomState):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.randn(d_model, d_model) * scale
        self.W_K = rng.randn(d_model, d_model) * scale
        self.W_V = rng.randn(d_model, d_model) * scale
        self.W_O = rng.randn(d_model, d_model) * scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Multi-head self-attention forward pass.

        Args:
            X: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        seq_len = X.shape[0]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into heads
        Q_h = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_h = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_h = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Attention per head
        head_outputs = []
        for h in range(self.n_heads):
            scores = Q_h[h] @ K_h[h].T / np.sqrt(self.d_k)
            weights = softmax(scores, axis=-1)
            head_out = weights @ V_h[h]
            head_outputs.append(head_out)

        # Concatenate and project
        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.W_O


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward:
    """Position-wise feed-forward network: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.RandomState):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = rng.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


# ============================================================
# Approach 1: Transformer Encoder Block (Post-Norm)
# Time Complexity: O(seq^2 * d + seq * d * d_ff)
# Space Complexity: O(seq^2 + seq * d_ff)
# ============================================================

class TransformerEncoderBlock:
    """
    Complete Transformer encoder block.
    X -> MHA -> Add & LayerNorm -> FFN -> Add & LayerNorm -> Output
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 42):
        rng = np.random.RandomState(seed)

        self.mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)
        self.ln2 = LayerNorm(d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the encoder block.

        Args:
            X: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        # Multi-head attention + residual + layer norm
        attn_out = self.mha.forward(X)
        X = self.ln1.forward(X + attn_out)

        # Feed-forward + residual + layer norm
        ffn_out = self.ffn.forward(X)
        X = self.ln2.forward(X + ffn_out)

        return X


# ============================================================
# Approach 2: Stacked Encoder (Multiple Blocks)
# ============================================================

class TransformerEncoder:
    """Stack of N encoder blocks."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 d_ff: int, seed: int = 42):
        self.blocks = []
        for i in range(n_layers):
            block = TransformerEncoderBlock(d_model, n_heads, d_ff, seed=seed + i)
            self.blocks.append(block)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all encoder blocks."""
        for block in self.blocks:
            X = block.forward(X)
        return X


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Example 1: Single encoder block
    X1 = np.random.randn(5, 64)
    encoder_block = TransformerEncoderBlock(d_model=64, n_heads=8, d_ff=256, seed=42)
    output1 = encoder_block.forward(X1)

    print(f"Output shape:      {'PASS' if output1.shape == (5, 64) else 'FAIL'} ({output1.shape})")
    print(f"Not same as input: {'PASS' if not np.allclose(output1, X1) else 'FAIL'}")

    # Test: Layer normalization
    ln = LayerNorm(64)
    x_test = np.random.randn(3, 64)
    x_normed = ln.forward(x_test)
    # After layer norm, each position should have approximately mean 0, variance 1
    means = np.mean(x_normed, axis=-1)
    variances = np.var(x_normed, axis=-1)
    print(f"LN mean ≈ 0:       {'PASS' if np.allclose(means, 0, atol=1e-5) else 'FAIL'}")
    print(f"LN var ≈ 1:        {'PASS' if np.allclose(variances, 1, atol=1e-5) else 'FAIL'}")

    # Test Example 2: Smaller model
    X2 = np.random.randn(3, 16)
    encoder_block2 = TransformerEncoderBlock(d_model=16, n_heads=4, d_ff=64, seed=0)
    output2 = encoder_block2.forward(X2)
    print(f"Small model:       {'PASS' if output2.shape == (3, 16) else 'FAIL'}")

    # Test: No NaN or Inf
    print(f"No NaN:            {'PASS' if not np.any(np.isnan(output2)) else 'FAIL'}")
    print(f"No Inf:            {'PASS' if not np.any(np.isinf(output2)) else 'FAIL'}")

    # Test: Stacked encoder
    encoder = TransformerEncoder(n_layers=3, d_model=64, n_heads=8, d_ff=256, seed=42)
    output_stacked = encoder.forward(X1)
    print(f"Stacked shape:     {'PASS' if output_stacked.shape == (5, 64) else 'FAIL'}")
    print(f"Stacked differs:   {'PASS' if not np.allclose(output_stacked, output1) else 'FAIL'}")
