"""
Problem: Two-Layer Neural Network
Category: Deep Learning
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# Approach 1: Full Two-Layer NN Class
# Time Complexity: O(N * D * H + N * H * C) per epoch
# Space Complexity: O(N * H + H * C)
# ============================================================

class TwoLayerNN:
    """
    Two-layer neural network: Input -> ReLU -> Softmax -> Cross-Entropy.

    Architecture: (N, D) -> (N, H) -> (N, C)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg: float = 0.0, seed: int = 42):
        """
        Args:
            input_dim: Number of input features (D)
            hidden_dim: Number of hidden units (H)
            output_dim: Number of classes (C)
            reg: L2 regularization strength
            seed: Random seed
        """
        rng = np.random.RandomState(seed)

        # He initialization for ReLU layer
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        # Xavier for output layer
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros(output_dim)
        self.reg = reg

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.

        Args:
            X: Input (N, D)

        Returns:
            Tuple of (probabilities, cache)
        """
        # Hidden layer
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU

        # Output layer
        Z2 = A1 @ self.W2 + self.b2

        # Softmax
        Z2_shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
        exp_Z2 = np.exp(Z2_shifted)
        probs = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

        cache = {"X": X, "Z1": Z1, "A1": A1, "probs": probs}
        return probs, cache

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss with optional L2 regularization.

        Args:
            probs: Softmax probabilities (N, C)
            y: True class indices (N,) with integer labels

        Returns:
            Scalar loss value
        """
        N = probs.shape[0]
        log_probs = -np.log(np.clip(probs[np.arange(N), y], 1e-15, 1.0))
        data_loss = np.mean(log_probs)
        reg_loss = 0.5 * self.reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        """
        Backward pass.

        Args:
            cache: From forward pass
            y: True class indices (N,)

        Returns:
            Dict of gradients {dW1, db1, dW2, db2}
        """
        N = cache["X"].shape[0]
        probs = cache["probs"].copy()

        # Softmax + CE combined gradient
        probs[np.arange(N), y] -= 1
        dZ2 = probs / N

        # Output layer gradients
        dW2 = cache["A1"].T @ dZ2 + self.reg * self.W2
        db2 = np.sum(dZ2, axis=0)

        # Hidden layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0).astype(np.float64)  # ReLU backward

        # Input layer gradients
        dW1 = cache["X"].T @ dZ1 + self.reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 1000, lr: float = 0.1,
              verbose: bool = False) -> List[float]:
        """
        Train the network.

        Args:
            X: Training data (N, D)
            y: Labels (N,) integer class indices
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print loss every 100 epochs

        Returns:
            List of loss values per epoch
        """
        loss_history = []

        for epoch in range(epochs):
            probs, cache = self.forward(X)
            loss = self.compute_loss(probs, y)
            loss_history.append(loss)

            grads = self.backward(cache, y)

            self.W1 -= lr * grads["dW1"]
            self.b1 -= lr * grads["db1"]
            self.W2 -= lr * grads["dW2"]
            self.b2 -= lr * grads["db2"]

            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


# ============================================================
# Approach 2: Functional Interface
# ============================================================

def create_network(D, H, C, seed=42):
    """Create weight dict for a two-layer network."""
    rng = np.random.RandomState(seed)
    return {
        "W1": rng.randn(D, H) * np.sqrt(2.0 / D),
        "b1": np.zeros(H),
        "W2": rng.randn(H, C) * np.sqrt(2.0 / (H + C)),
        "b2": np.zeros(C),
    }


def forward_fn(X, params):
    """Functional forward pass."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    Z2_s = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_z = np.exp(Z2_s)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return probs, {"X": X, "Z1": Z1, "A1": A1, "probs": probs}


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: XOR problem
    print("=== XOR Problem ===")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_xor = np.array([0, 1, 1, 0])

    nn = TwoLayerNN(input_dim=2, hidden_dim=16, output_dim=2, seed=42)
    losses = nn.train(X_xor, y_xor, epochs=2000, lr=0.5)
    acc = nn.accuracy(X_xor, y_xor)
    print(f"XOR accuracy: {acc:.2f}")
    print(f"XOR solved:   {'PASS' if acc == 1.0 else 'FAIL'}")
    print(f"Loss decreased: {'PASS' if losses[-1] < losses[0] else 'FAIL'}")

    # Test 2: Output shapes
    print("\n=== Shape Tests ===")
    nn2 = TwoLayerNN(input_dim=10, hidden_dim=20, output_dim=5)
    X_test = np.random.randn(8, 10)
    probs, cache = nn2.forward(X_test)
    print(f"Probs shape:   {'PASS' if probs.shape == (8, 5) else 'FAIL'}")
    print(f"Probs sum=1:   {'PASS' if np.allclose(probs.sum(axis=1), 1.0) else 'FAIL'}")

    grads = nn2.backward(cache, np.array([0, 1, 2, 3, 4, 0, 1, 2]))
    print(f"dW1 shape:     {'PASS' if grads['dW1'].shape == (10, 20) else 'FAIL'}")
    print(f"dW2 shape:     {'PASS' if grads['dW2'].shape == (20, 5) else 'FAIL'}")
    print(f"db1 shape:     {'PASS' if grads['db1'].shape == (20,) else 'FAIL'}")
    print(f"db2 shape:     {'PASS' if grads['db2'].shape == (5,) else 'FAIL'}")

    # Test 3: Gradient check
    print("\n=== Gradient Check ===")
    nn3 = TwoLayerNN(input_dim=3, hidden_dim=5, output_dim=2, reg=0.1, seed=0)
    X_gc = np.random.randn(4, 3)
    y_gc = np.array([0, 1, 0, 1])

    probs3, cache3 = nn3.forward(X_gc)
    grads3 = nn3.backward(cache3, y_gc)

    eps = 1e-5
    max_err = 0.0
    for name in ["W1", "W2"]:
        W = getattr(nn3, name)
        dW = grads3["d" + name]
        for i in range(min(W.shape[0], 3)):
            for j in range(min(W.shape[1], 3)):
                old = W[i, j]
                W[i, j] = old + eps
                p_plus, _ = nn3.forward(X_gc)
                loss_plus = nn3.compute_loss(p_plus, y_gc)
                W[i, j] = old - eps
                p_minus, _ = nn3.forward(X_gc)
                loss_minus = nn3.compute_loss(p_minus, y_gc)
                W[i, j] = old

                grad_num = (loss_plus - loss_minus) / (2 * eps)
                denom = max(abs(dW[i, j]) + abs(grad_num), 1e-8)
                err = abs(dW[i, j] - grad_num) / denom
                max_err = max(max_err, err)

    print(f"Max rel error: {max_err:.2e}")
    print(f"Gradient check: {'PASS' if max_err < 1e-4 else 'FAIL'}")

    # Test 4: Regularization increases loss
    print("\n=== Regularization ===")
    nn_noreg = TwoLayerNN(input_dim=2, hidden_dim=8, output_dim=2, reg=0.0, seed=42)
    nn_reg = TwoLayerNN(input_dim=2, hidden_dim=8, output_dim=2, reg=1.0, seed=42)
    p1, _ = nn_noreg.forward(X_xor)
    p2, _ = nn_reg.forward(X_xor)
    loss_noreg = nn_noreg.compute_loss(p1, y_xor)
    loss_reg = nn_reg.compute_loss(p2, y_xor)
    print(f"Reg increases loss: {'PASS' if loss_reg > loss_noreg else 'FAIL'}")

    # Test 5: Multi-class
    print("\n=== Multi-class ===")
    N = 300
    X_multi = np.random.randn(N, 4)
    y_multi = np.random.randint(0, 5, N)
    nn_mc = TwoLayerNN(input_dim=4, hidden_dim=32, output_dim=5, seed=42)
    losses_mc = nn_mc.train(X_multi, y_multi, epochs=500, lr=0.1)
    acc_mc = nn_mc.accuracy(X_multi, y_multi)
    print(f"Multi-class acc: {acc_mc:.2f}")
    print(f"Loss decreased:  {'PASS' if losses_mc[-1] < losses_mc[0] else 'FAIL'}")
