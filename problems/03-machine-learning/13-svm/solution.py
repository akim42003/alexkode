"""
Problem: Support Vector Machine (SVM)
Category: Machine Learning
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# Approach 1: Simplified SMO
# Time Complexity: O(n^2 * iterations)
# Space Complexity: O(n^2) for kernel matrix
# ============================================================

def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """Linear kernel: K(x1, x2) = x1 · x2"""
    return np.dot(x1, x2)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 1.0) -> float:
    """RBF (Gaussian) kernel: K(x1, x2) = exp(-γ||x1-x2||²)"""
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))


class SVM:
    """Support Vector Machine with simplified SMO."""

    def __init__(self, C: float = 1.0, kernel: str = 'linear',
                 gamma: float = 1.0, tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.kernel_name == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, self.gamma)
        raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Precompute kernel matrix for efficiency."""
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._kernel(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    def _predict_raw(self, x: np.ndarray) -> float:
        """Raw prediction (before sign)."""
        result = self.b
        for i in range(len(self.X)):
            if self.alphas[i] > 1e-8:
                result += self.alphas[i] * self.y[i] * self._kernel(self.X[i], x)
        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train SVM using simplified SMO algorithm.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values in {-1, +1}

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.X = X
        self.y = y
        n = X.shape[0]

        self.alphas = np.zeros(n)
        self.b = 0.0
        K = self._compute_kernel_matrix(X)

        for iteration in range(self.max_iter):
            num_changed = 0

            for i in range(n):
                # Compute error for i
                Ei = np.sum(self.alphas * y * K[i]) + self.b - y[i]

                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (y[i] * Ei > self.tol and self.alphas[i] > 0)):

                    # Select j randomly (j != i)
                    j = i
                    while j == i:
                        j = np.random.randint(n)

                    Ej = np.sum(self.alphas * y * K[j]) + self.b - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    # Update bias
                    b1 = (self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i]
                           - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j])
                    b2 = (self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j]
                           - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed += 1

            if num_changed == 0:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([np.sign(self._predict_raw(x)) for x in X])

    def support_vectors(self) -> np.ndarray:
        """Get support vectors (points with alpha > 0)."""
        sv_mask = self.alphas > 1e-8
        return self.X[sv_mask]


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Example 1: Linearly separable
    X1 = np.array([[1, 1], [2, 2], [1, 2], [4, 4], [5, 5], [4, 5]], dtype=np.float64)
    y1 = np.array([-1, -1, -1, 1, 1, 1], dtype=np.float64)

    svm1 = SVM(C=1.0, kernel='linear').fit(X1, y1)
    preds1 = svm1.predict(X1)
    acc1 = np.mean(preds1 == y1)
    print(f"Linear SVM accuracy: {'PASS' if acc1 == 1.0 else 'FAIL'} ({acc1})")

    svs = svm1.support_vectors()
    print(f"Support vectors:     {len(svs)} found")

    # Test: RBF kernel on XOR-like data
    X_xor = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float64)
    y_xor = np.array([-1, -1, 1, 1], dtype=np.float64)

    svm_rbf = SVM(C=10.0, kernel='rbf', gamma=5.0, max_iter=500).fit(X_xor, y_xor)
    preds_xor = svm_rbf.predict(X_xor)
    acc_xor = np.mean(preds_xor == y_xor)
    print(f"RBF XOR accuracy:    {'PASS' if acc_xor == 1.0 else 'FAIL'} ({acc_xor})")

    # Test: Linear kernel on larger data
    np.random.seed(42)
    X_large = np.vstack([np.random.randn(20, 2) - 2, np.random.randn(20, 2) + 2])
    y_large = np.array([-1]*20 + [1]*20, dtype=np.float64)
    svm_large = SVM(C=1.0, kernel='linear', max_iter=200).fit(X_large, y_large)
    acc_large = np.mean(svm_large.predict(X_large) == y_large)
    print(f"Large linear SVM:    {'PASS' if acc_large > 0.9 else 'FAIL'} ({acc_large})")
