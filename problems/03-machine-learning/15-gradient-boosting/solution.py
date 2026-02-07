"""
Problem: Gradient Boosting (Basic)
Category: Machine Learning
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# Helper: Simple Regression Tree (stump or shallow tree)
# ============================================================

class RegressionTree:
    """Simple regression tree for gradient boosting."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        best_mse = float('inf')
        best_feat = None
        best_thresh = None

        for feat in range(d):
            vals = np.unique(X[:, feat])
            for i in range(len(vals) - 1):
                thresh = (vals[i] + vals[i+1]) / 2
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                mse = (np.sum((y_left - np.mean(y_left))**2) +
                       np.sum((y_right - np.mean(y_right))**2))

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> 'RegressionTree':
        if depth >= self.max_depth or len(y) < 2 or np.std(y) < 1e-10:
            self.value = np.mean(y)
            return self

        feat, thresh = self._best_split(X, y)
        if feat is None:
            self.value = np.mean(y)
            return self

        self.feature = feat
        self.threshold = thresh

        left_mask = X[:, feat] <= thresh
        self.left = RegressionTree(self.max_depth)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)

        self.right = RegressionTree(self.max_depth)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

        return self

    def predict_one(self, x: np.ndarray) -> float:
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])


# ============================================================
# Approach 1: Gradient Boosting for Regression
# Time Complexity: O(n_estimators * n * d * depth)
# Space Complexity: O(n_estimators * nodes)
# ============================================================

class GradientBoosting:
    """Gradient Boosting regressor."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[RegressionTree] = []
        self.initial_prediction = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoosting':
        """
        Train gradient boosting by sequentially fitting trees to residuals.

        Args:
            X: Features (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        # Initialize with mean
        self.initial_prediction = np.mean(y)
        current_pred = np.full(n, self.initial_prediction)

        self.trees = []
        self.loss_history = []

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient of MSE)
            residuals = y - current_pred

            # Record loss
            mse = np.mean(residuals ** 2)
            self.loss_history.append(mse)

            # Fit a tree to residuals
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_pred = tree.predict(X)
            current_pred += self.learning_rate * tree_pred

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by summing initial prediction + all tree contributions."""
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y1 = np.array([1.0, 2.1, 2.9, 4.2, 4.8])

    gb1 = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=2).fit(X1, y1)
    preds1 = gb1.predict(X1)
    mse1 = np.mean((preds1 - y1) ** 2)
    print(f"Test 1 MSE < 0.1:  {'PASS' if mse1 < 0.1 else 'FAIL'} ({mse1:.4f})")

    # Test Example 2
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y2 = np.array([3, 7, 11, 15], dtype=np.float64)

    gb2 = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=2).fit(X2, y2)
    preds2 = gb2.predict(X2)
    mse2 = np.mean((preds2 - y2) ** 2)
    print(f"Test 2 MSE < 0.5:  {'PASS' if mse2 < 0.5 else 'FAIL'} ({mse2:.4f})")

    # Test: Loss decreases over iterations
    decreasing = gb1.loss_history[0] > gb1.loss_history[-1]
    print(f"Loss decreases:    {'PASS' if decreasing else 'FAIL'}")

    # Test: More estimators = better fit
    gb_few = GradientBoosting(n_estimators=5, learning_rate=0.1, max_depth=2).fit(X1, y1)
    gb_many = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=2).fit(X1, y1)
    mse_few = np.mean((gb_few.predict(X1) - y1) ** 2)
    mse_many = np.mean((gb_many.predict(X1) - y1) ** 2)
    print(f"More trees better: {'PASS' if mse_many < mse_few else 'FAIL'} ({mse_few:.4f} → {mse_many:.4f})")
