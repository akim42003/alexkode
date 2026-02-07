"""
Problem: Random Forest
Category: Machine Learning
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Optional, List
from collections import Counter


# ============================================================
# Helper: Simple Decision Tree (for use within the forest)
# ============================================================

class SimpleTree:
    """A simple decision tree for use within Random Forest."""

    def __init__(self, max_depth: int = 5, max_features: Optional[int] = None,
                 rng: Optional[np.random.RandomState] = None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = rng or np.random.RandomState()
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = Counter(y.tolist())
        n = len(y)
        return 1.0 - sum((c/n)**2 for c in counts.values())

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        # Random feature selection
        if self.max_features and self.max_features < d:
            feature_indices = self.rng.choice(d, self.max_features, replace=False)
        else:
            feature_indices = np.arange(d)

        best_gain = -1
        best_feat = None
        best_thresh = None
        parent_gini = self._gini(y)

        for feat in feature_indices:
            vals = np.unique(X[:, feat])
            for i in range(len(vals) - 1):
                thresh = (vals[i] + vals[i+1]) / 2
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                gain = parent_gini - (n_l * self._gini(y[left_mask]) + n_r * self._gini(y[right_mask])) / n

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> 'SimpleTree':
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            self.value = Counter(y.tolist()).most_common(1)[0][0]
            return self

        feat, thresh = self._best_split(X, y)
        if feat is None:
            self.value = Counter(y.tolist()).most_common(1)[0][0]
            return self

        self.feature = feat
        self.threshold = thresh

        left_mask = X[:, feat] <= thresh
        self.left = SimpleTree(self.max_depth, self.max_features, self.rng)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)

        self.right = SimpleTree(self.max_depth, self.max_features, self.rng)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

        return self

    def predict_one(self, x: np.ndarray):
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)


# ============================================================
# Approach 1: Random Forest with Bagging
# Time Complexity: O(n_trees * n * d_sub * n_log_n * depth)
# Space Complexity: O(n_trees * nodes)
# ============================================================

class RandomForest:
    """Random Forest classifier."""

    def __init__(self, n_trees: int = 10, max_depth: int = 5,
                 max_features: Optional[int] = None, seed: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees: List[SimpleTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Train the random forest using bootstrap aggregating (bagging).

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)

        max_feat = self.max_features or max(1, int(np.sqrt(d)))

        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = rng.choice(n, n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree_rng = np.random.RandomState(rng.randint(0, 2**31))
            tree = SimpleTree(max_depth=self.max_depth, max_features=max_feat, rng=tree_rng)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels by majority vote across all trees."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [tree.predict_one(x) for tree in self.trees]
            majority = Counter(votes).most_common(1)[0][0]
            predictions.append(majority)
        return np.array(predictions)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[1, 1], [1, 2], [2, 1], [5, 5], [6, 5], [5, 6], [2, 5], [5, 2]], dtype=np.float64)
    y1 = np.array([0, 0, 0, 1, 1, 1, 0, 1])

    rf = RandomForest(n_trees=20, max_depth=3, seed=42).fit(X1, y1)
    preds1 = rf.predict(X1)
    acc1 = np.mean(preds1 == y1)
    print(f"RF train accuracy: {'PASS' if acc1 >= 0.875 else 'FAIL'} ({acc1})")

    # Test: Well-separated clusters
    X2 = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]], dtype=np.float64)
    y2 = np.array([0, 0, 0, 1, 1, 1])
    rf2 = RandomForest(n_trees=10, max_depth=3, seed=42).fit(X2, y2)
    preds2 = rf2.predict(X2)
    print(f"Separated data:    {'PASS' if np.mean(preds2 == y2) == 1.0 else 'FAIL'}")

    # Test: Larger random data
    np.random.seed(42)
    X3 = np.vstack([np.random.randn(30, 3) - 1, np.random.randn(30, 3) + 1])
    y3 = np.array([0]*30 + [1]*30)
    rf3 = RandomForest(n_trees=30, max_depth=5, seed=42).fit(X3, y3)
    acc3 = np.mean(rf3.predict(X3) == y3)
    print(f"Larger data acc:   {'PASS' if acc3 > 0.9 else 'FAIL'} ({acc3})")

    print(f"Number of trees:   {'PASS' if len(rf3.trees) == 30 else 'FAIL'}")
