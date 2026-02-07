"""
Problem: Decision Tree (ID3/CART)
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Optional, Dict, Any
from collections import Counter


# ============================================================
# Approach 1: Decision Tree with Information Gain / Gini
# Time Complexity: O(n * d * n_log_n * depth)
# Space Complexity: O(n * depth)
# ============================================================

def entropy(y: np.ndarray) -> float:
    """Compute Shannon entropy of a label array."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y.tolist())
    h = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            h -= p * np.log2(p)
    return h


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity of a label array."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y.tolist())
    g = 1.0
    for count in counts.values():
        p = count / n
        g -= p ** 2
    return g


class DecisionTreeNode:
    """A node in the decision tree."""

    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None,
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class label for leaf nodes


class DecisionTree:
    """Decision Tree classifier."""

    def __init__(self, max_depth: int = 10, criterion: str = 'gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == 'gini':
            return gini_impurity(y)
        return entropy(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best feature and threshold to split on."""
        n, d = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)

        for feature in range(d):
            thresholds = np.unique(X[:, feature])
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2

                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted = (n_left * left_impurity + n_right * right_impurity) / n

                gain = parent_impurity - weighted

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively build the tree."""
        # Base cases
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            counts = Counter(y.tolist())
            return DecisionTreeNode(value=counts.most_common(1)[0][0])

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            counts = Counter(y.tolist())
            return DecisionTreeNode(value=counts.most_common(1)[0][0])

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Build the decision tree."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, x: np.ndarray, node: DecisionTreeNode):
        """Predict a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.root) for x in X])


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[2, 3], [1, 1], [3, 2], [6, 5], [7, 8], [8, 6]], dtype=np.float64)
    y1 = np.array([0, 0, 0, 1, 1, 1])

    tree1 = DecisionTree(max_depth=3, criterion='gini').fit(X1, y1)
    preds1 = tree1.predict(X1)
    acc1 = np.mean(preds1 == y1)
    print(f"Test 1 accuracy: {'PASS' if acc1 == 1.0 else 'FAIL'} ({acc1})")

    # Test Example 2
    X2 = np.array([[0], [1], [2], [3]], dtype=np.float64)
    y2 = np.array([0, 0, 1, 1])

    tree2 = DecisionTree(max_depth=2, criterion='entropy').fit(X2, y2)
    preds2 = tree2.predict(X2)
    print(f"Test 2 accuracy: {'PASS' if np.array_equal(preds2, y2) else 'FAIL'}")

    # Test: Entropy and Gini functions
    pure = np.array([1, 1, 1])
    mixed = np.array([0, 1, 0, 1])
    print(f"Entropy pure=0:  {'PASS' if entropy(pure) == 0.0 else 'FAIL'}")
    print(f"Entropy mixed=1: {'PASS' if abs(entropy(mixed) - 1.0) < 1e-10 else 'FAIL'}")
    print(f"Gini pure=0:     {'PASS' if gini_impurity(pure) == 0.0 else 'FAIL'}")
    print(f"Gini mixed=0.5:  {'PASS' if abs(gini_impurity(mixed) - 0.5) < 1e-10 else 'FAIL'}")

    # Test: Both criteria work
    tree_gini = DecisionTree(max_depth=3, criterion='gini').fit(X1, y1)
    tree_ent = DecisionTree(max_depth=3, criterion='entropy').fit(X1, y1)
    print(f"Gini accuracy:   {'PASS' if np.mean(tree_gini.predict(X1) == y1) == 1.0 else 'FAIL'}")
    print(f"Entropy accuracy: {'PASS' if np.mean(tree_ent.predict(X1) == y1) == 1.0 else 'FAIL'}")
