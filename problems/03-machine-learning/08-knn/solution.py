"""
Problem: K-Nearest Neighbors (KNN)
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Optional
from collections import Counter


# ============================================================
# Approach 1: Brute Force with Loop
# Time Complexity: O(n_test * n_train * d)
# Space Complexity: O(n_train)
# ============================================================

def knn_predict_loop(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, k: int = 3) -> np.ndarray:
    """
    KNN classification using brute-force distance computation.

    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training labels (n_train,)
        X_test: Test features (n_test, n_features)
        k: Number of neighbors

    Returns:
        Predicted labels for X_test
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    predictions = np.zeros(n_test, dtype=y_train.dtype)

    for i in range(n_test):
        # Compute distances to all training points
        distances = np.zeros(n_train)
        for j in range(n_train):
            diff = X_test[i] - X_train[j]
            distances[j] = np.sqrt(np.sum(diff ** 2))

        # Find k nearest
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]

        # Majority vote
        counts = Counter(nearest_labels.tolist())
        predictions[i] = counts.most_common(1)[0][0]

    return predictions


# ============================================================
# Approach 2: Vectorized Distance Matrix
# Time Complexity: O(n_test * n_train * d)
# Space Complexity: O(n_test * n_train)
# ============================================================

def knn_predict_vectorized(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, k: int = 3) -> np.ndarray:
    """
    KNN using vectorized pairwise distance computation.

    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training labels (n_train,)
        X_test: Test features (n_test, n_features)
        k: Number of neighbors

    Returns:
        Predicted labels for X_test
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    # Compute pairwise distances using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    train_sq = np.sum(X_train ** 2, axis=1)  # (n_train,)
    test_sq = np.sum(X_test ** 2, axis=1)    # (n_test,)
    cross = X_test @ X_train.T               # (n_test, n_train)

    # Distances: (n_test, n_train)
    dist_sq = test_sq[:, np.newaxis] + train_sq[np.newaxis, :] - 2 * cross
    dist_sq = np.maximum(dist_sq, 0)  # numerical safety
    distances = np.sqrt(dist_sq)

    # For each test point, find k nearest
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test, dtype=y_train.dtype)

    for i in range(n_test):
        nearest_idx = np.argsort(distances[i])[:k]
        nearest_labels = y_train[nearest_idx]
        counts = Counter(nearest_labels.tolist())
        predictions[i] = counts.most_common(1)[0][0]

    return predictions


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X_train1 = np.array([[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8]], dtype=np.float64)
    y_train1 = np.array([0, 0, 0, 1, 1, 1])
    X_test1 = np.array([[4, 4]], dtype=np.float64)

    for name, fn in [("Loop", knn_predict_loop), ("Vectorized", knn_predict_vectorized)]:
        pred = fn(X_train1, y_train1, X_test1, k=3)
        print(f"Test 1 ({name}):     {'PASS' if pred[0] == 0 else 'FAIL'} (predicted {pred[0]})")

    # Test Example 2
    X_train2 = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=np.float64)
    y_train2 = np.array([0, 0, 0, 1, 1, 1])
    X_test2 = np.array([[2, 2], [4, 4]], dtype=np.float64)

    for name, fn in [("Loop", knn_predict_loop), ("Vectorized", knn_predict_vectorized)]:
        pred = fn(X_train2, y_train2, X_test2, k=3)
        print(f"Test 2 ({name}):     {'PASS' if pred[0] == 0 and pred[1] == 1 else 'FAIL'} ({pred})")

    # Test: k=1 should memorize training data
    pred_k1 = knn_predict_vectorized(X_train1, y_train1, X_train1, k=1)
    acc_k1 = np.mean(pred_k1 == y_train1)
    print(f"k=1 memorization:   {'PASS' if acc_k1 == 1.0 else 'FAIL'}")

    # Test: Both approaches give same result
    pred_loop = knn_predict_loop(X_train2, y_train2, X_test2, k=3)
    pred_vec = knn_predict_vectorized(X_train2, y_train2, X_test2, k=3)
    print(f"Approaches match:   {'PASS' if np.array_equal(pred_loop, pred_vec) else 'FAIL'}")
