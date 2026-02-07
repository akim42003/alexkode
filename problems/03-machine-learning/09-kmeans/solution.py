"""
Problem: K-Means Clustering
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Lloyd's Algorithm with Random Initialization
# Time Complexity: O(n * k * d * iterations)
# Space Complexity: O(n + k * d)
# ============================================================

def kmeans_random(X: np.ndarray, k: int, max_iter: int = 100,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means with random centroid initialization.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (labels, centroids)
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.RandomState(seed)

    # Random initialization: pick k random data points
    indices = rng.choice(n, k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign step: find nearest centroid for each point
        new_labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            min_dist = float('inf')
            for c in range(k):
                dist = np.sum((X[i] - centroids[c]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    new_labels[i] = c

        # Check convergence
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update step: recompute centroids
        for c in range(k):
            mask = labels == c
            if np.sum(mask) > 0:
                centroids[c] = np.mean(X[mask], axis=0)

    return labels, centroids


# ============================================================
# Approach 2: K-Means++ Initialization
# Time Complexity: O(n * k * d * iterations)
# Space Complexity: O(n + k * d)
# ============================================================

def kmeans_plus_init(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    K-Means++ initialization: spread initial centroids apart.

    Args:
        X: Data matrix
        k: Number of clusters
        rng: Random state

    Returns:
        Initial centroids of shape (k, d)
    """
    n, d = X.shape
    centroids = np.zeros((k, d), dtype=np.float64)

    # First centroid: random
    centroids[0] = X[rng.randint(n)]

    for c in range(1, k):
        # Compute squared distances to nearest existing centroid
        dists = np.full(n, float('inf'))
        for j in range(c):
            d_j = np.sum((X - centroids[j]) ** 2, axis=1)
            dists = np.minimum(dists, d_j)

        # Choose next centroid proportional to distance squared
        probs = dists / np.sum(dists)
        idx = rng.choice(n, p=probs)
        centroids[c] = X[idx]

    return centroids


def kmeans_plusplus(X: np.ndarray, k: int, max_iter: int = 100,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means with k-means++ initialization.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (labels, centroids)
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.RandomState(seed)

    centroids = kmeans_plus_init(X, k, rng)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign
        new_labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            dists = np.sum((centroids - X[i]) ** 2, axis=1)
            new_labels[i] = np.argmin(dists)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update
        for c in range(k):
            mask = labels == c
            if np.sum(mask) > 0:
                centroids[c] = np.mean(X[mask], axis=0)

    return labels, centroids


def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute total within-cluster sum of squared distances."""
    total = 0.0
    for i in range(len(X)):
        total += np.sum((X[i] - centroids[labels[i]]) ** 2)
    return total


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    X1 = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]], dtype=np.float64)
    labels1, centroids1 = kmeans_random(X1, k=2)
    print(f"Test 1 labels:   {labels1}")
    # Check that first two points are in same cluster and last four together (or vice versa)
    same_cluster = (labels1[0] == labels1[1]) and (labels1[2] == labels1[3] == labels1[4] == labels1[5])
    print(f"Test 1 correct:  {'PASS' if same_cluster else 'FAIL'}")

    # Test Example 2: Clearly separated
    X2 = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]], dtype=np.float64)
    labels2, centroids2 = kmeans_plusplus(X2, k=2)
    same_near = (labels2[0] == labels2[1] == labels2[2])
    same_far = (labels2[3] == labels2[4] == labels2[5])
    diff = labels2[0] != labels2[3]
    print(f"Test 2 separated: {'PASS' if same_near and same_far and diff else 'FAIL'}")

    # Test: Inertia should be low for well-separated clusters
    iner = inertia(X2, labels2, centroids2)
    print(f"Inertia (low):   {'PASS' if iner < 10 else 'FAIL'} ({iner:.4f})")

    # Test: k=1 should assign all to same cluster
    labels_k1, _ = kmeans_random(X2, k=1)
    print(f"k=1 all same:    {'PASS' if len(np.unique(labels_k1)) == 1 else 'FAIL'}")
