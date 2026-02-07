# K-Means Clustering

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement K-Means clustering using Lloyd's algorithm:

1. Initialize k centroids (random selection or k-means++)
2. **Assign:** Each point to its nearest centroid
3. **Update:** Recompute each centroid as the mean of its assigned points
4. Repeat until convergence (assignments don't change) or max iterations

### Constraints

- X has shape (n_samples, n_features)
- k ≥ 1 and k ≤ n_samples
- Use Euclidean distance
- Return cluster assignments and final centroids

---

## Examples

### Example 1

**Input:**
```
X = [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]]
k = 2
```

**Output:**
```
centroids ≈ [[1.25, 1.5], [4.0, 5.25]]
labels = [0, 0, 1, 1, 1, 1]  (or equivalent relabeling)
```

**Explanation:** Two natural clusters: bottom-left and upper-right.

### Example 2

**Input:**
```
X = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]
k = 2
```

**Output:**
```
centroids ≈ [[0.33, 0.33], [10.33, 10.33]]
labels = [0, 0, 0, 1, 1, 1]
```

**Explanation:** Two clearly separated clusters.

---

## Approach Hints

1. **Random Initialization:** Pick k random data points as initial centroids
2. **K-Means++ Initialization:** Choose centroids probabilistically to spread them out — better convergence

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Lloyd's Algorithm | O(n × k × d × iterations) | O(n + k × d) |
