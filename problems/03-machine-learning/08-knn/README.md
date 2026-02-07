# K-Nearest Neighbors (KNN)

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement K-Nearest Neighbors for classification. Given training data, classify new points by finding the k closest training points and taking a majority vote.

**Steps:**
1. Compute distances from query point to all training points
2. Select the k nearest neighbors
3. Return the most common class among them

Use Euclidean distance: `d(x, y) = √Σ(x_i - y_i)²`

### Constraints

- X_train has shape (n_train, n_features), X_test has shape (n_test, n_features)
- k is a positive odd integer (to avoid ties) with k ≤ n_train
- Handle multi-class classification

---

## Examples

### Example 1

**Input:**
```
X_train = [[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[4, 4]]
k = 3
```

**Output:**
```
predictions = [0]
```

**Explanation:** The 3 nearest points to (4,4) are (3,3), (2,2), (6,6) with labels [0, 0, 1]. Majority vote → 0.

### Example 2

**Input:**
```
X_train = [[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[2, 2], [4, 4]]
k = 3
```

**Output:**
```
predictions = [0, 1]
```

**Explanation:** (2,2) is closer to cluster 0; (4,4) is closer to cluster 1.

---

## Approach Hints

1. **Brute Force:** Compute all pairwise distances, sort, pick top k
2. **Distance Matrix:** Vectorize using broadcasting for efficient pairwise distance computation

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Brute Force | O(n_test × n_train × d + n_test × n_train × log(k)) | O(n_train) |
