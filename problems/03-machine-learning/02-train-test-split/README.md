# Train-Test Split

**Difficulty:** Easy
**Category:** Machine Learning

---

## Description

Implement a function that splits a dataset into training and testing subsets. The data should be shuffled randomly before splitting.

Given features X and labels y, split them into X_train, X_test, y_train, y_test based on a specified test size ratio.

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,)
- test_size is a float in (0, 1) representing the fraction for testing
- Shuffling must maintain correspondence between X and y
- Optional random_seed for reproducibility

---

## Examples

### Example 1

**Input:**
```
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]
test_size = 0.4
random_seed = 42
```

**Output:**
```
X_train shape: (3, 2), X_test shape: (2, 2)
y_train shape: (3,), y_test shape: (2,)
```

**Explanation:** 40% of 5 samples = 2 test samples, 3 training samples. Data is shuffled before splitting.

### Example 2

**Input:**
```
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
test_size = 0.2
```

**Output:**
```
X_train shape: (8, 1), X_test shape: (2, 1)
```

**Explanation:** 20% of 10 = 2 test samples.

---

## Approach Hints

1. **Index Shuffling:** Generate shuffled indices, then slice at the split point
2. **Fisher-Yates Shuffle:** Implement the shuffle algorithm from scratch, then split

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Index Shuffle | O(n) | O(n) |
