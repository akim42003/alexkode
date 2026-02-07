# Random Forest

**Difficulty:** Hard
**Category:** Machine Learning

---

## Description

Implement a Random Forest classifier. Random Forest is an ensemble method that builds multiple decision trees and combines their predictions through majority voting.

**Key components:**
1. **Bootstrap Sampling:** Each tree trains on a random sample (with replacement) of the data
2. **Random Feature Selection:** At each split, consider only a random subset of features
3. **Majority Voting:** Final prediction is the most common class among all tree predictions

### Constraints

- X has shape (n_samples, n_features)
- y has shape (n_samples,)
- n_trees ≥ 1, max_depth ≥ 1
- max_features typically = √(n_features) for classification

---

## Examples

### Example 1

**Input:**
```
X = [[1, 1], [1, 2], [2, 1], [5, 5], [6, 5], [5, 6], [2, 5], [5, 2]]
y = [0, 0, 0, 1, 1, 1, 0, 1]
n_trees = 10, max_depth = 3
```

**Output:**
```
accuracy ≥ 0.875 (typically perfect on training data)
```

**Explanation:** Multiple trees combined are more robust than a single tree.

### Example 2

**Input:**
```
n_trees = 5, feature importances
```

**Output:**
```
Feature importances show which features are most useful for splitting
```

**Explanation:** Features used more frequently in splits across all trees are more important.

---

## Approach Hints

1. **Bagging + Feature Randomization:** Build each tree on a bootstrap sample with random feature subsets
2. **Out-of-Bag Error:** Estimate generalization error using samples not in each tree's bootstrap

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Random Forest | O(n_trees × n × d_sub × n_log_n × depth) | O(n_trees × nodes) |
