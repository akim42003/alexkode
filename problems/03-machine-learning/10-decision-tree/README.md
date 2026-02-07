# Decision Tree (ID3/CART)

**Difficulty:** Medium
**Category:** Machine Learning

---

## Description

Implement a decision tree classifier supporting two splitting criteria:

1. **Information Gain (ID3):** Split on the feature that maximizes information gain (reduction in entropy)
2. **Gini Impurity (CART):** Split on the feature that minimizes weighted Gini impurity

Build the tree recursively. At each node, find the best feature and threshold to split on. Support a max_depth parameter to prevent overfitting.

### Constraints

- X has shape (n_samples, n_features) with continuous features
- y has shape (n_samples,) with discrete class labels
- max_depth ≥ 1
- Minimum 2 samples required to split a node

---

## Examples

### Example 1

**Input:**
```
X = [[2, 3], [1, 1], [3, 2], [6, 5], [7, 8], [8, 6]]
y = [0, 0, 0, 1, 1, 1]
max_depth = 2
```

**Output:**
```
predictions on training data = [0, 0, 0, 1, 1, 1]  (100% accuracy)
```

**Explanation:** The tree finds a threshold to perfectly separate the two clusters.

### Example 2

**Input:**
```
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
```

**Output:**
```
Split at x[0] ≤ 1.5 → left: class 0, right: class 1
```

**Explanation:** Best split is at threshold 1.5 on feature 0, perfectly separating the classes.

---

## Approach Hints

1. **Entropy + Information Gain:** H(S) = -Σ p_i log₂(p_i). IG = H(parent) - weighted avg H(children)
2. **Gini Impurity:** G(S) = 1 - Σ p_i². Pick split that minimizes weighted Gini of children

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Decision Tree | O(n × d × n_log_n × depth) | O(n × depth) |
