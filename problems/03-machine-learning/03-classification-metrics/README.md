# Accuracy, Precision, Recall, F1

**Difficulty:** Easy
**Category:** Machine Learning

---

## Description

Implement classification evaluation metrics from scratch:

1. **Accuracy:** Fraction of correct predictions: `correct / total`
2. **Precision:** Of all predicted positives, how many are truly positive: `TP / (TP + FP)`
3. **Recall (Sensitivity):** Of all actual positives, how many were correctly identified: `TP / (TP + FN)`
4. **F1 Score:** Harmonic mean of precision and recall: `2 × (P × R) / (P + R)`

For multi-class, implement macro-averaging (compute per-class, then average).

### Constraints

- y_true and y_pred are 1D arrays of the same length
- For binary: positive class is 1, negative class is 0
- Handle edge cases: if TP+FP=0, precision is 0; if TP+FN=0, recall is 0

---

## Examples

### Example 1

**Input:**
```
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
```

**Output:**
```
accuracy = 0.75
precision = 0.75
recall = 0.75
f1 = 0.75
```

**Explanation:** TP=3, TN=3, FP=1, FN=1. Precision=3/4, Recall=3/4, F1=2×0.75×0.75/1.5=0.75.

### Example 2

**Input:**
```
y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0]
```

**Output:**
```
accuracy = 0.5
precision = 0.0
recall = 0.0
f1 = 0.0
```

**Explanation:** No positive predictions: TP=0, FP=0. Precision=0/0→0, Recall=0/3=0.

---

## Approach Hints

1. **Confusion Matrix:** Build the confusion matrix first, then derive all metrics
2. **Direct Counting:** Count TP, FP, TN, FN directly from the arrays

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Direct | O(n) | O(1) for binary, O(c) for c classes |
