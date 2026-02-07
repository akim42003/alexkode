"""
Problem: Accuracy, Precision, Recall, F1
Category: Machine Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Dict


# ============================================================
# Approach 1: Direct Counting (Binary)
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    tp = fp = tn = fn = 0
    for i in range(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Approach 2: Confusion Matrix Based (Multi-class)
# Time Complexity: O(n + c^2) where c = number of classes
# Space Complexity: O(c^2)
# ============================================================

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Build a confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
        cm[i][j] = number of samples with true label i and predicted label j
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(len(y_true)):
        true_idx = class_to_idx[y_true[i]]
        pred_idx = class_to_idx[y_pred[i]]
        cm[true_idx, pred_idx] += 1

    return cm


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute multi-class metrics using macro averaging.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dict with accuracy, macro_precision, macro_recall, macro_f1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)

    accuracy = np.sum(y_true == y_pred) / n

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "accuracy": accuracy,
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "macro_f1": np.mean(f1s),
    }


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    y_true1 = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred1 = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    m1 = binary_metrics(y_true1, y_pred1)
    print(f"Accuracy:  {'PASS' if abs(m1['accuracy'] - 0.75) < 1e-10 else 'FAIL'} ({m1['accuracy']})")
    print(f"Precision: {'PASS' if abs(m1['precision'] - 0.75) < 1e-10 else 'FAIL'} ({m1['precision']})")
    print(f"Recall:    {'PASS' if abs(m1['recall'] - 0.75) < 1e-10 else 'FAIL'} ({m1['recall']})")
    print(f"F1:        {'PASS' if abs(m1['f1'] - 0.75) < 1e-10 else 'FAIL'} ({m1['f1']})")

    # Test Example 2: All predicted negative
    y_true2 = np.array([0, 0, 0, 1, 1, 1])
    y_pred2 = np.array([0, 0, 0, 0, 0, 0])

    m2 = binary_metrics(y_true2, y_pred2)
    print(f"\nAll Negative Predictions:")
    print(f"Accuracy:  {'PASS' if abs(m2['accuracy'] - 0.5) < 1e-10 else 'FAIL'}")
    print(f"Precision: {'PASS' if m2['precision'] == 0.0 else 'FAIL'}")
    print(f"Recall:    {'PASS' if m2['recall'] == 0.0 else 'FAIL'}")
    print(f"F1:        {'PASS' if m2['f1'] == 0.0 else 'FAIL'}")

    # Test Multi-class
    y_true3 = np.array([0, 0, 1, 1, 2, 2])
    y_pred3 = np.array([0, 1, 1, 1, 2, 0])
    m3 = multiclass_metrics(y_true3, y_pred3)
    print(f"\nMulti-class:")
    print(f"Accuracy:        {m3['accuracy']:.4f}")
    print(f"Macro Precision: {m3['macro_precision']:.4f}")
    print(f"Macro Recall:    {m3['macro_recall']:.4f}")
    print(f"Macro F1:        {m3['macro_f1']:.4f}")

    # Test Confusion Matrix
    cm = confusion_matrix(y_true3, y_pred3)
    print(f"Confusion Matrix:\n{cm}")
    print(f"CM shape correct: {'PASS' if cm.shape == (3, 3) else 'FAIL'}")
