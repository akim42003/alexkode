"""
Problem: Naive Bayes Classifier
Category: Machine Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Dict


# ============================================================
# Approach 1: Gaussian Naive Bayes
# Time Complexity: O(n*d) training, O(c*d) per prediction
# Space Complexity: O(c*d)
# ============================================================

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier."""

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        self.classes = None
        self.priors = None
        self.means = None
        self.variances = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit the model by computing class priors, means, and variances.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = len(X_c) / n_samples
            self.means[idx] = np.mean(X_c, axis=0)
            self.variances[idx] = np.var(X_c, axis=0) + self.epsilon

        return self

    def _log_likelihood(self, x: np.ndarray, class_idx: int) -> float:
        """Compute log P(x|C) for a single sample."""
        mean = self.means[class_idx]
        var = self.variances[class_idx]

        # Log of Gaussian PDF
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_prob -= 0.5 * np.sum((x - mean) ** 2 / var)
        return log_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        predictions = []

        for x in X:
            log_posteriors = np.zeros(len(self.classes))
            for idx in range(len(self.classes)):
                log_posteriors[idx] = np.log(self.priors[idx]) + self._log_likelihood(x, idx)
            predictions.append(self.classes[np.argmax(log_posteriors)])

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        n_classes = len(self.classes)
        proba = np.zeros((n, n_classes))

        for i, x in enumerate(X):
            log_posts = np.zeros(n_classes)
            for idx in range(n_classes):
                log_posts[idx] = np.log(self.priors[idx]) + self._log_likelihood(x, idx)
            # Softmax for probabilities
            log_posts -= np.max(log_posts)
            exp_posts = np.exp(log_posts)
            proba[i] = exp_posts / np.sum(exp_posts)

        return proba


# ============================================================
# Approach 2: Functional Implementation
# Time Complexity: Same as above
# Space Complexity: O(c * d)
# ============================================================

def fit_naive_bayes(X: np.ndarray, y: np.ndarray) -> Dict:
    """Functional fit: returns model parameters as a dict."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    eps = 1e-9

    params = {'classes': classes, 'priors': {}, 'means': {}, 'vars': {}}
    n = len(y)
    for c in classes:
        X_c = X[y == c]
        params['priors'][c] = len(X_c) / n
        params['means'][c] = np.mean(X_c, axis=0)
        params['vars'][c] = np.var(X_c, axis=0) + eps

    return params


def predict_naive_bayes(X: np.ndarray, params: Dict) -> np.ndarray:
    """Functional predict."""
    X = np.asarray(X, dtype=np.float64)
    preds = []
    for x in X:
        best_class = None
        best_log_post = -np.inf
        for c in params['classes']:
            log_prior = np.log(params['priors'][c])
            mean = params['means'][c]
            var = params['vars'][c]
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean)**2 / var)
            log_post = log_prior + log_lik
            if log_post > best_log_post:
                best_log_post = log_post
                best_class = c
        preds.append(best_class)
    return np.array(preds)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 2: Well-separated clusters
    X_train = np.array([[1, 1], [1, 2], [2, 1], [5, 5], [6, 5], [5, 6]], dtype=np.float64)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[1.5, 1.5], [5.5, 5.5]], dtype=np.float64)

    # Class-based
    nb = GaussianNaiveBayes().fit(X_train, y_train)
    preds = nb.predict(X_test)
    print(f"Class NB:        {'PASS' if preds[0] == 0 and preds[1] == 1 else 'FAIL'} ({preds})")

    # Functional
    params = fit_naive_bayes(X_train, y_train)
    preds_fn = predict_naive_bayes(X_test, params)
    print(f"Functional NB:   {'PASS' if preds_fn[0] == 0 and preds_fn[1] == 1 else 'FAIL'} ({preds_fn})")

    # Train accuracy
    train_preds = nb.predict(X_train)
    train_acc = np.mean(train_preds == y_train)
    print(f"Train accuracy:  {'PASS' if train_acc == 1.0 else 'FAIL'} ({train_acc})")

    # Probabilities sum to 1
    proba = nb.predict_proba(X_test)
    sums = np.sum(proba, axis=1)
    print(f"Proba sum to 1:  {'PASS' if np.allclose(sums, 1.0) else 'FAIL'}")
    print(f"Proba P(0|near0) > 0.5: {'PASS' if proba[0, 0] > 0.5 else 'FAIL'}")
