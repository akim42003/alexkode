"""
Problem: Bayes' Theorem Implementation
Category: Statistics & Probability
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Union


# ============================================================
# Approach 1: Direct Formula
# Time Complexity: O(n) for n hypotheses
# Space Complexity: O(n)
# ============================================================

def bayes_simple(prior: float, likelihood: float, evidence: float) -> float:
    """
    Compute posterior probability using Bayes' theorem.
    P(A|B) = P(B|A) * P(A) / P(B)

    Args:
        prior: P(A) - prior probability of hypothesis
        likelihood: P(B|A) - probability of evidence given hypothesis
        evidence: P(B) - total probability of evidence

    Returns:
        Posterior probability P(A|B)
    """
    if evidence <= 0:
        raise ValueError("Evidence P(B) must be positive")
    if not (0 <= prior <= 1) or not (0 <= likelihood <= 1):
        raise ValueError("Probabilities must be in [0, 1]")

    return (likelihood * prior) / evidence


def bayes_multiple(priors: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    """
    Compute posterior probabilities for multiple hypotheses.
    P(H_i|B) = P(B|H_i) * P(H_i) / P(B)
    where P(B) = Σ P(B|H_i) * P(H_i)

    Args:
        priors: Array of prior probabilities P(H_i), must sum to 1
        likelihoods: Array of likelihoods P(B|H_i)

    Returns:
        Array of posterior probabilities (sums to 1)
    """
    priors = np.asarray(priors, dtype=np.float64)
    likelihoods = np.asarray(likelihoods, dtype=np.float64)

    if abs(np.sum(priors) - 1.0) > 1e-8:
        raise ValueError(f"Priors must sum to 1, got {np.sum(priors)}")

    # Law of total probability
    evidence = 0.0
    for i in range(len(priors)):
        evidence += likelihoods[i] * priors[i]

    if evidence <= 0:
        raise ValueError("Total evidence P(B) must be positive")

    # Compute posteriors
    posteriors = np.zeros(len(priors), dtype=np.float64)
    for i in range(len(priors)):
        posteriors[i] = (likelihoods[i] * priors[i]) / evidence

    return posteriors


# ============================================================
# Approach 2: Log-Space Computation (Numerically Stable)
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def bayes_log_space(log_priors: np.ndarray, log_likelihoods: np.ndarray) -> np.ndarray:
    """
    Compute posterior probabilities in log space for numerical stability.
    Uses the log-sum-exp trick to avoid underflow.

    Args:
        log_priors: Array of log prior probabilities
        log_likelihoods: Array of log likelihoods

    Returns:
        Array of posterior probabilities (in regular space, sums to 1)
    """
    log_priors = np.asarray(log_priors, dtype=np.float64)
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)

    # log(P(B|H_i) * P(H_i)) = log P(B|H_i) + log P(H_i)
    log_joint = log_likelihoods + log_priors

    # Log-sum-exp trick for log P(B)
    max_log = np.max(log_joint)
    log_evidence = max_log + np.log(np.sum(np.exp(log_joint - max_log)))

    # Log posteriors
    log_posteriors = log_joint - log_evidence

    return np.exp(log_posteriors)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Medical test
    posterior = bayes_simple(prior=0.001, likelihood=0.99, evidence=0.05)
    expected = 0.99 * 0.001 / 0.05
    print(f"Simple Bayes:          {'PASS' if abs(posterior - expected) < 1e-10 else 'FAIL'} ({posterior:.4f} ≈ {expected:.4f})")

    # Test Example 2: Multiple hypotheses
    priors = np.array([0.6, 0.3, 0.1])
    likelihoods = np.array([0.2, 0.5, 0.9])
    posteriors = bayes_multiple(priors, likelihoods)

    evidence = 0.6*0.2 + 0.3*0.5 + 0.1*0.9  # = 0.36
    expected_post = np.array([0.12/0.36, 0.15/0.36, 0.09/0.36])

    print(f"Multiple Hypotheses:   {'PASS' if np.allclose(posteriors, expected_post) else 'FAIL'}")
    print(f"Posteriors sum to 1:   {'PASS' if abs(np.sum(posteriors) - 1.0) < 1e-10 else 'FAIL'}")
    print(f"  H1: {posteriors[0]:.4f}, H2: {posteriors[1]:.4f}, H3: {posteriors[2]:.4f}")

    # Test: Log-space version
    log_priors = np.log(priors)
    log_likelihoods = np.log(likelihoods)
    posteriors_log = bayes_log_space(log_priors, log_likelihoods)
    print(f"Log-space matches:     {'PASS' if np.allclose(posteriors, posteriors_log) else 'FAIL'}")

    # Test: Uniform prior (all equal)
    uni_priors = np.array([0.5, 0.5])
    uni_likes = np.array([0.8, 0.2])
    uni_post = bayes_multiple(uni_priors, uni_likes)
    print(f"Uniform prior:         {'PASS' if np.allclose(uni_post, [0.8, 0.2]) else 'FAIL'}")

    # Test: Prior dominates when likelihoods are equal
    dom_priors = np.array([0.9, 0.1])
    dom_likes = np.array([0.5, 0.5])
    dom_post = bayes_multiple(dom_priors, dom_likes)
    print(f"Prior dominates:       {'PASS' if np.allclose(dom_post, [0.9, 0.1]) else 'FAIL'}")
