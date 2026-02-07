"""
Problem: Maximum Likelihood Estimation
Category: Statistics & Probability
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple


# ============================================================
# Approach 1: Closed-Form MLE
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def mle_gaussian_closed(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate Gaussian parameters using closed-form MLE.
    μ_MLE = mean(data), σ_MLE = sqrt(mean((data - μ)²))

    Args:
        data: 1D array of samples

    Returns:
        Tuple of (mu_mle, sigma_mle)
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    # MLE for mean
    mu = 0.0
    for x in data:
        mu += x
    mu /= n

    # MLE for variance (population variance, divide by n)
    var = 0.0
    for x in data:
        var += (x - mu) ** 2
    var /= n

    sigma = np.sqrt(var)
    return mu, sigma


# ============================================================
# Approach 2: Gradient Ascent on Log-Likelihood
# Time Complexity: O(n * iterations)
# Space Complexity: O(1)
# ============================================================

def log_likelihood(data: np.ndarray, mu: float, sigma: float) -> float:
    """Compute Gaussian log-likelihood."""
    n = len(data)
    ll = -n * np.log(sigma) - n / 2 * np.log(2 * np.pi)
    for x in data:
        ll -= (x - mu) ** 2 / (2 * sigma ** 2)
    return ll


def mle_gaussian_gradient(data: np.ndarray, lr: float = 0.01,
                           n_iter: int = 5000) -> Tuple[float, float]:
    """
    Estimate Gaussian parameters using gradient ascent on log-likelihood.

    Gradients of log-likelihood:
    ∂LL/∂μ = Σ(x_i - μ) / σ²
    ∂LL/∂σ = -n/σ + Σ(x_i - μ)² / σ³

    Args:
        data: 1D array of samples
        lr: Learning rate
        n_iter: Number of iterations

    Returns:
        Tuple of (mu_mle, sigma_mle)
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    # Initialize with rough estimates
    mu = data[0]
    sigma = 1.0

    for _ in range(n_iter):
        # Compute gradients
        diff = data - mu
        grad_mu = np.sum(diff) / (sigma ** 2)
        grad_sigma = -n / sigma + np.sum(diff ** 2) / (sigma ** 3)

        # Update
        mu += lr * grad_mu
        sigma += lr * grad_sigma

        # Ensure sigma stays positive
        sigma = max(sigma, 1e-8)

    return mu, sigma


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    data1 = np.array([2.1, 2.5, 3.6, 4.0, 4.5, 5.2])
    mu1, sigma1 = mle_gaussian_closed(data1)
    expected_mu1 = np.sum(data1) / len(data1)
    expected_sigma1 = np.sqrt(np.sum((data1 - expected_mu1)**2) / len(data1))

    print(f"Closed-Form mu:       {'PASS' if abs(mu1 - expected_mu1) < 1e-10 else 'FAIL'} ({mu1:.4f})")
    print(f"Closed-Form sigma:    {'PASS' if abs(sigma1 - expected_sigma1) < 1e-10 else 'FAIL'} ({sigma1:.4f})")

    # Test Example 2
    data2 = np.array([10.2, 9.8, 10.1, 10.0, 9.9])
    mu2, sigma2 = mle_gaussian_closed(data2)
    print(f"Test 2 mu ≈ 10.0:    {'PASS' if abs(mu2 - 10.0) < 0.01 else 'FAIL'} ({mu2:.4f})")
    print(f"Test 2 sigma small:   {'PASS' if sigma2 < 0.2 else 'FAIL'} ({sigma2:.4f})")

    # Test: Gradient ascent should converge to same answer
    mu1_ga, sigma1_ga = mle_gaussian_gradient(data1, lr=0.01, n_iter=10000)
    print(f"Gradient mu:          {'PASS' if abs(mu1_ga - expected_mu1) < 0.05 else 'FAIL'} ({mu1_ga:.4f} ≈ {expected_mu1:.4f})")
    print(f"Gradient sigma:       {'PASS' if abs(sigma1_ga - expected_sigma1) < 0.05 else 'FAIL'} ({sigma1_ga:.4f} ≈ {expected_sigma1:.4f})")

    # Test: Log-likelihood
    ll = log_likelihood(data1, mu1, sigma1)
    print(f"Log-likelihood:       {ll:.4f} (should be near maximum)")

    # Verify that the MLE parameters give higher LL than shifted params
    ll_shifted = log_likelihood(data1, mu1 + 1.0, sigma1)
    print(f"MLE > shifted:        {'PASS' if ll > ll_shifted else 'FAIL'}")
