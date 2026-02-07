"""
Problem: Monte Carlo Sampling
Category: Statistics & Probability
Difficulty: Hard

Libraries: numpy
"""

import numpy as np
from typing import Callable, Tuple


# ============================================================
# Approach 1: Estimate Pi
# Time Complexity: O(n)
# Space Complexity: O(n) for vectorized, O(1) for loop
# ============================================================

def estimate_pi(n_samples: int, seed: int = 42) -> float:
    """
    Estimate π using Monte Carlo method.
    Generate random points in [0,1]² and check if they fall inside the unit circle.

    Args:
        n_samples: Number of random points to generate
        seed: Random seed for reproducibility

    Returns:
        Estimate of π
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 1, n_samples)
    y = rng.uniform(0, 1, n_samples)

    # Count points inside the quarter circle (x² + y² ≤ 1)
    inside = np.sum(x**2 + y**2 <= 1.0)

    # Area of quarter circle / area of square = π/4
    return 4.0 * inside / n_samples


def estimate_pi_loop(n_samples: int, seed: int = 42) -> float:
    """Estimate π using explicit loop (no vectorization)."""
    rng = np.random.RandomState(seed)
    inside = 0

    for _ in range(n_samples):
        x = rng.uniform(0, 1)
        y = rng.uniform(0, 1)
        if x**2 + y**2 <= 1.0:
            inside += 1

    return 4.0 * inside / n_samples


# ============================================================
# Approach 2: Monte Carlo Integration
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def mc_integrate(f: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                 n_samples: int = 100000, seed: int = 42) -> Tuple[float, float]:
    """
    Estimate the definite integral of f(x) over [a, b] using Monte Carlo.
    Integral ≈ (b - a) × (1/n) × Σ f(x_i) where x_i ~ Uniform[a, b]

    Args:
        f: Function to integrate (must accept numpy arrays)
        a: Lower bound
        b: Upper bound
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Tuple of (estimate, standard_error)
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(a, b, n_samples)
    fx = f(x)

    # Estimate = (b-a) * mean(f(x))
    estimate = (b - a) * np.mean(fx)

    # Standard error
    std_err = (b - a) * np.std(fx) / np.sqrt(n_samples)

    return estimate, std_err


# ============================================================
# Approach 3: Importance Sampling
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def importance_sampling(f: Callable, p_pdf: Callable, q_pdf: Callable,
                        q_sample: Callable, n_samples: int = 100000,
                        seed: int = 42) -> Tuple[float, float]:
    """
    Estimate E_p[f(x)] using importance sampling with proposal distribution q.
    E_p[f(x)] ≈ (1/n) × Σ f(x_i) × p(x_i) / q(x_i) where x_i ~ q

    This is useful when f(x)*p(x) has high values in regions where p(x) is low,
    and q(x) better covers those regions.

    Args:
        f: Function to evaluate
        p_pdf: Target distribution PDF
        q_pdf: Proposal distribution PDF
        q_sample: Function to generate samples from q (takes n, seed)
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Tuple of (estimate, standard_error)
    """
    # Sample from proposal distribution
    x = q_sample(n_samples, seed)

    # Compute importance weights
    weights = p_pdf(x) / q_pdf(x)
    fx = f(x)
    weighted = fx * weights

    estimate = np.mean(weighted)
    std_err = np.std(weighted) / np.sqrt(n_samples)

    return estimate, std_err


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Estimate Pi
    pi_est = estimate_pi(1000000)
    print(f"Pi estimate (1M):      {'PASS' if abs(pi_est - np.pi) < 0.01 else 'FAIL'} ({pi_est:.6f} ≈ {np.pi:.6f})")

    pi_est_loop = estimate_pi_loop(100000)
    print(f"Pi estimate (loop):    {'PASS' if abs(pi_est_loop - np.pi) < 0.05 else 'FAIL'} ({pi_est_loop:.6f})")

    # Test 2: Integrate x^2 from 0 to 1 (exact = 1/3)
    integral, err = mc_integrate(lambda x: x**2, 0, 1, n_samples=1000000)
    print(f"∫x² dx [0,1]:         {'PASS' if abs(integral - 1/3) < 0.01 else 'FAIL'} ({integral:.6f} ≈ {1/3:.6f})")

    # Test 3: Integrate sin(x) from 0 to pi (exact = 2)
    integral2, err2 = mc_integrate(lambda x: np.sin(x), 0, np.pi, n_samples=1000000)
    print(f"∫sin(x) dx [0,π]:     {'PASS' if abs(integral2 - 2.0) < 0.01 else 'FAIL'} ({integral2:.6f} ≈ 2.0)")

    # Test 4: Integrate e^(-x) from 0 to 1 (exact = 1 - 1/e ≈ 0.6321)
    integral3, err3 = mc_integrate(lambda x: np.exp(-x), 0, 1, n_samples=1000000)
    exact3 = 1 - 1/np.e
    print(f"∫e^(-x) dx [0,1]:     {'PASS' if abs(integral3 - exact3) < 0.01 else 'FAIL'} ({integral3:.6f} ≈ {exact3:.6f})")

    # Test 5: Importance Sampling
    # Estimate E_p[f(x)] where p = N(0,1) and f(x) = x^2 (should give 1, the variance)
    def f_test(x):
        return x ** 2

    def p_pdf_normal(x):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def q_pdf_normal(x):
        return (1 / np.sqrt(2 * np.pi * 4)) * np.exp(-0.5 * x**2 / 4)

    def q_sample_normal(n, seed):
        return np.random.RandomState(seed).normal(0, 2, n)

    is_est, is_err = importance_sampling(f_test, p_pdf_normal, q_pdf_normal,
                                          q_sample_normal, n_samples=500000)
    print(f"Importance Sampling:   {'PASS' if abs(is_est - 1.0) < 0.05 else 'FAIL'} ({is_est:.4f} ≈ 1.0)")
