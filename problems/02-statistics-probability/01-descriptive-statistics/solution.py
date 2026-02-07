"""
Problem: Calculate Mean, Variance, Std Dev
Category: Statistics & Probability
Difficulty: Easy

Libraries: numpy
"""

import numpy as np
from typing import Tuple
import math


# ============================================================
# Approach 1: Two-Pass Algorithm
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def mean(data: np.ndarray) -> float:
    """Compute the arithmetic mean."""
    total = 0.0
    for x in data:
        total += x
    return total / len(data)


def variance(data: np.ndarray, ddof: int = 0) -> float:
    """
    Compute variance using two-pass algorithm.

    Args:
        data: 1D array of values
        ddof: Delta degrees of freedom. 0 for population, 1 for sample.

    Returns:
        Variance value
    """
    n = len(data)
    if n - ddof <= 0:
        raise ValueError(f"Need at least {ddof + 1} samples for ddof={ddof}")

    mu = mean(data)
    sum_sq = 0.0
    for x in data:
        sum_sq += (x - mu) ** 2
    return sum_sq / (n - ddof)


def std_dev(data: np.ndarray, ddof: int = 0) -> float:
    """Compute standard deviation (square root of variance)."""
    return math.sqrt(variance(data, ddof))


def descriptive_stats(data: np.ndarray) -> dict:
    """
    Compute all descriptive statistics at once.

    Args:
        data: 1D array of values

    Returns:
        Dict with mean, pop_var, sample_var, pop_std, sample_std
    """
    mu = mean(data)
    pop_var = variance(data, ddof=0)
    sam_var = variance(data, ddof=1)
    return {
        "mean": mu,
        "population_variance": pop_var,
        "sample_variance": sam_var,
        "population_std": math.sqrt(pop_var),
        "sample_std": math.sqrt(sam_var),
    }


# ============================================================
# Approach 2: Welford's Online Algorithm (Single Pass)
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def welford_stats(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute mean, population variance, and sample variance using
    Welford's numerically stable online algorithm.

    Args:
        data: 1D array of values

    Returns:
        Tuple of (mean, population_variance, sample_variance)
    """
    n = 0
    mu = 0.0
    M2 = 0.0  # sum of squared differences from the running mean

    for x in data:
        n += 1
        delta = x - mu
        mu += delta / n
        delta2 = x - mu
        M2 += delta * delta2

    pop_var = M2 / n if n > 0 else 0.0
    sam_var = M2 / (n - 1) if n > 1 else 0.0
    return mu, pop_var, sam_var


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    data1 = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=np.float64)
    stats1 = descriptive_stats(data1)
    print(f"Mean:         {'PASS' if abs(stats1['mean'] - 5.0) < 1e-10 else 'FAIL'} ({stats1['mean']})")
    print(f"Pop Var:      {'PASS' if abs(stats1['population_variance'] - 4.0) < 1e-10 else 'FAIL'} ({stats1['population_variance']})")
    print(f"Sample Var:   {'PASS' if abs(stats1['sample_variance'] - 32/7) < 1e-10 else 'FAIL'} ({stats1['sample_variance']})")
    print(f"Pop Std:      {'PASS' if abs(stats1['population_std'] - 2.0) < 1e-10 else 'FAIL'} ({stats1['population_std']})")
    print(f"Sample Std:   {'PASS' if abs(stats1['sample_std'] - math.sqrt(32/7)) < 1e-10 else 'FAIL'} ({stats1['sample_std']})")

    # Test Welford's
    mu_w, pv_w, sv_w = welford_stats(data1)
    print(f"Welford Mean: {'PASS' if abs(mu_w - 5.0) < 1e-10 else 'FAIL'}")
    print(f"Welford PVar: {'PASS' if abs(pv_w - 4.0) < 1e-10 else 'FAIL'}")
    print(f"Welford SVar: {'PASS' if abs(sv_w - 32/7) < 1e-10 else 'FAIL'}")

    # Test Example 2
    data2 = np.array([10, 20, 30], dtype=np.float64)
    stats2 = descriptive_stats(data2)
    print(f"\nTest 2 Mean:       {'PASS' if abs(stats2['mean'] - 20.0) < 1e-10 else 'FAIL'}")
    print(f"Test 2 Pop Var:    {'PASS' if abs(stats2['population_variance'] - 200/3) < 1e-10 else 'FAIL'}")
    print(f"Test 2 Sample Var: {'PASS' if abs(stats2['sample_variance'] - 100.0) < 1e-10 else 'FAIL'}")
    print(f"Test 2 Sample Std: {'PASS' if abs(stats2['sample_std'] - 10.0) < 1e-10 else 'FAIL'}")
