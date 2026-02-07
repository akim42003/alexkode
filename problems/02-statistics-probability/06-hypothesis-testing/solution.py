"""
Problem: Hypothesis Testing (t-test)
Category: Statistics & Probability
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Tuple
import math


# ============================================================
# Helper: t-distribution PDF and p-value approximation
# ============================================================

def t_distribution_pdf(t: float, df: float) -> float:
    """Compute the PDF of the t-distribution at t with df degrees of freedom."""
    # Using the gamma function for the t-distribution PDF
    # f(t) = Γ((df+1)/2) / (√(df*π) * Γ(df/2)) * (1 + t²/df)^(-(df+1)/2)
    coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return coeff * (1 + t**2 / df) ** (-(df + 1) / 2)


def p_value_two_tailed(t_stat: float, df: float, n_steps: int = 10000) -> float:
    """
    Approximate two-tailed p-value using numerical integration (trapezoidal rule).
    p = 2 * P(T > |t|) = 2 * integral from |t| to large_value of PDF.
    """
    abs_t = abs(t_stat)
    upper = abs_t + 50  # Integrate far enough into the tail

    h = (upper - abs_t) / n_steps
    total = 0.5 * (t_distribution_pdf(abs_t, df) + t_distribution_pdf(upper, df))
    for i in range(1, n_steps):
        x = abs_t + i * h
        total += t_distribution_pdf(x, df)

    one_tail = total * h
    return min(2 * one_tail, 1.0)


# ============================================================
# Approach 1: Student's t-test (Equal Variance)
# Time Complexity: O(n1 + n2)
# Space Complexity: O(1)
# ============================================================

def students_ttest(group1: np.ndarray, group2: np.ndarray,
                   alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Two-sample Student's t-test assuming equal variance.

    Args:
        group1: First sample array
        group2: Second sample array
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, p_value, reject_null)
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)

    # Compute means
    mean1 = np.sum(g1) / n1
    mean2 = np.sum(g2) / n2

    # Compute sample variances
    var1 = np.sum((g1 - mean1) ** 2) / (n1 - 1)
    var2 = np.sum((g2 - mean2) ** 2) / (n2 - 1)

    # Pooled standard deviation
    sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # t-statistic
    t_stat = (mean1 - mean2) / (sp * np.sqrt(1/n1 + 1/n2))

    # Degrees of freedom
    df = n1 + n2 - 2

    # p-value
    p_val = p_value_two_tailed(t_stat, df)

    return t_stat, p_val, p_val < alpha


# ============================================================
# Approach 2: Welch's t-test (Unequal Variance)
# Time Complexity: O(n1 + n2)
# Space Complexity: O(1)
# ============================================================

def welch_ttest(group1: np.ndarray, group2: np.ndarray,
                alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Two-sample Welch's t-test (does not assume equal variance).

    Args:
        group1: First sample array
        group2: Second sample array
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, p_value, reject_null)
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)

    mean1 = np.sum(g1) / n1
    mean2 = np.sum(g2) / n2

    var1 = np.sum((g1 - mean1) ** 2) / (n1 - 1)
    var2 = np.sum((g2 - mean2) ** 2) / (n2 - 1)

    # t-statistic
    se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2) ** 2
    denom = (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
    df = num / denom

    p_val = p_value_two_tailed(t_stat, df)

    return t_stat, p_val, p_val < alpha


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Different means
    g1 = np.array([5.1, 4.9, 5.0, 5.2, 4.8])
    g2 = np.array([5.5, 5.7, 5.6, 5.8, 5.4])

    t1, p1, reject1 = students_ttest(g1, g2)
    print(f"Student's t-test:")
    print(f"  t-stat:    {t1:.4f}")
    print(f"  p-value:   {p1:.4f}")
    print(f"  reject H0: {'PASS' if reject1 == True else 'FAIL'} ({reject1})")

    t1w, p1w, reject1w = welch_ttest(g1, g2)
    print(f"Welch's t-test:")
    print(f"  t-stat:    {t1w:.4f}")
    print(f"  reject H0: {'PASS' if reject1w == True else 'FAIL'} ({reject1w})")

    # Test Example 2: Similar means
    g3 = np.array([3.0, 3.1, 2.9, 3.0])
    g4 = np.array([3.1, 2.9, 3.0, 3.0])

    t2, p2, reject2 = students_ttest(g3, g4)
    print(f"\nSimilar groups:")
    print(f"  t-stat:    {t2:.4f} (should be ≈ 0)")
    print(f"  reject H0: {'PASS' if reject2 == False else 'FAIL'} ({reject2})")

    # Test: Identical groups
    g5 = np.array([1.0, 2.0, 3.0])
    t3, p3, reject3 = students_ttest(g5, g5.copy())
    print(f"\nIdentical groups:")
    print(f"  t-stat:    {'PASS' if abs(t3) < 1e-10 else 'FAIL'} ({t3:.6f})")
    print(f"  reject H0: {'PASS' if reject3 == False else 'FAIL'}")
