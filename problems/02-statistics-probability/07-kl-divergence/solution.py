"""
Problem: KL Divergence
Category: Statistics & Probability
Difficulty: Medium

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Direct Summation
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute KL divergence KL(P || Q) using direct summation.
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))

    Args:
        P: Probability distribution (1D array, sums to 1)
        Q: Probability distribution (1D array, sums to 1)

    Returns:
        KL divergence value (non-negative)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if len(P) != len(Q):
        raise ValueError("P and Q must have the same length")

    kl = 0.0
    for i in range(len(P)):
        if P[i] > 0:
            if Q[i] <= 0:
                return float('inf')
            kl += P[i] * np.log(P[i] / Q[i])
    return kl


def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric).
    JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2

    Args:
        P: Probability distribution
        Q: Probability distribution

    Returns:
        JS divergence value (non-negative, bounded by log(2))
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


# ============================================================
# Approach 2: Vectorized with Masking
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def kl_divergence_vectorized(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Vectorized KL divergence using numpy operations with masking.

    Args:
        P: Probability distribution
        Q: Probability distribution

    Returns:
        KL divergence value
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if len(P) != len(Q):
        raise ValueError("P and Q must have the same length")

    # Where P > 0 but Q <= 0, KL is infinite
    mask_p = P > 0
    if np.any(mask_p & (Q <= 0)):
        return float('inf')

    # Only compute for terms where P > 0
    result = np.zeros_like(P)
    result[mask_p] = P[mask_p] * np.log(P[mask_p] / Q[mask_p])

    return np.sum(result)


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1
    P1 = np.array([0.4, 0.6])
    Q1 = np.array([0.5, 0.5])
    expected1 = 0.4 * np.log(0.4/0.5) + 0.6 * np.log(0.6/0.5)

    kl1 = kl_divergence(P1, Q1)
    print(f"KL Test 1 (Direct):      {'PASS' if abs(kl1 - expected1) < 1e-10 else 'FAIL'} ({kl1:.6f} ≈ {expected1:.6f})")

    kl1v = kl_divergence_vectorized(P1, Q1)
    print(f"KL Test 1 (Vectorized):  {'PASS' if abs(kl1v - expected1) < 1e-10 else 'FAIL'}")

    # Test Example 2
    P2 = np.array([0.1, 0.2, 0.3, 0.4])
    Q2 = np.array([0.25, 0.25, 0.25, 0.25])
    kl2 = kl_divergence(P2, Q2)
    expected2 = sum(P2[i] * np.log(P2[i]/Q2[i]) for i in range(4))
    print(f"KL Test 2:               {'PASS' if abs(kl2 - expected2) < 1e-10 else 'FAIL'} ({kl2:.6f})")

    # Test: KL is non-negative
    print(f"KL non-negative:         {'PASS' if kl1 >= 0 and kl2 >= 0 else 'FAIL'}")

    # Test: KL(P||P) = 0
    kl_self = kl_divergence(P1, P1)
    print(f"KL(P||P) = 0:            {'PASS' if abs(kl_self) < 1e-10 else 'FAIL'}")

    # Test: Asymmetry
    kl_pq = kl_divergence(P2, Q2)
    kl_qp = kl_divergence(Q2, P2)
    print(f"KL asymmetric:           {'PASS' if abs(kl_pq - kl_qp) > 1e-6 else 'FAIL'} ({kl_pq:.4f} ≠ {kl_qp:.4f})")

    # Test: JS divergence is symmetric
    js_pq = js_divergence(P2, Q2)
    js_qp = js_divergence(Q2, P2)
    print(f"JS symmetric:            {'PASS' if abs(js_pq - js_qp) < 1e-10 else 'FAIL'} ({js_pq:.6f})")
    print(f"JS bounded by log(2):    {'PASS' if js_pq <= np.log(2) + 1e-10 else 'FAIL'}")

    # Test: Q has zero where P is non-zero → infinity
    P3 = np.array([0.5, 0.5])
    Q3 = np.array([1.0, 0.0])
    kl3 = kl_divergence(P3, Q3)
    print(f"KL with zero Q:          {'PASS' if kl3 == float('inf') else 'FAIL'}")
