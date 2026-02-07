"""
Problem: Probability Distributions (Gaussian)
Category: Statistics & Probability
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Direct Formula
# Time Complexity: O(1) for PDF, O(n_steps) for CDF
# Space Complexity: O(1)
# ============================================================

def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Compute the Gaussian probability density function.

    Args:
        x: Point at which to evaluate
        mu: Mean of the distribution
        sigma: Standard deviation (must be > 0)

    Returns:
        PDF value at x
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def gaussian_cdf_trapezoidal(x: float, mu: float = 0.0, sigma: float = 1.0,
                              n_steps: int = 10000) -> float:
    """
    Compute the Gaussian CDF using the trapezoidal rule for numerical integration.
    Integrates PDF from (mu - 10*sigma) to x.

    Args:
        x: Upper bound of integration
        mu: Mean of the distribution
        sigma: Standard deviation
        n_steps: Number of integration steps

    Returns:
        CDF value (approximate probability P(X <= x))
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    lower = mu - 10 * sigma
    if x <= lower:
        return 0.0

    h = (x - lower) / n_steps
    total = 0.5 * (gaussian_pdf(lower, mu, sigma) + gaussian_pdf(x, mu, sigma))

    for i in range(1, n_steps):
        xi = lower + i * h
        total += gaussian_pdf(xi, mu, sigma)

    return total * h


# ============================================================
# Approach 2: Vectorized PDF (for arrays of x values)
# Time Complexity: O(n) for n points
# Space Complexity: O(n)
# ============================================================

def gaussian_pdf_vectorized(x: np.ndarray, mu: float = 0.0,
                             sigma: float = 1.0) -> np.ndarray:
    """
    Vectorized Gaussian PDF for an array of x values.

    Args:
        x: Array of points
        mu: Mean
        sigma: Standard deviation

    Returns:
        Array of PDF values
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def gaussian_cdf_vectorized(x: float, mu: float = 0.0, sigma: float = 1.0,
                             n_steps: int = 10000) -> float:
    """
    Vectorized CDF computation using trapezoidal rule with numpy arrays.

    Args:
        x: Upper bound
        mu: Mean
        sigma: Standard deviation
        n_steps: Number of steps

    Returns:
        CDF value
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    lower = mu - 10 * sigma
    if x <= lower:
        return 0.0

    points = np.linspace(lower, x, n_steps + 1)
    pdf_values = gaussian_pdf_vectorized(points, mu, sigma)
    # Trapezoidal rule
    h = (x - lower) / n_steps
    return h * (0.5 * pdf_values[0] + np.sum(pdf_values[1:-1]) + 0.5 * pdf_values[-1])


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Example 1: Standard normal at x=0
    pdf_val = gaussian_pdf(0, 0, 1)
    expected_pdf = 1 / np.sqrt(2 * np.pi)
    print(f"PDF(0,0,1):        {'PASS' if abs(pdf_val - expected_pdf) < 1e-6 else 'FAIL'} ({pdf_val:.6f} ≈ {expected_pdf:.6f})")

    cdf_val = gaussian_cdf_trapezoidal(0, 0, 1)
    print(f"CDF(0,0,1):        {'PASS' if abs(cdf_val - 0.5) < 1e-4 else 'FAIL'} ({cdf_val:.6f} ≈ 0.5)")

    # Test Example 2: Standard normal at x=1
    pdf_val2 = gaussian_pdf(1, 0, 1)
    print(f"PDF(1,0,1):        {'PASS' if abs(pdf_val2 - 0.24197) < 1e-4 else 'FAIL'} ({pdf_val2:.6f} ≈ 0.2420)")

    cdf_val2 = gaussian_cdf_trapezoidal(1, 0, 1)
    print(f"CDF(1,0,1):        {'PASS' if abs(cdf_val2 - 0.8413) < 1e-3 else 'FAIL'} ({cdf_val2:.6f} ≈ 0.8413)")

    # Test: CDF at -1 (should be ~0.1587)
    cdf_neg1 = gaussian_cdf_trapezoidal(-1, 0, 1)
    print(f"CDF(-1,0,1):       {'PASS' if abs(cdf_neg1 - 0.1587) < 1e-3 else 'FAIL'} ({cdf_neg1:.6f} ≈ 0.1587)")

    # Test: Non-standard normal
    pdf_val3 = gaussian_pdf(5, 5, 2)
    expected_pdf3 = 1 / (2 * np.sqrt(2 * np.pi))
    print(f"PDF(5,5,2):        {'PASS' if abs(pdf_val3 - expected_pdf3) < 1e-6 else 'FAIL'} ({pdf_val3:.6f})")

    # Test: Vectorized matches scalar
    x_arr = np.array([0, 1, -1, 2])
    vec_results = gaussian_pdf_vectorized(x_arr, 0, 1)
    scalar_results = np.array([gaussian_pdf(x, 0, 1) for x in x_arr])
    print(f"Vectorized match:  {'PASS' if np.allclose(vec_results, scalar_results) else 'FAIL'}")

    # Test: CDF vectorized
    cdf_vec = gaussian_cdf_vectorized(0, 0, 1)
    print(f"CDF vectorized:    {'PASS' if abs(cdf_vec - 0.5) < 1e-4 else 'FAIL'} ({cdf_vec:.6f})")
