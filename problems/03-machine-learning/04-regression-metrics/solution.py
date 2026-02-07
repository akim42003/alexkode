"""
Problem: Mean Squared Error / MAE
Category: Machine Learning
Difficulty: Easy

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Loop-based
# Time Complexity: O(n)
# Space Complexity: O(1)
# ============================================================

def mse_loop(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error using explicit loop."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        total += (y_true[i] - y_pred[i]) ** 2
    return total / n


def mae_loop(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error using explicit loop."""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        total += abs(y_true[i] - y_pred[i])
    return total / n


def rmse_loop(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse_loop(y_true, y_pred))


def r_squared_loop(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (Coefficient of Determination) using explicit loop."""
    n = len(y_true)

    # Compute mean of y_true
    y_mean = 0.0
    for i in range(n):
        y_mean += y_true[i]
    y_mean /= n

    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - y_mean) ** 2

    if ss_tot < 1e-12:
        return 0.0 if ss_res > 1e-12 else 1.0

    return 1.0 - ss_res / ss_tot


# ============================================================
# Approach 2: Vectorized
# Time Complexity: O(n)
# Space Complexity: O(n)
# ============================================================

def mse_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error using numpy vectorization."""
    errors = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return np.sum(errors ** 2) / len(errors)


def mae_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error using numpy vectorization."""
    errors = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return np.sum(np.abs(errors)) / len(errors)


def rmse_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse_vectorized(y_true, y_pred))


def r_squared_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² using numpy vectorization."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_mean = np.sum(y_true) / len(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    if ss_tot < 1e-12:
        return 0.0 if ss_res > 1e-12 else 1.0
    return 1.0 - ss_res / ss_tot


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    y_true1 = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred1 = np.array([2.5, 0.0, 2.0, 8.0])

    # Test MSE
    mse_val = mse_loop(y_true1, y_pred1)
    print(f"MSE (Loop):       {'PASS' if abs(mse_val - 0.375) < 1e-10 else 'FAIL'} ({mse_val})")
    print(f"MSE (Vec):        {'PASS' if abs(mse_vectorized(y_true1, y_pred1) - 0.375) < 1e-10 else 'FAIL'}")

    # Test MAE
    mae_val = mae_loop(y_true1, y_pred1)
    print(f"MAE (Loop):       {'PASS' if abs(mae_val - 0.5) < 1e-10 else 'FAIL'} ({mae_val})")
    print(f"MAE (Vec):        {'PASS' if abs(mae_vectorized(y_true1, y_pred1) - 0.5) < 1e-10 else 'FAIL'}")

    # Test RMSE
    rmse_val = rmse_loop(y_true1, y_pred1)
    print(f"RMSE:             {'PASS' if abs(rmse_val - np.sqrt(0.375)) < 1e-10 else 'FAIL'} ({rmse_val:.4f})")

    # Test R²
    r2_val = r_squared_loop(y_true1, y_pred1)
    ss_res = np.sum((y_true1 - y_pred1)**2)
    ss_tot = np.sum((y_true1 - np.mean(y_true1))**2)
    expected_r2 = 1 - ss_res / ss_tot
    print(f"R² (Loop):        {'PASS' if abs(r2_val - expected_r2) < 1e-10 else 'FAIL'} ({r2_val:.4f})")
    print(f"R² (Vec):         {'PASS' if abs(r_squared_vectorized(y_true1, y_pred1) - expected_r2) < 1e-10 else 'FAIL'}")

    # Test perfect prediction
    y_perfect = np.array([1, 2, 3], dtype=np.float64)
    print(f"\nPerfect prediction:")
    print(f"MSE = 0:          {'PASS' if mse_loop(y_perfect, y_perfect) == 0 else 'FAIL'}")
    print(f"R² = 1:           {'PASS' if r_squared_loop(y_perfect, y_perfect) == 1.0 else 'FAIL'}")

    # Test: R² can be negative
    y_bad = np.array([0.0, 0.0, 0.0, 0.0])  # predict constant 0
    r2_bad = r_squared_loop(y_true1, y_bad)
    print(f"R² negative:      {'PASS' if r2_bad < 0 else 'FAIL'} ({r2_bad:.4f})")
