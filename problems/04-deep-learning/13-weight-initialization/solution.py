"""
Problem: Weight Initialization
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np
from typing import Optional


# ============================================================
# Approach 1: Four Standard Initialization Methods
# Time Complexity: O(fan_in * fan_out)
# Space Complexity: O(fan_in * fan_out)
# ============================================================

def xavier_uniform(fan_in: int, fan_out: int,
                   seed: Optional[int] = None) -> np.ndarray:
    """Xavier/Glorot Uniform: U(-limit, limit), limit = sqrt(6/(fan_in+fan_out))."""
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def xavier_normal(fan_in: int, fan_out: int,
                  seed: Optional[int] = None) -> np.ndarray:
    """Xavier/Glorot Normal: N(0, sqrt(2/(fan_in+fan_out)))."""
    rng = np.random.RandomState(seed)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, (fan_in, fan_out))


def he_uniform(fan_in: int, fan_out: int,
               seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming Uniform: U(-limit, limit), limit = sqrt(6/fan_in)."""
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def he_normal(fan_in: int, fan_out: int,
              seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming Normal: N(0, sqrt(2/fan_in))."""
    rng = np.random.RandomState(seed)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, (fan_in, fan_out))


INIT_METHODS = {
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
}


# ============================================================
# Approach 2: Signal Propagation Analysis
# ============================================================

def analyze_propagation(layer_dims: list, activation: str,
                        init_method: str, n_samples: int = 1000,
                        seed: int = 42) -> list:
    """
    Forward-propagate random data through a deep network and measure
    activation statistics at each layer.

    Args:
        layer_dims: List of layer widths (e.g., [256, 256, 256, 256])
        activation: "relu" or "tanh"
        init_method: One of the initialization method names
        n_samples: Number of random samples
        seed: Random seed

    Returns:
        List of (mean, std) tuples for each layer's activations
    """
    rng = np.random.RandomState(seed)
    init_fn = INIT_METHODS[init_method]

    x = rng.randn(n_samples, layer_dims[0])
    stats = [(float(np.mean(x)), float(np.std(x)))]

    for i in range(len(layer_dims) - 1):
        W = init_fn(layer_dims[i], layer_dims[i + 1], seed=seed + i)
        b = np.zeros(layer_dims[i + 1])
        z = x @ W + b

        if activation == "relu":
            x = np.maximum(0, z)
        elif activation == "tanh":
            x = np.tanh(z)
        elif activation == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        else:
            x = z

        stats.append((float(np.mean(x)), float(np.std(x))))

    return stats


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test 1: Weight statistics
    print("=== Weight Statistics ===")
    W_xu = xavier_uniform(256, 256, seed=42)
    W_xn = xavier_normal(256, 256, seed=42)
    W_hu = he_uniform(256, 256, seed=42)
    W_hn = he_normal(256, 256, seed=42)

    expected_xu_limit = np.sqrt(6.0 / 512)
    expected_xn_std = np.sqrt(2.0 / 512)
    expected_hu_limit = np.sqrt(6.0 / 256)
    expected_hn_std = np.sqrt(2.0 / 256)

    print(f"Xavier Uniform range: [{W_xu.min():.4f}, {W_xu.max():.4f}] (expected ±{expected_xu_limit:.4f})")
    print(f"Xavier Normal std:    {W_xn.std():.4f} (expected {expected_xn_std:.4f})")
    print(f"He Uniform range:     [{W_hu.min():.4f}, {W_hu.max():.4f}] (expected ±{expected_hu_limit:.4f})")
    print(f"He Normal std:        {W_hn.std():.4f} (expected {expected_hn_std:.4f})")

    print(f"Xavier Normal std ok: {'PASS' if abs(W_xn.std() - expected_xn_std) < 0.005 else 'FAIL'}")
    print(f"He Normal std ok:     {'PASS' if abs(W_hn.std() - expected_hn_std) < 0.005 else 'FAIL'}")

    # Test 2: He + ReLU maintains activations
    print("\n=== He Normal + ReLU Propagation ===")
    dims = [256] * 6  # 5 hidden layers
    stats_he = analyze_propagation(dims, "relu", "he_normal")
    for i, (m, s) in enumerate(stats_he):
        label = "Input" if i == 0 else f"Layer {i}"
        print(f"  {label}: mean={m:.4f}, std={s:.4f}")

    # Check that std doesn't vanish or explode
    first_std = stats_he[1][1]
    last_std = stats_he[-1][1]
    ratio = last_std / (first_std + 1e-8)
    print(f"Std ratio (last/first): {ratio:.2f}")
    print(f"Stable propagation: {'PASS' if 0.1 < ratio < 10 else 'FAIL'}")

    # Test 3: Xavier + ReLU (should degrade)
    print("\n=== Xavier Normal + ReLU (less ideal) ===")
    stats_xavier_relu = analyze_propagation(dims, "relu", "xavier_normal")
    for i, (m, s) in enumerate(stats_xavier_relu):
        label = "Input" if i == 0 else f"Layer {i}"
        print(f"  {label}: mean={m:.4f}, std={s:.4f}")

    # Test 4: Xavier + tanh (designed for this)
    print("\n=== Xavier Normal + tanh ===")
    stats_xavier_tanh = analyze_propagation(dims, "tanh", "xavier_normal")
    first_std_tanh = stats_xavier_tanh[1][1]
    last_std_tanh = stats_xavier_tanh[-1][1]
    ratio_tanh = last_std_tanh / (first_std_tanh + 1e-8)
    print(f"Std ratio: {ratio_tanh:.2f}")
    print(f"Stable with tanh: {'PASS' if 0.1 < ratio_tanh < 10 else 'FAIL'}")

    # Test 5: Shapes
    print("\n=== Shape Tests ===")
    for name, fn in INIT_METHODS.items():
        W = fn(128, 64, seed=0)
        print(f"{name} shape: {'PASS' if W.shape == (128, 64) else 'FAIL'}")
