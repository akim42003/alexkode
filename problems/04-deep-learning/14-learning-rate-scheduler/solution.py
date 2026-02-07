"""
Problem: Learning Rate Scheduler
Category: Deep Learning
Difficulty: Medium

Libraries: numpy
"""

import numpy as np


# ============================================================
# Approach 1: Scheduler Classes
# Time Complexity: O(1) per call
# Space Complexity: O(1)
# ============================================================

class StepDecay:
    """Multiply LR by gamma every step_size epochs."""

    def __init__(self, lr_init: float = 0.1, step_size: int = 10,
                 gamma: float = 0.5):
        self.lr_init = lr_init
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return self.lr_init * (self.gamma ** (epoch // self.step_size))


class ExponentialDecay:
    """lr = lr_0 * decay^epoch."""

    def __init__(self, lr_init: float = 0.1, decay: float = 0.95):
        self.lr_init = lr_init
        self.decay = decay

    def get_lr(self, epoch: int) -> float:
        return self.lr_init * (self.decay ** epoch)


class CosineAnnealing:
    """Cosine annealing from lr_max to lr_min over T_max epochs."""

    def __init__(self, lr_max: float = 0.1, lr_min: float = 0.001,
                 T_max: int = 100):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max

    def get_lr(self, epoch: int) -> float:
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * epoch / self.T_max))


class WarmupDecay:
    """Linear warmup for warmup_steps, then exponential decay."""

    def __init__(self, lr_max: float = 0.1, warmup_steps: int = 10,
                 decay: float = 0.95):
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.decay = decay

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_steps:
            return self.lr_max * (epoch + 1) / self.warmup_steps
        else:
            return self.lr_max * (self.decay ** (epoch - self.warmup_steps))


# ============================================================
# Approach 2: Functional Schedulers
# ============================================================

def step_decay(epoch: int, lr_init: float = 0.1,
               step_size: int = 10, gamma: float = 0.5) -> float:
    """Functional step decay."""
    return lr_init * (gamma ** (epoch // step_size))


def cosine_annealing(epoch: int, lr_max: float = 0.1,
                     lr_min: float = 0.001, T_max: int = 100) -> float:
    """Functional cosine annealing."""
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + np.cos(np.pi * epoch / T_max))


def warmup_cosine(epoch: int, lr_max: float = 0.1, lr_min: float = 0.0,
                  warmup_steps: int = 10, T_max: int = 100) -> float:
    """Warmup then cosine annealing."""
    if epoch < warmup_steps:
        return lr_max * (epoch + 1) / warmup_steps
    progress = (epoch - warmup_steps) / max(T_max - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":
    # Test Step Decay
    sd = StepDecay(lr_init=0.1, step_size=10, gamma=0.5)
    print(f"Step epoch 0:   {sd.get_lr(0):.4f}")
    print(f"Step epoch 9:   {sd.get_lr(9):.4f}")
    print(f"Step epoch 10:  {sd.get_lr(10):.4f}")
    print(f"Step epoch 20:  {sd.get_lr(20):.4f}")
    print(f"Step decay:     {'PASS' if np.isclose(sd.get_lr(0), 0.1) and np.isclose(sd.get_lr(10), 0.05) else 'FAIL'}")

    # Test Exponential Decay
    ed = ExponentialDecay(lr_init=0.1, decay=0.95)
    print(f"\nExp epoch 0:    {ed.get_lr(0):.4f}")
    print(f"Exp epoch 10:   {ed.get_lr(10):.4f}")
    print(f"Exp monotonic:  {'PASS' if ed.get_lr(0) > ed.get_lr(10) > ed.get_lr(50) else 'FAIL'}")

    # Test Cosine Annealing
    ca = CosineAnnealing(lr_max=0.1, lr_min=0.001, T_max=100)
    print(f"\nCosine epoch 0:  {ca.get_lr(0):.4f}")
    print(f"Cosine epoch 50: {ca.get_lr(50):.4f}")
    print(f"Cosine epoch 100:{ca.get_lr(100):.4f}")
    print(f"Cosine start:    {'PASS' if np.isclose(ca.get_lr(0), 0.1) else 'FAIL'}")
    print(f"Cosine end:      {'PASS' if np.isclose(ca.get_lr(100), 0.001) else 'FAIL'}")
    mid_lr = ca.get_lr(50)
    expected_mid = 0.001 + 0.5 * 0.099 * (1 + np.cos(np.pi * 0.5))
    print(f"Cosine mid:      {'PASS' if np.isclose(mid_lr, expected_mid, atol=1e-4) else 'FAIL'}")

    # Test Warmup + Decay
    wd = WarmupDecay(lr_max=0.1, warmup_steps=10, decay=0.95)
    print(f"\nWarmup epoch 0:  {wd.get_lr(0):.4f}")
    print(f"Warmup epoch 5:  {wd.get_lr(5):.4f}")
    print(f"Warmup epoch 9:  {wd.get_lr(9):.4f}")
    print(f"Warmup epoch 10: {wd.get_lr(10):.4f}")
    print(f"Warmup epoch 20: {wd.get_lr(20):.4f}")
    print(f"Warmup ramp:     {'PASS' if wd.get_lr(0) < wd.get_lr(5) < wd.get_lr(9) else 'FAIL'}")
    print(f"Warmup peak:     {'PASS' if np.isclose(wd.get_lr(9), 0.1) else 'FAIL'}")
    print(f"Post-warmup dec: {'PASS' if wd.get_lr(10) > wd.get_lr(20) else 'FAIL'}")

    # Test Warmup Cosine
    lrs = [warmup_cosine(e, lr_max=0.1, warmup_steps=5, T_max=50) for e in range(50)]
    print(f"\nWarmup-cosine ramp: {'PASS' if lrs[0] < lrs[4] else 'FAIL'}")
    print(f"Warmup-cosine dec:  {'PASS' if lrs[10] > lrs[40] else 'FAIL'}")

    # Test all are positive
    all_pos = all(sd.get_lr(e) > 0 and ed.get_lr(e) > 0 and
                  ca.get_lr(e) > 0 and wd.get_lr(e) > 0
                  for e in range(100))
    print(f"All positive:       {'PASS' if all_pos else 'FAIL'}")
