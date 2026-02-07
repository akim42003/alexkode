# Learning Rate Scheduler

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement common learning rate scheduling strategies:

1. **Step Decay:** Multiply LR by `gamma` every `step_size` epochs.
2. **Exponential Decay:** `lr = lr_0 * decay^epoch`.
3. **Cosine Annealing:** `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / T))`.
4. **Warmup + Decay:** Linear warmup for `warmup_steps`, then decay.

### Constraints

- Each scheduler takes the current epoch/step and returns the learning rate.
- Must be deterministic (same epoch always gives same LR).
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
scheduler = StepDecay(lr_init=0.1, step_size=10, gamma=0.5)
```

**Output:**
```
epoch 0-9:   lr = 0.1
epoch 10-19: lr = 0.05
epoch 20-29: lr = 0.025
```

**Explanation:** LR halves every 10 epochs.

### Example 2

**Input:**
```python
scheduler = CosineAnnealing(lr_max=0.1, lr_min=0.001, T_max=100)
```

**Output:**
```
epoch 0:   lr = 0.1
epoch 50:  lr ≈ 0.0505
epoch 100: lr = 0.001
```

**Explanation:** LR follows a cosine curve from max to min over T_max epochs.

---

## Approach Hints

1. Step decay: `lr = lr_0 * gamma^(epoch // step_size)`.
2. Cosine annealing produces smooth decay that's popular in modern training.
3. Warmup is critical for transformers -- start small and ramp up.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| All schedulers | O(1) per call | O(1) |
