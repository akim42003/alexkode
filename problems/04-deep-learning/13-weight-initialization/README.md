# Weight Initialization

**Difficulty:** Medium

**Category:** Deep Learning

---

## Description

Implement standard weight initialization methods and demonstrate their impact on signal propagation through deep networks.

Methods:

1. **Xavier/Glorot Uniform:** `U(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`
2. **Xavier/Glorot Normal:** `N(0, std)` where `std = sqrt(2 / (fan_in + fan_out))`
3. **He/Kaiming Uniform:** `U(-limit, limit)` where `limit = sqrt(6 / fan_in)`
4. **He/Kaiming Normal:** `N(0, std)` where `std = sqrt(2 / fan_in)`

### Constraints

- `fan_in`: number of input units to a layer.
- `fan_out`: number of output units.
- Xavier for tanh/sigmoid, He for ReLU.
- Use only NumPy.

---

## Examples

### Example 1

**Input:**
```python
fan_in = 256, fan_out = 256, method = "he_normal"
```

**Output:**
```
W shape: (256, 256), std ≈ 0.0884 (sqrt(2/256))
```

**Explanation:** He Normal sets std = sqrt(2/fan_in) to maintain variance through ReLU layers.

### Example 2

**Input:**
```python
5-layer network, 256 units, ReLU, He init
```

**Output:**
```
Layer activations maintain stable mean/std across all layers.
```

**Explanation:** Proper initialization prevents vanishing/exploding activations.

---

## Approach Hints

1. Implement each init as a function returning a `(fan_in, fan_out)` matrix.
2. Forward-propagate random input through several layers and measure activation statistics.
3. Compare He+ReLU vs Xavier+ReLU to see the difference.

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Single layer init | O(fan_in * fan_out) | O(fan_in * fan_out) |
