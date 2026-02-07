# Mean Squared Error / MAE

**Difficulty:** Easy
**Category:** Machine Learning

---

## Description

Implement regression evaluation metrics from scratch:

1. **MSE (Mean Squared Error):** `(1/n) × Σ(y_true - y_pred)²`
2. **MAE (Mean Absolute Error):** `(1/n) × Σ|y_true - y_pred|`
3. **RMSE (Root Mean Squared Error):** `√MSE`
4. **R² (Coefficient of Determination):** `1 - SS_res / SS_tot` where `SS_res = Σ(y - ŷ)²` and `SS_tot = Σ(y - ȳ)²`

### Constraints

- y_true and y_pred are 1D arrays of the same length
- n ≥ 1
- R² can be negative if the model is worse than predicting the mean

---

## Examples

### Example 1

**Input:**
```
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
```

**Output:**
```
MSE = 0.375
MAE = 0.5
RMSE ≈ 0.6124
R² ≈ 0.9486
```

**Explanation:** Errors are [0.5, -0.5, 0.0, -1.0]. MSE = (0.25+0.25+0+1)/4 = 0.375.

### Example 2

**Input:**
```
y_true = [1, 2, 3]
y_pred = [1, 2, 3]
```

**Output:**
```
MSE = 0.0, MAE = 0.0, RMSE = 0.0, R² = 1.0
```

**Explanation:** Perfect predictions. All errors are zero.

---

## Approach Hints

1. **Direct Loop:** Iterate and accumulate the error terms
2. **Vectorized:** Use array subtraction and aggregation

---

## Expected Complexity

| Approach | Time | Space |
|----------|------|-------|
| Direct | O(n) | O(1) |
