# Linear Regression

A complete breakdown and implementation of **Linear Regression**—one of the foundational algorithms in supervised machine learning—built both **from scratch using NumPy** and using standard ML tooling.

---

## 1. Intuition (For Beginners)

Linear Regression predicts a **continuous numerical value** (such as house price, temperature, or revenue) by fitting a straight line (in 1D) or a flat hyperplane (in higher dimensions) through the data.

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = Xw + b$$

* **Inputs ($X$):** The features describing each example (e.g., square footage, number of rooms).
* **Weights ($w$):** Learned parameters that control how strongly—and in which direction (positive or negative)—each feature influences the prediction.
* **Bias ($b$):** The baseline offset (intercept) when all input features equal `0`, shifting the prediction line/plane so it is not forced through the origin.
* **Prediction ($\hat{y}$):** The model's continuous output estimate.

---

## 2. Objective Function (Loss)

To measure how far predictions $\hat{y}$ are from the true labels $y$, Linear Regression typically minimizes **Mean Squared Error (MSE)**:

$$\mathcal{L}(w, b) = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Why Square the Errors?
1. **Sign elimination:** Positive and negative errors do not cancel each other out.
2. **Disproportionate penalty for large misses:** Squaring penalizes large errors far more heavily than small ones.
3. **Smooth differentiability:** Unlike Mean Absolute Error ($|y - \hat{y}|$), MSE has a continuous derivative everywhere, making gradient-based optimization clean and well-behaved.

---

## 3. Optimization Mechanics

There are two primary ways to find optimal parameters $(w, b)$:

### A. Gradient Descent (Iterative Optimization)
Starting from initialized parameters, we compute the **partial derivatives** (gradients) of the loss with respect to $w$ and $b$, then take small steps in the opposite direction of the gradient:

$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n} X^T (y - \hat{y}), \qquad \frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

* **Learning Rate ($\alpha$) Sensitivity:**
  * **Too high:** Steps overshoot the minimum, causing oscillation or divergence.
  * **Too low:** Convergence becomes extremely slow and risks stopping prematurely (underfitting).

### B. Normal Equation (Closed-Form Analytical Solution)
For small-to-medium feature dimensions, optimal weights can be solved directly in one step without iterations or a learning rate:

$$w = (X^T X)^{-1} X^T y$$

*(with a column of `1`s appended to $X$ to absorb the bias term $b$).*

### Gradient Descent vs. Normal Equation

| Dimension | Gradient Descent | Normal Equation |
| :--- | :--- | :--- |
| **Complexity** | $O(k \cdot n \cdot d)$ for $k$ iterations | $O(d^3)$ due to matrix inversion |
| **High-dimensional features ($d$ large)** | Scales well | Slow and memory-intensive |
| **Feature scaling required?** | **Yes** — prevents elongated loss contours | No |
| **Collinear / non-invertible $X^T X$** | Handles naturally | Fails unless regularized or pseudo-inverted |
| **Generalization to deep learning** | Direct foundation for neural networks | Not applicable |

---

## 4. Practical Engineering Considerations

* **Feature Scaling (Standardization):** When features span vastly different magnitudes (e.g., `1,000–5,000 sq ft` vs. `1–4 bedrooms`), the MSE loss surface stretches into narrow valleys. Gradient descent zig-zags inefficiently unless features are standardized ($\mu = 0, \sigma = 1$).
* **Sensitivity to Outliers:** Because MSE squares residuals, a small number of extreme outliers can disproportionately pull the fitted line/plane away from the majority distribution.
* **Underfitting (High Bias):** Plain linear regression cannot model strongly non-linear relationships without feature engineering (e.g., polynomial features) or a non-linear model.
* **Overfitting (High Variance):** Easily diagnosed when training error is low while validation/test error is high—typically mitigated via **L2 (Ridge)** or **L1 (Lasso)** regularization, dimensionality reduction, or additional training data.

---

## 5. Repository Files

* **`Linear_reg_numpy.py`** — From-scratch implementation in NumPy explicitly coding the forward pass ($\hat{y} = Xw + b$), MSE loss computation, analytical gradients, and iterative parameter updates.
* **`linear_regression.py`** — Reference implementation using standard ML libraries (`scikit-learn`) for comparison and baseline benchmarking.

### Quick Start

```bash
# Install dependencies
pip install numpy scikit-learn matplotlib

# Run the from-scratch NumPy implementation
python Linear_reg_numpy.py
```