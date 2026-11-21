# Logistic Regression

Logistic Regression is a **supervised classification algorithm** used to predict discrete class probabilities (most commonly binary outcomes $y \in \{0, 1\}$). Despite having "regression" in its name, it models the probability of a class rather than a continuous unbounded target.

---

## Core Intuition & Mathematics

### 1. Linear Score (Log-Odds / Logit)
First, compute a linear combination of input features $X \in \mathbb{R}^{N \times D}$, weights $W \in \mathbb{R}^{D \times 1}$, and bias $b \in \mathbb{R}$:

$$z = XW + b$$

Here, $z$ represents the **log-odds** (logit):

$$\ln\left(\frac{p}{1 - p}\right) = z$$

### 2. Sigmoid Activation
Pass $z$ through the **sigmoid** function to map any real value into a probability $\hat{p} \in (0, 1)$:

$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Decision boundary:** Occurs where $\hat{p} = 0.5 \iff z = 0 \iff XW + b = 0$ (a linear hyperplane in feature space).
* **Numerical stability:** Clipping $z$ (e.g., to `[-500, 500]`) prevents overflow in `np.exp(-z)` for large negative inputs.
* **Thresholding:** Convert probabilities into class labels using a threshold $\tau$ (default `0.5`): $\hat{y} = \mathbb{I}(\hat{p} \ge \tau)$.

### 3. Binary Cross-Entropy (Log Loss) & Gradients
Using MSE with sigmoid yields a non-convex loss surface. Instead, Logistic Regression minimizes **Binary Cross-Entropy (BCE)**:

$$\mathcal{L}(W, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

Differentiating BCE with respect to $z$, $W$, and $b$ simplifies cleanly to:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$$

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} X^T (\hat{p} - y), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)$$

Notice the shape discipline: reshaping $y$ to `(N, 1)` keeps `error = y_hat - y` aligned as `(N, 1)`, matching `Z` and `y_hat`.

---

## Key Practical Considerations

* **Feature Scaling (`StandardScaler`):** Put features on comparable scales so gradient descent converges smoothly and regularization penalizes features fairly. Always `.fit()` on training data only and `.transform()` on test data to avoid data leakage.
* **Stratified Train/Test Split (`stratify=y`):** Preserves class proportions across train and test splits, which is critical for imbalanced datasets.
* **Class Imbalance (`class_weight='balanced'`):** Automatically weights classes inversely proportional to their frequency so minority classes aren't drowned out.
* **No Closed-Form Solution:** Unlike Linear Regression's Normal Equation, Logistic Regression has no general closed-form analytical solution and relies on iterative optimization (e.g., Gradient Descent, L-BFGS).

---

## Classification Evaluation Metrics

Because accuracy can be misleading on imbalanced datasets, evaluate predictions using:

* **Confusion Matrix:**
  * **True Positive (TP):** Actual `1`, predicted `1`
  * **True Negative (TN):** Actual `0`, predicted `0`
  * **False Positive (FP):** Actual `0`, predicted `1` (Type I error)
  * **False Negative (FN):** Actual `1`, predicted `0` (Type II error)
* **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Of all predicted positives, what fraction are actually positive?
* **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Of all actual positives, what fraction did the model catch?
* **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean balancing precision and recall.
* **ROC-AUC:** Measures the model's ranking ability across all classification thresholds (`1.0` = perfect separation, `0.5` = random guessing).

---

## Repository Structure

* `logistic_reg_numpy.py` — From-scratch NumPy implementation with sigmoid, gradient descent, probability prediction (`predict_proba`), and thresholded class prediction (`predict`).
* `logistic_regression_pipeline.py` — End-to-end workflow covering data loading, duplicate removal, target distribution inspection, stratified splitting, feature scaling, training, and evaluation.