# Logistic Regression

Logistic Regression predicts probabilities and categories (like `0` or `1`, yes or no) rather than unbounded continuous numbers.

## How It Works
- Start with a linear score from your inputs, weights, and bias[cite: 2].
- Squash that score through the **sigmoid** function so the output stays between `0` and `1` as a probability[cite: 2].
- Pick a threshold (like `0.5`) to turn probabilities into class labels[cite: 2].
- Update weights and bias step by step using gradients to lower classification loss[cite: 2].

## What's in This Folder
- **NumPy implementation (`logistic_reg_numpy.py`)** — Built from scratch with sigmoid, gradient descent, `predict_proba`, and `predict` for a clear look at how it works under the hood[cite: 2].
- **Standard implementation (`logistic_regression_pipeline.py`)** — Uses scikit-learn with preprocessing, train/test splitting, scaling, and evaluation metrics[cite: 1].

## Good Things to Remember
- Keep shapes aligned (such as reshaping `y` to `(N, 1)` in NumPy) so calculations match cleanly[cite: 2].
- Scale features with `StandardScaler` and split with `stratify=y` when class balance matters[cite: 1].
- Look beyond raw accuracy—check precision, recall, F1-score, confusion matrix, and ROC-AUC, especially on imbalanced data[cite: 1].