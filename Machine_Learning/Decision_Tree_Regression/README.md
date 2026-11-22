# Decision Tree Classifier

A **Decision Tree Classifier** is a non-parametric **supervised classification algorithm** that predicts discrete class labels by learning a sequence of **if-else splitting rules** on input features. Each internal node tests a feature against a threshold (or category), branches route samples left or right, and leaf nodes output class predictions (or class probabilities).

---

## Core Intuition & Mathematics

### 1. Recursive Splitting & Impurity
At each node, the tree searches across candidate features and split thresholds to find the split that maximizes **purity** in the resulting child nodes. Two standard impurity metrics for classification are:

* **Gini Impurity** (default in scikit-learn's `DecisionTreeClassifier`):
  $$G = 1 - \sum_{k=1}^{K} p_k^2$$
  where $p_k$ is the proportion of samples belonging to class $k$ at that node. $G = 0$ means the node is completely pure (all samples share one class).

* **Entropy & Information Gain**:
  $$H = -\sum_{k=1}^{K} p_k \ln(p_k)$$
  $$\text{Information Gain} = H_{\text{parent}} - \sum_{j \in \{\text{left}, \text{right}\}} \frac{N_j}{N_{\text{parent}}} H_j$$
  Higher information gain means a larger reduction in uncertainty after the split.

### 2. Predictions at Leaf Nodes
* **Class prediction (`predict`):** Typically assigns the majority class present in the reached leaf node.
* **Probability prediction (`predict_proba`):** Estimates class probabilities from the empirical class proportions inside the leaf node.
* **Axis-aligned boundaries:** Splits are orthogonal to individual feature axes, producing step-like / rectangular decision regions rather than smooth linear hyperplanes.

---

## Key Practical Considerations

* **Overfitting vs. Underfitting (Depth & Pruning):** Deep, unrestricted trees can memorize training noise (high variance / overfitting). Control complexity using hyperparameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and cost-complexity pruning (`ccp_alpha`).
* **Feature Scaling:** Decision trees are **invariant to monotonic feature scaling** (like `StandardScaler`), because splits depend only on rank/order comparisons ($x_j \le t$), not magnitudes or distances.
* **Non-Linearity & Interactions:** Naturally captures non-linear relationships and higher-order feature interactions without needing manual polynomial features.
* **Instability:** Small changes in training data can alter top splits and change tree structure—a key reason ensemble methods (Random Forests, Gradient Boosted Trees) build on trees.
* **Classification vs. Regression:** Classification trees split on **Gini** or **entropy** and predict classes/probabilities, whereas regression trees (as in `decision_tree_reg.py` with `DecisionTreeRegressor` on `MedHouseVal`) split to minimize **MSE/variance** and predict continuous averages[cite: 3].

---

## Classification Evaluation Metrics

* **Accuracy:** $\frac{\text{TP} + \text{TN}}{\text{Total}}$ — Useful when classes are balanced, but can mislead under class imbalance.
* **Precision & Recall:** Precision measures exactness among predicted positives ($\frac{\text{TP}}{\text{TP} + \text{FP}}$); Recall measures coverage of actual positives ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
* **F1-Score:** Harmonic mean of precision and recall ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$).
* **Confusion Matrix & ROC-AUC:** Breakdown of TP/TN/FP/FN counts and threshold-independent class ranking performance. *(In `decision_tree_reg.py`, evaluation uses **R²** and **RMSE** instead because the target is continuous[cite: 3].)*

---

## Workflow & Script Notes

* **Pipeline Structure:** Follows modular steps for data loading, dataset summary, duplicate removal, feature/target separation, train/test splitting, model training, and evaluation (`decision_tree_reg.py` demonstrates this end-to-end flow for regression with `max_depth=30`)[cite: 3].
* **Adapting to Classification:** Swap `DecisionTreeRegressor` for `DecisionTreeClassifier`, pass a discrete class target column (using `stratify=y` in `train_test_split` if needed), and evaluate with classification metrics (`classification_report`, `confusion_matrix`, `roc_auc_score`).