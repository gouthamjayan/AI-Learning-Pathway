# Multi-Layer Perceptron (MLP)

A **Multi-Layer Perceptron (MLP)** is a feed-forward artificial neural network consisting of an input layer, one or more hidden layers with non-linear activation functions, and an output layer[cite: 4, 5]. Stacking linear transformations with non-linear activations enables MLPs to learn complex, non-linear decision boundaries and feature representations that a single linear layer cannot capture[cite: 4, 5].

---

## Core Intuition & Architecture

### 1. Affine Transformation + Non-Linearity
At each layer $l$, input activations $A^{[l-1]}$ undergo a linear projection followed by an element-wise non-linear activation[cite: 4]:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, \qquad A^{[l]} = g\left(Z^{[l]}\right)$$

* **Why non-linearity matters:** Stacking linear layers without a non-linear activation collapses into a single linear transformation.
* **ReLU (Rectified Linear Unit):** $g(z) = \max(0, z)$ — outputs `0` for negative values and passes positive values through linearly[cite: 4, 5]. Helps mitigate vanishing gradients in deeper networks and keeps activations sparse/fast to compute[cite: 4].

### 2. Low-Level Tensor & Layer Mechanics (`mlp_foundation.cpp`)
In `mlp_foundation.cpp`, the building blocks of a forward pass are constructed explicitly from scratch[cite: 4]:

* **`Tensor` struct:** Manages dynamically allocated `float*` memory (`new float[rows * cols]` / `delete[] data`), row-major indexing `(row_idx * cols) + col_idx`, and Gaussian weight initialization (`std::normal_distribution`)[cite: 4].
* **Modular operations:** Separates matrix multiplication (`matmul`), row-wise bias addition (`add_bias`), and non-linear activation (`relu`)[cite: 4].
* **`LinearLayer::forward`:** Computes `matmul(input, weights)` $\rightarrow$ `add_bias` $\rightarrow$ `relu`, mapping shape `(batch_size, input_dim)` to `(batch_size, output_dim)`[cite: 4].

### 3. Training with Backpropagation & Loss (`mlp.py`)
In `mlp.py`, `TransactionClassifier` uses `nn.Sequential` with `Linear(input_dim -> 64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)`[cite: 5]:

* **Raw logits output:** The final layer outputs unnormalized logits rather than applying sigmoid inside `forward`, pairing cleanly with `nn.BCEWithLogitsLoss` for numerical stability[cite: 5].
* **Class imbalance handling:** Computes `pos_weight = num_negatives / num_positives` on the training set and passes it to `BCEWithLogitsLoss(pos_weight=pos_weight)` so minority-class errors carry higher weight during training[cite: 5].
* **Optimization loop:** Mini-batches (`batch_size=256` via `DataLoader`) run `optimizer.zero_grad()`, forward pass, `loss.backward()` (autograd gradient computation), and `optimizer.step()` with `Adam`[cite: 5].

---

## Key Practical Considerations

* **Feature Scaling (`StandardScaler`):** Neural networks are sensitive to feature scale—unscaled features with large magnitudes can skew gradients and slow or destabilize training. Fit on training data only and transform test data[cite: 5].
* **Regularization (`Dropout`):** Randomly zeroes a fraction of activations (`p = 0.2`) during training to reduce co-adaptation and overfitting[cite: 5]. Remember to switch to `model.eval()` during inference/evaluation so dropout is disabled[cite: 5] (and use `torch.no_grad()` to avoid tracking gradients).
* **Imbalanced Classification Metrics:** When positive cases are rare (such as fraud detection), rely on **ROC-AUC**, **Precision**, **Recall**, and **F1-score** (`classification_report`) rather than accuracy alone[cite: 5].
* **Memory Management in C++:** Dynamic allocations (`new float[]`) in `Tensor` require proper cleanup in `~Tensor()` and careful handling of copies/assignments to avoid double-frees or memory leaks[cite: 4].

---

## Files in This Module

* `mlp_foundation.cpp` — C++ from-scratch `Tensor`, `matmul`, `add_bias`, `relu`, and `LinearLayer` forward pass demonstration[cite: 4].
* `mlp.py` — End-to-end PyTorch pipeline for tabular binary classification with scaling, `DataLoader` batching, `BCEWithLogitsLoss` with `pos_weight`, `Dropout`, `Adam`, and evaluation metrics[cite: 5].