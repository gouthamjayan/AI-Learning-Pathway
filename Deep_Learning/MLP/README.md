# Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) stacks layers of neurons with activation functions so a model can learn patterns beyond a simple straight line[cite: 4, 5].

## How It Works
- Pass inputs through a linear layer (weights and bias) to mix features together[cite: 4, 5].
- Apply an activation function like `ReLU` to add non-linearity so the network can learn curved or complex patterns[cite: 4, 5].
- Repeat across layers, compute loss, and update weights during training[cite: 5].

## What's in This Folder
- **`mlp_foundation.cpp`** — A C++ forward-pass building block with a custom `Tensor` struct, matrix multiplication (`matmul`), bias addition (`add_bias`), `relu`, and `LinearLayer`[cite: 4].
- **`mlp.py`** — A PyTorch training and evaluation script using `TransactionClassifier`, `StandardScaler`, `DataLoader`, `Dropout`, `BCEWithLogitsLoss` with `pos_weight`, and `Adam`[cite: 5].

## Good Things to Remember
- Scale features with `StandardScaler` so neural network training stays stable[cite: 5].
- Use `Dropout` during training to reduce overfitting, and switch to `model.eval()` when evaluating[cite: 5].
- When classes are imbalanced, check `ROC-AUC` and the classification report rather than relying on accuracy alone[cite: 5].