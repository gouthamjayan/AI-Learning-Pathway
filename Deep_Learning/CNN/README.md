# Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)** is a neural network architecture designed for grid-like data—such as 2D images—where spatial structure and local patterns matter[cite: 6]. Instead of connecting every input pixel to every neuron like a standard dense layer, CNNs slide learnable filters (kernels) across the input to extract local features and build hierarchical representations[cite: 6].

---

## Core Intuition & Architecture

### 1. Convolutional & Pooling Blocks (`self.features`)
In `cnn.py`, the feature extractor stacks convolutional layers, ReLU activations, and max-pooling layers on single-channel (`in_channels=1`) MNIST inputs[cite: 6]:

* **2D Convolution (`nn.Conv2d`):** Slides a learnable filter over spatial dimensions using `kernel_size`, `stride`, and `padding`[cite: 6].
  * First block: `in_channels=1 -> out_channels=32`, `kernel_size=3`, `stride=1`, `padding=1`[cite: 6].
  * Second block: `in_channels=32 -> out_channels=64`, `kernel_size=3`, `stride=1`, `padding=1`[cite: 6].
  * `padding=1` with a `3x3` kernel and `stride=1` preserves spatial height and width across the convolution[cite: 6].
* **ReLU Activation (`nn.ReLU`):** Introduces non-linearity after each convolution while keeping negative activations at `0`[cite: 6].
* **Max Pooling (`nn.MaxPool2d`):** Downsamples spatial dimensions (`kernel_size=2, stride=2` halves height and width) while keeping the strongest activation in each `2x2` window, reducing computation and adding local translation tolerance[cite: 6].

### 2. Spatial Shape Flow & Classifier Head (`self.classifier`)
Tracing an MNIST image of shape `(1, 28, 28)` through `self.features` in `cnn.py`[cite: 6]:

1. Input: `(1, 28, 28)`[cite: 6]
2. `Conv2d(1, 32, kernel_size=3, padding=1)` $\rightarrow$ `(32, 28, 28)`[cite: 6]
3. `MaxPool2d(kernel_size=2, stride=2)` $\rightarrow$ `(32, 14, 14)`[cite: 6]
4. `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `(64, 14, 14)`[cite: 6]
5. `MaxPool2d(kernel_size=2, stride=2)` $\rightarrow$ `(64, 7, 7)`[cite: 6]

Then `self.classifier` flattens `(64, 7, 7)` into `64 * 7 * 7 = 3136` features and maps through dense layers to `num_classes=10` raw class logits[cite: 6]:

$$\text{Flatten } (64 \times 7 \times 7) \rightarrow \text{Linear}(3136, 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.5) \rightarrow \text{Linear}(128, 10)$$

### 3. Training Setup (`cnn.py`)
* **Preprocessing:** Converts images to tensors (`transforms.ToTensor()`) and normalizes with MNIST mean `(0.1307,)` and std `(0.3081,)` (`transforms.Normalize`)[cite: 6].
* **Loss & Optimizer:** Uses `nn.CrossEntropyLoss()` (which combines log-softmax and negative log-likelihood loss for multi-class logits) with `optim.Adam(model.parameters(), lr=0.001)` and mini-batches (`batch_size=64`)[cite: 6].
* **Training Mode:** Calls `model.train()` before the loop so `Dropout(0.5)` is active during training[cite: 6].

---

## Key Practical Considerations

* **Parameter Sharing & Inductive Bias:** Convolutions reuse the same filter weights across spatial positions, drastically cutting parameters compared to flattening raw pixels directly into a dense layer and encoding spatial locality.
* **Shape Alignment into `nn.Linear`:** `64 * 7 * 7` in `nn.Linear(64 * 7 * 7, 128)` depends directly on the input image size (`28x28`) and pooling layers—changing input resolution or pooling strides requires updating that flattened dimension[cite: 6].
* **Regularization (`Dropout(0.5)`):** Randomly drops half of the hidden units during training to curb overfitting in the dense head; switch to `model.eval()` when evaluating on validation/test sets[cite: 6].
* **Hardware Acceleration:** Automatically selects `"cuda"` when available and falls back to `"cpu"`, moving both `model` and `(images, labels)` batches to `device`[cite: 6].

---

## Files in This Module

* `cnn.py` — PyTorch CNN architecture (`CNN`), MNIST data pipeline (`DataLoader` + normalization), and training loop with `CrossEntropyLoss` and `Adam`[cite: 6].