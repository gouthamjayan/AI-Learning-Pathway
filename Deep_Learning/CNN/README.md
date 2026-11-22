# Convolutional Neural Network (CNN)

A Convolutional Neural Network (CNN) is a neural network built for grid-like data like images, using sliding filters to pick up visual patterns[cite: 6].

## How It Works
- Slide small filters (`Conv2d`) across the image to spot local features and keep spatial structure[cite: 6].
- Apply `ReLU` to add non-linearity and shrink the grid size with `MaxPool2d` so the network focuses on stronger signals[cite: 6].
- Flatten the feature maps and pass them through linear layers (`nn.Linear`) to predict classes[cite: 6].

## What's in This Folder
- **`cnn.py`** — Defines a PyTorch `CNN` model for MNIST (`10` classes), loads and normalizes the data with `DataLoader`, and trains it using `CrossEntropyLoss` and `Adam`[cite: 6].

## Good Things to Remember
- Make sure the flattened size going into `nn.Linear` (`64 * 7 * 7`) matches the output size after your convolution and pooling layers[cite: 6].
- Use `Dropout` to help prevent overfitting[cite: 6], and call `model.train()` during training[cite: 6].
- Move both the model and data batches to the selected `device` (`cuda` or `cpu`)[cite: 6].