import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()  # <--- FIX 1: Added mandatory super() call
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # <--- FIX 2: kernel_size
            nn.ReLU(),                                                                    # <--- FIX 3: Added comma
            nn.MaxPool2d(kernel_size=2, stride=2),                                        # <--- FIX 3: Added comma

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),# <--- FIX 2: kernel_size
            nn.ReLU(),                                                                    # <--- FIX 3: Added comma
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),   # Prevents overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =================================-
# Set up data and hyperparameters
# =================================-

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

# Transforming images into tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()                 # <--- FIX 4: Corrected spelling (CrossEntropyLoss)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 2
model.train()                                     # <--- FIX 5: Corrected mode -> model

print("Starting the training Pipeline...")
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 200 == 199:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/200:.4f}")
            running_loss = 0.0

print("\nTraining complete! Pipeline verified successfully.")