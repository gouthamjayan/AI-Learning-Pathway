import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = "/home/user/heart_ai_copy/Research/MLP/creditcard.csv"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_csv(path):
    return pd.read_csv(path)

def summarize_dataset(df):
    print("___Dataset Summary___")
    df.info()
    print("___Summary Statistics___")
    print(df.describe().T[["min", "mean", "max"]])

def check_and_remove_duplicates(df):
    duplicated_count = df.duplicated().sum()
    print(f"The number of Duplicated Rows: {duplicated_count}")
    
    if duplicated_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No Duplicates Found!")
    return df

def check_target_distribution(df, target_column):
    counts = df[target_column].value_counts()
    percentages = df[target_column].value_counts(normalize=True) * 100
    print(f"\n___ Target Distribution ({target_column})___")
    print("counts: ")
    print(counts)
    print("percentages (y) :")
    print(percentages.round(2))

def separate_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("___Separation Complete___")
    print(f"features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    return X, y

def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler
    
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=256):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class TransactionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(neural_model, train_loader, epochs=20, lr=0.001):
    print("\n___Training the PyTorch Deep Learning Model___")

    y_train_all = train_loader.dataset.tensors[1]
    num_negatives = (y_train_all == 0).sum().item()
    num_positives = (y_train_all == 1).sum().item()
    pos_weight_val = num_negatives / num_positives
    
    print(f"Calculated pos_weight: {pos_weight_val:.2f}")
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=lr)

    neural_model.to(device)

    for epoch in range(1, epochs + 1):
        neural_model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = neural_model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Training Loss: {epoch_loss:.4f}")

    print("PyTorch Model Trained Successfully!")
    return neural_model

def evaluate_pytorch_model(model, test_loader):
    print("\n___PyTorch Neural Network Evaluation___")
    model.eval()

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probabilities = torch.sigmoid(logits)

            y_pred_list.append(probabilities.cpu().numpy().flatten())
            y_true_list.append(batch_y.numpy().flatten())

    y_pred_proba = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary))

if __name__ == "__main__":
    df = load_csv(DATASET_PATH)
    summarize_dataset(df)
    df = check_and_remove_duplicates(df)
    check_target_distribution(df, "Class")
    X, y = separate_features_and_target(df, "Class")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_split(X, y)
    train_loader, test_loader = create_dataloaders(X_train_scaled, X_test_scaled, y_train, y_test, batch_size=256)
    input_dim = X_train_scaled.shape[1]
    neural_model = TransactionClassifier(input_dim=input_dim)
    model = train_pytorch_model(neural_model, train_loader, epochs=20, lr=0.001)
    evaluate_pytorch_model(model, test_loader)