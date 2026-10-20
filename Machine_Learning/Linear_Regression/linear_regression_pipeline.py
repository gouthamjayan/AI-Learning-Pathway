import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATASET_PATH = "path_to_your_data.csv"

def load_csv(path):
    return pd.read_csv(path)

# Summarizing the dataset
def summarize_dataset(df):
    print("Summarizing the dataset")
    df.info()
    print("Summary Statistics: ")
    print(df.describe().T[["min", "mean", "max"]])

# Checking for duplicates
def duplicate_check(df):
    duplicated_count = df.duplicated().sum()  # Added parentheses
    if duplicated_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)  # Reassigned df
        print(f"The number of duplicated rows: {duplicated_count}")
    else:
        print("No duplicates found.")
    return df

# Understanding the target variable (For continuous targets in regression)
def check_target_distribution(df, target_column):
    print(f"\n______ Target Distribution ({target_column}) ______")
    print(f"Mean: {df[target_column].mean():.2f}")
    print(f"Min: {df[target_column].min():.2f} | Max: {df[target_column].max():.2f}")

# Separating input features from the target
def separate_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("Separation Completed!")
    print(f"Input Features Shape: {X.shape}")
    print(f"Output Target Shape: {y.shape}")
    return X, y

# Splitting Data for training and testing
def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
        # Removed stratify=y because Linear Regression expects continuous targets
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converting back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("Dataset split into training and testing data!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Actual Training Process
def train_model(X_train, y_train):
    print("Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluation Metrics Section (Missing Piece Added)
def evaluate_model(model, X_test, y_test):
    print("\n______ Model Evaluation ______")
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f} (Higher is better, max 1.0)")
    print(f"RMSE:     {rmse:.4f} (Lower is better)")
    print(f"MAE:      {mae:.4f} (Lower is better)")

if __name__ == "__main__":
    df = load_csv(DATASET_PATH)

    summarize_dataset(df)

    df = duplicate_check(df)

    TARGET_COL = "your_target_column_name"  # Replace with your actual target column name
    
    check_target_distribution(df, TARGET_COL)

    X, y = separate_features_and_target(df, TARGET_COL)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_split(X, y)

    model = train_model(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)