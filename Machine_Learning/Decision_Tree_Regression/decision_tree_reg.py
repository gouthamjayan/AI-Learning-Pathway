from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_housing_data():
    print("Loading California Housing Dataset as DataFrame")
    housing = fetch_california_housing(as_frame=True)
    return housing.frame

def summarize_dataset(df):
    print("Summarizing the dataset:")
    df.info()
    print("Summary Statistics:")
    print(df.describe().T[["min", "mean", "max"]])

def duplicate_check(df):
    duplicated_count = df.duplicated().sum()
    if duplicated_count > 0:
        print(f"The no of duplicated rows : {duplicated_count}")
        df = df.drop_duplicates().reset_index(drop = True)
        print("Duplicates Removed Successfully!")
    else:
        print("No Duplicates Found")
    return df

def separate_features_and_target(df, target_column):
    X = df.drop(columns = [target_column])
    y = df[target_column]
    print("Separation Complete.")
    print(f"Input Features Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")
    return X, y

def split_data(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state
    )
    print("Dataset Split into Training Data and Testing Data!")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training Decision Tree Regressor:")
    model = DecisionTreeRegressor(max_depth = 30, random_state = 42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("\n______ Model Evaluation ______")
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R² Score: {r2:.4f} (Higher is better, max 1.0)")
    print(f"RMSE:     {rmse:.4f} (Lower is better)")

if __name__ == "__main__":
    df = load_housing_data()
    summarize_dataset(df)
    df = duplicate_check(df)
    X, y = separate_features_and_target(df, "MedHouseVal")
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)