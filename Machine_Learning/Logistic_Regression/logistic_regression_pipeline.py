import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATASET_PATH = ""

#loading the csv file as Dataframe
def load_csv(path):
    df = pd.read_csv(path)
    return df

#summarizing the dataset
def summarize_dataset(df):
    print("Dataset info:")
    df.info()
    print("Summary Statitics:")
    print(df.describe().T[["min","mean","max"]])

#removing duplicate values
def check_and_remove_duplicates(df):
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"No of Duplicate rows: {duplicate_count}")
        df = df.drop_duplicates().reset_index(drop = True)
        print("Duplicate rows removed successfully!")
    else:
        print("No duplicate Rows Found!")

    return df

#Understanding the target variable
def check_target_distribution(df,target_column):
    counts = df[target_column].value_counts()
    percentages = (counts / len(df)) * 100
    print(f"Target distribution: ")
    print("counts")
    print(counts)
    print("percentages: ")
    print(percentages)

#seperating input features form the target
def seperate_features_and_target(df,target_column):
    X = df.drop(columns = [target_column])
    y = df[target_column]

    print("Seperation Complete")
    print(f"Input Features Shape -> {X.shape}")
    print(f"Output Features Shape -> {y.shape}")

    return X,y

#Splitting data from training and splitting
def preprocess_and_split(X,y,test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size = test_size
        random_state = random_state
        stratify = y
    )

    Scaler = StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)

    #Converting back to Dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = X.columns)

    print("Dataset Splitted into training and testing data.")
    return X_train_scaled,X_test_scaled,y_train,y_test,Scaler

#Actual Training Process
def train_model(X_train, y_train):
    print("Training Logistic Regression Model!")
    
    model = LogisticRegression(
        max_iter = 1000, class_weight = 'balanced' , random_state = 42
    )
    model.fit(X_train, y_train)
    print("Logistic Regression Model Trained Successfully!")
    return model


def evaluate_model(y_test, y_pred, y_proba = None):
    print("Linear regression Model Evaluation")

    print("Classification Report:")
    print(classification_report(y_test,y_pred,digits = 4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n Confusion Matrix Breakdown:")
    print(f"True Negative (Healthy flagged as healthy), {cm[0][0]}")
    print(f"False Negative (Missed Diseases), {cm[1][0]}")
    print(f"True Positive (Identifying correctly people with disease, {cm[1][1]})")
    print(f"False Positive (Marking healthy people as unhealthy), {cm[0][1]}")

    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc_score:.4f}")