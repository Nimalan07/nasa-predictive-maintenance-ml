import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = "data/processed/processed_train.csv"
MODEL_CLF_PATH = "models/model.pkl"
MODEL_REG_PATH = "models/model_reg.pkl"

def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_data(df):
    X = df.drop(columns=["engine_id", "cycle", "max_cycle", "RUL", "failure"], errors="ignore")
    y_clf = df["failure"]
    y_reg = df["RUL"]
    return train_test_split(X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf)

def train_classification_model(X_train, y_train):
    clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_regression_model(X_train, y_train):
    reg = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    reg.fit(X_train, y_train)
    return reg

def save_models(clf, reg):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_CLF_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_REG_PATH, "wb") as f:
        pickle.dump(reg, f)
    print(f" Classification model saved to {MODEL_CLF_PATH}")
    print(f" Regression model saved to {MODEL_REG_PATH}")

def main():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = prepare_data(df)

    print("Training Classification Model (Failure Prediction)...")
    clf = train_classification_model(X_train, y_train_clf)
    
    print("Training Regression Model (RUL Prediction)...")
    reg = train_regression_model(X_train, y_train_reg)

    save_models(clf, reg)

    # Evaluate metrics
    y_pred_clf = clf.predict(X_test)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    prec = precision_score(y_test_clf, y_pred_clf)

    y_pred_reg = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print("\n--- Model Evaluation Results ---")
    print(f" Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f" Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f" RMSE      : {rmse:.4f}")
    print(f" MAE       : {mae:.4f}")
    print(f" R2 Score  : {r2:.4f}")


if __name__ == "__main__":
    main()