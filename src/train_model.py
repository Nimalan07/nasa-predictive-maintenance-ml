import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/processed/processed_train.csv"
MODEL_PATH = "models/model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def split_data(df):
    X = df.drop(columns=["engine_id", "cycle", "max_cycle", "RUL", "failure"])
    y = df["failure"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def save_model(model):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model)

if __name__ == "__main__":
    main()