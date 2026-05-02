import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_score, average_precision_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/processed_train.csv"
MODEL_PATH = "models/model.pkl"

def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_data(df):
    X = df.drop(columns=["engine_id", "cycle", "max_cycle", "RUL", "failure"])
    y = df["failure"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Precision:", precision_score(y_test, y_pred))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = load_model()
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()