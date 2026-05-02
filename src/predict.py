import pickle
import pandas as pd

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed/processed_train.csv"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_data():
    return pd.read_csv(DATA_PATH)

def get_sample(df, index=0):
    row = df.iloc[index]
    X = row.drop(["engine_id", "cycle", "max_cycle", "RUL", "failure"])
    y = row["failure"]
    return X, y

def predict():
    model = load_model()
    df = load_data()
    X, y_true = get_sample(df, index=0)
    
    X_df = pd.DataFrame([X])
    
    pred = model.predict(X_df)[0]
    prob = model.predict_proba(X_df)[0][1]
    
    print("Actual:", int(y_true))
    print("Prediction:", int(pred))
    print("Probability:", float(prob))

if __name__ == "__main__":
    predict()