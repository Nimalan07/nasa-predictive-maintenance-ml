import pickle
import pandas as pd
import os

MODEL_CLF_PATH = "models/model.pkl"
MODEL_REG_PATH = "models/model_reg.pkl"
DATA_PATH = "data/processed/processed_train.csv"

def load_models():
    clf, reg = None, None
    if os.path.exists(MODEL_CLF_PATH):
        with open(MODEL_CLF_PATH, "rb") as f:
            clf = pickle.load(f)
    if os.path.exists(MODEL_REG_PATH):
        with open(MODEL_REG_PATH, "rb") as f:
            reg = pickle.load(f)
    return clf, reg

def load_data():
    return pd.read_csv(DATA_PATH)

def get_sample(df, index=0):
    row = df.iloc[index]
    X = row.drop(["engine_id", "cycle", "max_cycle", "RUL", "failure"], errors="ignore")
    y_clf = row.get("failure", None)
    y_reg = row.get("RUL", None)
    return X, y_clf, y_reg

def predict():
    clf, reg = load_models()
    df = load_data()
    X, y_clf_true, y_reg_true = get_sample(df, index=0)
    
    X_df = pd.DataFrame([X])
    
    print("--- Inference Output ---")
    if y_clf_true is not None:
        print(f"Actual Failure : {int(y_clf_true)}")
    if y_reg_true is not None:
        print(f"Actual RUL     : {float(y_reg_true):.1f} cycles")

    if clf is not None:
        pred_clf = clf.predict(X_df)[0]
        prob_clf = clf.predict_proba(X_df)[0][1]
        print(f"Pred Failure   : {int(pred_clf)}")
        print(f"Failure Prob   : {float(prob_clf):.4f}")

    if reg is not None:
        pred_reg = reg.predict(X_df)[0]
        print(f"Pred RUL       : {float(pred_reg):.2f} cycles")

if __name__ == "__main__":
    predict()