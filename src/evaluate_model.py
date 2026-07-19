import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/processed_train.csv"
MODEL_CLF_PATH = "models/model.pkl"
MODEL_REG_PATH = "models/model_reg.pkl"

def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_data(df):
    drop_cols = ["engine_id", "cycle", "max_cycle", "RUL", "failure", "prediction", "probability", "predicted_rul"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y_clf = df["failure"]
    y_reg = df["RUL"]
    return train_test_split(X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf)


def load_models():
    clf, reg = None, None
    if os.path.exists(MODEL_CLF_PATH):
        with open(MODEL_CLF_PATH, "rb") as f:
            clf = pickle.load(f)
    if os.path.exists(MODEL_REG_PATH):
        with open(MODEL_REG_PATH, "rb") as f:
            reg = pickle.load(f)
    return clf, reg

def compute_all_metrics(clf, reg, X_test, y_test_clf, y_test_reg):
    metrics = {}

    if clf is not None:
        y_pred_clf = clf.predict(X_test)
        y_prob_clf = clf.predict_proba(X_test)[:, 1]

        metrics["accuracy"] = float(accuracy_score(y_test_clf, y_pred_clf))
        metrics["precision"] = float(precision_score(y_test_clf, y_pred_clf))
        metrics["recall"] = float(recall_score(y_test_clf, y_pred_clf))
        metrics["f1_score"] = float(f1_score(y_test_clf, y_pred_clf))
        metrics["roc_auc"] = float(roc_auc_score(y_test_clf, y_prob_clf))
        metrics["pr_auc"] = float(average_precision_score(y_test_clf, y_prob_clf))
        metrics["confusion_matrix"] = confusion_matrix(y_test_clf, y_pred_clf).tolist()

    if reg is not None:
        y_pred_reg = reg.predict(X_test)

        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
        metrics["mae"] = float(mean_absolute_error(y_test_reg, y_pred_reg))
        metrics["r2"] = float(r2_score(y_test_reg, y_pred_reg))

    return metrics

def print_evaluation_report(metrics):
    print("=" * 50)
    print("      NASA CMAPSS MODEL EVALUATION METRICS      ")
    print("=" * 50)

    print("\n[+] CLASSIFICATION METRICS (Failure Prediction):")
    if "accuracy" in metrics:
        print(f"  * Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  * Precision : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  * Recall    : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  * F1-Score  : {metrics['f1_score']:.4f}")
        print(f"  * ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  * PR-AUC    : {metrics['pr_auc']:.4f}")
    else:
        print("  Classifier model not found.")

    print("\n[+] REGRESSION METRICS (RUL Cycle Prediction):")
    if "rmse" in metrics:
        print(f"  * RMSE      : {metrics['rmse']:.4f} cycles")
        print(f"  * MAE       : {metrics['mae']:.4f} cycles")
        print(f"  * R2 Score  : {metrics['r2']:.4f}")
    else:
        print("  Regressor model not found.")

    print("=" * 50)


def main():
    df = load_data()
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = prepare_data(df)
    clf, reg = load_models()
    metrics = compute_all_metrics(clf, reg, X_test, y_test_clf, y_test_reg)
    print_evaluation_report(metrics)

if __name__ == "__main__":
    main()