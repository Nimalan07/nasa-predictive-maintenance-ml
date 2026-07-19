from flask import Flask, request, jsonify
import pickle
import os
import sys
import pandas as pd

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_CLF_PATH = "models/model.pkl"
MODEL_REG_PATH = "models/model_reg.pkl"


app = Flask(__name__)

model_clf = None
model_reg = None

if os.path.exists(MODEL_CLF_PATH):
    with open(MODEL_CLF_PATH, "rb") as f:
        model_clf = pickle.load(f)

if os.path.exists(MODEL_REG_PATH):
    with open(MODEL_REG_PATH, "rb") as f:
        model_reg = pickle.load(f)

@app.route("/")
def home():
    return jsonify({
        "status": "Predictive Maintenance API Running",
        "classifier_loaded": model_clf is not None,
        "regressor_loaded": model_reg is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        # Drop metadata columns if passed in JSON body
        df_clean = df.drop(columns=["engine_id", "cycle", "max_cycle", "RUL", "failure"], errors="ignore")

        response = {}
        if model_clf is not None:
            pred = model_clf.predict(df_clean)[0]
            prob = model_clf.predict_proba(df_clean)[0][1]
            response["prediction"] = int(pred)
            response["probability"] = float(prob)

        if model_reg is not None:
            rul_pred = model_reg.predict(df_clean)[0]
            response["predicted_rul"] = float(rul_pred)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        from src.evaluate_model import load_data, prepare_data, compute_all_metrics
        df = load_data()
        X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = prepare_data(df)
        metrics = compute_all_metrics(model_clf, model_reg, X_test, y_test_clf, y_test_reg)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)