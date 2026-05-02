from flask import Flask, request, jsonify
import pickle
import pandas as pd

MODEL_PATH = "models/model.pkl"

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Predictive Maintenance API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    return jsonify({
        "prediction": int(pred),
        "probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)