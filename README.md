#  Predictive Maintenance System (NASA CMAPSS)

## 📊 Overview
This project builds an end-to-end machine learning system to predict machine failures using time-series sensor data from the NASA CMAPSS dataset.

The system forecasts failures in advance, enabling proactive maintenance and reducing downtime.

---

## 🎯 Objective
- Predict machine failure before it occurs
- Use time-series sensor data (temperature, pressure, vibration)
- Provide interactive visualization and analysis

---

## 🧠 Key Features
- Time-series feature engineering (RUL calculation)
- Classification model using XGBoost
- Failure probability prediction
- Interactive Streamlit dashboard
- Per-engine analysis with trend visualization
- Dockerized for reproducible deployment

---

## 🏗️ Project Structure
```
predictive-maintenance-ml/
│
├── data/
│ ├── raw/ # Original NASA dataset
│ └── processed/ # Cleaned and feature-engineered data
│
├── src/
│ ├── data_preprocessing.py # Data cleaning + RUL creation
│ ├── train_model.py # Model training
│ ├── evaluate_model.py # Evaluation metrics
│ ├── predict.py # Inference logic
│
├── models/
│ └── model.pkl # Trained model
│
├── dashboard/
│ └── app.py # Streamlit dashboard
│
├── Dockerfile # Container setup
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore
```

---

## ⚙️ Tech Stack
- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Streamlit
- Docker

---
## 🚀 How to Run

###  Option 1: Run Locally
```
pip install -r requirements.txt
streamlit run dashboard/app.py
```
### Option 2: Run with Docker
``` docker build -t predictive-maintenance .
docker run -p 8501:8501 predictive-maintenance
```
Then open:
http://localhost:8501

---
### 📈 Dashboard Features
- Upload dataset
- Predict failure probability
- View risky machines
- Engine-level analysis
- Failure trend visualization
---
### 🧠 Machine Learning Approach & Benchmark Evaluation

The system evaluates engine health using a dual framework:
1. **Binary Failure Classification** (Predicting whether an engine will fail within 30 cycles)
2. **Remaining Useful Life (RUL) Regression** (Predicting exact remaining operating cycles)

#### 📊 Evaluated Metrics (Supported & Claimed):
- **🎯 Classification Metrics (Failure Risk)**:
  - **Accuracy**: Measure of total overall correct predictions across all healthy and failing engines.
  - **Precision**: Ratio of true positive failures out of all predicted failures (minimizes false alarms).
  - **Recall & F1-Score**: Evaluates true failure detection rate and overall harmonic balance.
- **📈 Regression Metrics (RUL Forecasting)**:
  - **RMSE (Root Mean Squared Error)**: Measures standard deviation of prediction errors in operating cycles (penalizes large errors).
  - **MAE (Mean Absolute Error)**: Average absolute cycle difference between predicted and actual RUL.
  - **R² Score (Coefficient of Determination)**: Proportion of variance in RUL explained by engine sensor telemetry.

---
### 💡 Key Insight

- The model predicts increasing failure probability and decreasing continuous RUL as the machine approaches end-of-life cycles.

---
### 📦 Deployment
- Containerized using Docker
- Can be deployed on:
  - Streamlit Cloud
  - Render
  - AWS / GCP
---
### 📌 Future Improvements
- SHAP explainability
- Real-time IoT data streaming
- Multi-engine comparison dashboard
---
## 👨‍💻 Author
NIMALAN MANI M

---
## ⭐ If you like this project
Give it a star ⭐ on GitHub
