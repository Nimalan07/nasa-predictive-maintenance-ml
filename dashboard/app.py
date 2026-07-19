import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import sys

# Ensure parent path is in sys.path to import evaluate_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_CLF_PATH = os.path.join("models", "model.pkl")
MODEL_REG_PATH = os.path.join("models", "model_reg.pkl")

st.set_page_config(
    page_title="NASA Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar elements via CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 Predictive Maintenance Dashboard (NASA CMAPSS)")

@st.cache_resource
def load_all_models():
    clf, reg = None, None
    if os.path.exists(MODEL_CLF_PATH):
        with open(MODEL_CLF_PATH, "rb") as f:
            clf = pickle.load(f)
    if os.path.exists(MODEL_REG_PATH):
        with open(MODEL_REG_PATH, "rb") as f:
            reg = pickle.load(f)
    return clf, reg

model, reg_model = load_all_models()

if "df" not in st.session_state:
    st.session_state.df = None
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

# Main Controls
input_col1, input_col2 = st.columns([3, 1])

with input_col1:
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset for Analysis & Batch Inference", type=["csv"])

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_default = st.checkbox("⚙️ Use Default NASA CMAPSS Dataset", value=False)

# Read CSV only when file changes
if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.current_file_id != file_id:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.current_file_id = file_id
elif use_default:
    if st.session_state.current_file_id != "default":
        try:
            from src.evaluate_model import load_data
            st.session_state.df = load_data()
            st.session_state.current_file_id = "default"
        except Exception as e:
            st.error(f"Error loading default dataset: {e}")
else:
    st.session_state.df = None
    st.session_state.current_file_id = None

def color_risk(val):
    if val > 0.7:
        return "background-color: #ff4d4d; color: white;"
    elif val > 0.4:
        return "background-color: #ffa64d; color: black;"
    else:
        return "background-color: #70db70; color: black;"

# Render Dashboard only when data is loaded
if st.session_state.df is not None:
    df = st.session_state.df

    # Calculate Evaluation Metrics dynamically if dataset contains target ground truth
    has_clf_ground_truth = "failure" in df.columns
    has_reg_ground_truth = "RUL" in df.columns

    if has_clf_ground_truth or has_reg_ground_truth:
        st.subheader("🎯 Dataset Model Evaluation Metrics")
        try:
            from src.evaluate_model import prepare_data, compute_all_metrics
            X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = prepare_data(df)
            metrics = compute_all_metrics(model, reg_model, X_test, y_test_clf, y_test_reg)

            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            with metric_col1:
                st.metric("🎯 Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            with metric_col2:
                st.metric("🎯 Precision", f"{metrics.get('precision', 0)*100:.2f}%")
            with metric_col3:
                st.metric("📐 RMSE", f"{metrics.get('rmse', 0):.2f} cycles")
            with metric_col4:
                st.metric("📐 MAE", f"{metrics.get('mae', 0):.2f} cycles")
            with metric_col5:
                st.metric("📊 R² Score", f"{metrics.get('r2', 0):.4f}")

        except Exception as e:
            st.info("Custom uploaded dataset loaded. (Evaluation metrics available when target ground-truth columns are present).")

    st.markdown("---")
    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Run Prediction Button
    if "prediction" not in df.columns:
        if st.button("🚀 Run Prediction (Failure & RUL Forecast)"):
            with st.spinner("Calculating predictions..."):
                drop_cols = ["engine_id", "cycle", "max_cycle", "RUL", "failure", "prediction", "probability", "predicted_rul"]
                X = df.drop(columns=drop_cols, errors="ignore")


                if model is not None:
                    preds = model.predict(X)
                    probs = model.predict_proba(X)[:, 1]
                    df["prediction"] = preds
                    df["probability"] = probs

                if reg_model is not None:
                    rul_preds = reg_model.predict(X)
                    df["predicted_rul"] = np.maximum(0, rul_preds).round(1)

                st.session_state.df = df
                st.rerun()

    if "prediction" in df.columns:
        st.subheader("📈 Predictions Overview")

        col1, col2, col3 = st.columns(3)

        risky = df[df["prediction"] == 1]

        with col1:
            st.metric("⚠️ Risky Machines", len(risky))

        with col2:
            risk_percent = (len(risky) / len(df)) * 100
            st.metric("📊 Risk Percentage", f"{risk_percent:.2f}%")

        with col3:
            if "predicted_rul" in df.columns:
                avg_rul = df["predicted_rul"].mean()
                st.metric("⏱️ Avg Predicted RUL", f"{avg_rul:.1f} cycles")

        st.subheader("📋 Styled Preview (Top 200 Rows)")
        st.dataframe(df.head(200).style.map(color_risk, subset=["probability"]))

        st.subheader("⚠️ Top High-Risk Machines")
        display_cols = [c for c in ["engine_id", "cycle", "probability", "predicted_rul", "RUL", "failure"] if c in df.columns]
        st.write(risky.sort_values("probability", ascending=False)[display_cols].head(10))

        st.subheader("📈 Global Failure Trend (Average Failure Probability by Cycle)")
        trend = df.groupby("cycle")["probability"].mean()
        st.line_chart(trend)

        st.subheader("🔍 Engine-Level Analysis")

        engine_ids = df["engine_id"].unique()
        selected_engine = st.selectbox("Select Engine ID", engine_ids)

        engine_df = df[df["engine_id"] == selected_engine].sort_values("cycle")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric("Total Cycles", int(engine_df["cycle"].max()))

        with col4:
            latest_prob = engine_df["probability"].iloc[-1]
            st.metric("Latest Failure Probability", f"{latest_prob:.4f}")

        with col5:
            if "predicted_rul" in engine_df.columns:
                latest_rul = engine_df["predicted_rul"].iloc[-1]
                st.metric("Latest Predicted RUL", f"{latest_rul:.1f} cycles")

        if latest_prob > 0.7:
            st.error("🚨 High Failure Risk!")
        elif latest_prob > 0.4:
            st.warning("⚠️ Moderate Risk")
        else:
            st.success("✅ Healthy")

        st.subheader("📈 Engine Failure Trend & RUL Forecast")
        chart_df = engine_df.set_index("cycle")
        if "predicted_rul" in chart_df.columns:
            st.line_chart(chart_df[["probability", "predicted_rul"]])
        else:
            st.line_chart(chart_df["probability"])

        st.subheader("📋 Recent Engine Data")
        st.dataframe(engine_df.tail(20))

        st.subheader("📊 Feature Importance")
        if model is not None and hasattr(model, "get_booster"):
            feature_names = model.get_booster().feature_names
            importance = model.feature_importances_
            min_len = min(len(feature_names), len(importance))

            importance_df = pd.DataFrame({
                "feature": feature_names[:min_len],
                "importance": importance[:min_len]
            }).sort_values("importance", ascending=False)

            st.write("🔝 Top Important Features")
            st.dataframe(importance_df.head(10))
            st.bar_chart(importance_df.set_index("feature"))

        st.subheader("📊 Full Dataset (No Styling)")
        st.dataframe(df)

else:
    st.info("ℹ️ **No dataset loaded.** Upload a CSV file above or check 'Use Default NASA CMAPSS Dataset' to run predictions.")

