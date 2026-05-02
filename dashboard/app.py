import streamlit as st
import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join("models", "model.pkl")

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")

model = pickle.load(open(MODEL_PATH, "rb"))

if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

def color_risk(val):
    if val > 0.7:
        return "background-color: red"
    elif val > 0.4:
        return "background-color: orange"
    else:
        return "background-color: lightgreen"

if uploaded_file and st.session_state.df is None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

if st.session_state.df is not None:

    df = st.session_state.df

    st.subheader("📊 Data Preview")
    st.write(df.head())

    if "prediction" not in df.columns:
        if st.button("🚀 Run Prediction"):
            X = df.drop(columns=["engine_id", "cycle", "max_cycle", "RUL", "failure"], errors="ignore")

            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

            df["prediction"] = preds
            df["probability"] = probs

            st.session_state.df = df

    if "prediction" in df.columns:

        st.subheader("📈 Predictions Overview")

        col1, col2 = st.columns(2)

        risky = df[df["prediction"] == 1]

        with col1:
            st.metric("⚠️ Risky Machines", len(risky))

        with col2:
            risk_percent = (len(risky) / len(df)) * 100
            st.metric("📊 Risk Percentage", f"{risk_percent:.2f}%")

        st.subheader("📋 Styled Preview (Top 200 Rows)")
        st.dataframe(df.head(200).style.map(color_risk, subset=["probability"]))

        st.subheader("⚠️ Top High-Risk Machines")
        st.write(risky.sort_values("probability", ascending=False).head(10))

        st.subheader("📈 Global Failure Trend (Average by Cycle)")
        trend = df.groupby("cycle")["probability"].mean()
        st.line_chart(trend)

        st.subheader("🔍 Engine-Level Analysis")

        engine_ids = df["engine_id"].unique()
        selected_engine = st.selectbox("Select Engine ID", engine_ids)

        engine_df = df[df["engine_id"] == selected_engine].sort_values("cycle")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Total Cycles", int(engine_df["cycle"].max()))

        with col4:
            latest_prob = engine_df["probability"].iloc[-1]
            st.metric("Latest Failure Probability", f"{latest_prob:.4f}")

        if latest_prob > 0.7:
            st.error("🚨 High Failure Risk!")
        elif latest_prob > 0.4:
            st.warning("⚠️ Moderate Risk")
        else:
            st.success("✅ Healthy")

        st.subheader("📈 Engine Failure Trend")
        st.line_chart(engine_df.set_index("cycle")["probability"])

        st.subheader("📋 Recent Engine Data")
        st.dataframe(engine_df.tail(20))

        st.subheader("📊 Full Dataset (No Styling)")
        st.dataframe(df)