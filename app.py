import streamlit as st
import joblib
import pandas as pd

# Load model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

pipeline = joblib.load(MODELS_DIR / "loan_default_pipeline.pkl")
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Risk Prediction")

st.write("Upload a CSV file to predict loan default risk.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

   # Predict
    prediction = pipeline.predict(data)

    result = data.copy()
    result["Prediction"] = prediction
    st.success(f"✅ Prediction completed successfully for {len(result)} records.")

    st.subheader("Prediction Results")
    st.dataframe(result.head(100))

    csv = result.head(100).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )