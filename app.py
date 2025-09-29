import streamlit as st
import joblib
import pandas as pd

# ---- Title ----
st.title("🌊 Flood Prediction App")
st.write("Enter feature values below, then click **Predict** to see flood risk.")

# ---- Load preprocessor, PCA, and model ----
pre = joblib.load("preprocessor.joblib")
pca = joblib.load("pca.joblib")
model = joblib.load("rf_model.joblib")

# ---- Define feature names exactly as in your training data ----
feature_names = [
    "Rainfall_mm",
    "Temperature_c",
    "Humidity_percent",
    "Latitude",
    "Longitude"
]

# ---- Input fields ----
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(
        f"{feature}:",
        value=0.0,
        format="%.4f"
    )

# ---- Predict button ----
if st.button("Predict"):
    try:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([inputs])

        # Preprocess input
        X_new_pre = pre.transform(input_df)
        X_new_pca = pca.transform(X_new_pre)

        # Predict probability
        proba = model.predict_proba(X_new_pca)[0][1]  # probability of flood
        st.write(f"Flood probability: **{proba:.2f}**")

        # Show result
        if proba >= 0.5:
            st.error("⚠️ Flood is likely to occur!")
        else:
            st.success("✅ No flood predicted.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
