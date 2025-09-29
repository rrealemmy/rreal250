
import streamlit as st
import joblib
import pandas as pd

st.title('Flood Prediction Demo')

pre = joblib.load('preprocessor.joblib')
pca = joblib.load('pca.joblib')
model = joblib.load('rf_model.joblib')

uploaded = st.file_uploader('Upload CSV with same feature columns', type=['csv'])
if uploaded is not None:
    df_new = pd.read_csv(uploaded)
    X_new_pre = pre.transform(df_new)
    X_new_pca = pca.transform(X_new_pre)
    preds = model.predict(X_new_pca)
    df_new['prediction'] = preds
    st.dataframe(df_new)
    st.write('Predictions added as column `prediction` (0 = no flood, 1 = flood)')
