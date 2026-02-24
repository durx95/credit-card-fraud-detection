import streamlit as st
import joblib 
import numpy as np
import pandas as pd

# load model 
model = joblib.load("fraud_model.pkl")
scaler =  joblib.load("scaler.pkl")

st.title("💳 Cradit Card Fraud Detection")

uploaded_file = st.file_uploader("upload transaction csv")

if uploaded_file is not None:
    data =pd.read_csv(uploaded_file)
   

    data[['Time','Amount']] = scaler.transform(data[['Time','Amount']])

    data = data[model.feature_names_in_]

    prediction = model.predict(data)


    data["prediction"] = prediction
    st.write(data)