import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

st.title("🔮 ML Prediction App")

st.write("Enter input values:")

# -------------------------------
# USER INPUT (AUTO GENERATE)
# -------------------------------
input_data = {}

for col in model_columns:
    if "_" in col:
        # skip one-hot encoded columns
        continue
    value = st.number_input(f"{col}", value=0.0)
    input_data[col] = value

input_df = pd.DataFrame([input_data])

# -------------------------------
# PREPROCESS INPUT
# -------------------------------
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]:.2f}")