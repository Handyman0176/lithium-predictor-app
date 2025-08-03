# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# Load model (CatBoost from .cbm file)
import joblib
model = joblib.load("catboost_model.pkl")

# Load KMeans model
kmeans = joblib.load("kmeans_model.pkl")

# Set feature names
numerical_features = ['LATITUDE', 'LONGITUDE', 'Na', 'K', 'Mg', 'Ca', 'Sr', 'Cl', 'TDS', 'PH', 'I', 'B']
categorical_features = ['FORMATION', 'BASIN']
all_features = numerical_features + categorical_features + ['RegionCluster']

st.set_page_config(page_title="Lithium Predictor", layout="centered")
st.title("🔮 Real-Time Lithium Predictor (CatBoost Model)")

st.markdown("Enter the following input parameters to predict lithium concentration (mg/L).")

# --- User Input ---
user_input = {}

# Numerical inputs
for col in numerical_features:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# FORMATION dropdown with mapping
formation_mapping = {
    "Wolfcamp (All)": "Wolfcamp",
    "Clear Fork (All)": "Clear Fork",
    "Devonian (All)": "Devonian",
    "Ellenburger (All)": "Ellenburger",
    "San Andres (All)": "San Andres",
    "Delaware (All)": "Delaware",
    "Canyon (All)": "Canyon",
    "Strawn (All)": "Strawn",
    "Pennsylvanian (All)": "Pennsylvanian",
    "Fusselman": "Fusselman",
    "Cisco": "Cisco",
    "Cherry Canyon": "Cherry Canyon",
    "Glorieta": "Glorieta",
    "Dean": "Dean",
    "Spraberry": "Spraberry",
    "Atoka": "Atoka",
    "Montoya Ellenburger": "Montoya Ellenburger",
    "Bend": "Bend",
    "Lo Abo": "Lo Abo",
    "Bell Canyon": "Bell Canyon",
    "Unknown": "Unknown"
}
selected_ui_value = st.selectbox("FORMATION", list(formation_mapping.keys()), index=0)
user_input["FORMATION"] = formation_mapping[selected_ui_value]
user_input["BASIN"] = "Permian"

# --- Auto-assign Region Cluster ---
latlon = [[user_input["LATITUDE"], user_input["LONGITUDE"]]]
user_input["RegionCluster"] = int(kmeans.predict(latlon)[0])
st.markdown(f"🧭 **Auto-Assigned Region Cluster**: `{user_input['RegionCluster']}`")

# --- Create DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Type Enforcement ---
for col in numerical_features + ['RegionCluster']:
    input_df[col] = input_df[col].astype(float)
for col in categorical_features:
    input_df[col] = input_df[col].astype(str)

# --- Reorder Columns (important for CatBoost) ---
input_df = input_df[all_features]

# --- Debug ---
st.write("📋 Model Input Preview:")
st.dataframe(input_df)
st.write("🧬 Data Types:")
st.write(input_df.dtypes)

# --- Predict ---
if st.button("🔍 Predict Lithium"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🧪 Predicted Lithium Concentration: **{prediction:.2f} mg/L**")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)
