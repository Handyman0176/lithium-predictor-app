import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Define categorical features
cat_features = ['FORMATION', 'BASIN', 'RegionCluster']

# Predict using CatBoost's Pool with cat_features specified
data_pool = Pool(data=input_df, cat_features=cat_features)
prediction = model.predict(data_pool)[0]
model.load_model("catboost_model.cbm")
kmeans = joblib.load("kmeans_model.pkl")

st.title("🔮 Lithium Concentration Predictor")
st.markdown("Input your brine chemistry to predict lithium concentration (mg/L)")

# Define input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=32.0)
        longitude = st.number_input("Longitude", value=-103.0)
        na = st.number_input("Sodium (Na)", value=10000.0)
        k = st.number_input("Potassium (K)", value=100.0)
        mg = st.number_input("Magnesium (Mg)", value=500.0)
        ca = st.number_input("Calcium (Ca)", value=3000.0)
    with col2:
        sr = st.number_input("Strontium (Sr)", value=50.0)
        cl = st.number_input("Chloride (Cl)", value=18000.0)
        tds = st.number_input("Total Dissolved Solids (TDS)", value=300000.0)
        ph = st.number_input("pH", value=6.5)
        iodine = st.number_input("Iodine (I)", value=0.1)
        boron = st.number_input("Boron (B)", value=50.0)

    formation = st.selectbox("Formation", ["San Andres", "Ellenburger", "Wolfcamp", "Delaware", "Other"])
    basin = st.selectbox("Basin", ["Delaware", "Midland", "Central Platform", "Other"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Cluster prediction
    coords = np.array([[latitude, longitude]])
    region_cluster = kmeans.predict(coords)[0]

    # Assemble input
    input_df = pd.DataFrame([{
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "Na": na,
        "K": k,
        "Mg": mg,
        "Ca": ca,
        "Sr": sr,
        "Cl": cl,
        "TDS": tds,
        "PH": ph,
        "I": iodine,
        "B": boron,
        "FORMATION": formation,
        "BASIN": basin,
        "RegionCluster": region_cluster
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"📌 Predicted Lithium Concentration: **{prediction:.2f} mg/L**")
