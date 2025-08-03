import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool

# 🎯 Load models
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")  # CatBoost native format

kmeans = joblib.load("kmeans_model.pkl")  # KMeans for region clustering

# 🎯 Define categorical features (MUST MATCH TRAINING)
cat_features = ['FORMATION', 'BASIN', 'RegionCluster']

# 🎯 Define full column order (MUST MATCH TRAINING)
all_features = ['LATITUDE', 'LONGITUDE', 'Na', 'K', 'Mg', 'Ca', 'Sr', 'Cl',
                'TDS', 'PH', 'I', 'B', 'FORMATION', 'BASIN', 'RegionCluster']

# -------------------------
# 🌐 Streamlit Interface
# -------------------------
st.title("🔮 Lithium Concentration Predictor")
st.markdown("Input brine chemistry to estimate lithium concentration (mg/L)")

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

# -------------------------
# 🔮 Prediction Logic
# -------------------------
if submitted:
    # 🔹 Predict region cluster
    coords = np.array([[latitude, longitude]])
    region_cluster = str(kmeans.predict(coords)[0])  # cast to str to treat as categorical

    # 🔹 Assemble input DataFrame
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

    # 🔹 Reorder columns
    input_df = input_df[all_features]

    # 🔹 Ensure categorical columns are strings
    for col in cat_features:
        input_df[col] = input_df[col].astype(str)

    # 🔹 Predict using Pool
    input_pool = Pool(data=input_df, cat_features=cat_features)
    prediction = model.predict(input_pool)[0]

    # 🎉 Display result
    st.success(f"📌 Predicted Lithium Concentration: **{prediction:.2f} mg/L**")
