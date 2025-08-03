import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

# Load model and setup
model = joblib.load("lithium_model.pkl")

# UI Title
st.title("🧪 Lithium Concentration Predictor (Permian Basin)")

# Input features
st.header("📥 Input Sample Data")

latitude = st.number_input("Latitude", value=31.5)
longitude = st.number_input("Longitude", value=-103.5)
na = st.number_input("Na (mg/L)", value=10000.0)
k = st.number_input("K (mg/L)", value=500.0)
mg = st.number_input("Mg (mg/L)", value=300.0)
ca = st.number_input("Ca (mg/L)", value=1500.0)
sr = st.number_input("Sr (mg/L)", value=100.0)
cl = st.number_input("Cl (mg/L)", value=18000.0)
tds = st.number_input("TDS (mg/L)", value=25000.0)
ph = st.number_input("pH", value=6.5)
i = st.number_input("Iodine (I) (mg/L)", value=0.05)
b = st.number_input("Boron (B) (mg/L)", value=5.0)

formation = st.selectbox("Formation", ["Bone Spring", "Delaware", "Wolfcamp"])
basin = st.selectbox("Basin", ["Delaware", "Midland", "Central Platform"])

# RegionCluster (use same kmeans logic)
coords = pd.DataFrame({"LATITUDE": [latitude], "LONGITUDE": [longitude]})
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit with same training coordinates
kmeans.fit(pd.read_csv("Permian_Lithium_CLEANED.csv")[["LATITUDE", "LONGITUDE"]].dropna())
cluster = int(kmeans.predict(coords)[0])

# Prepare dataframe for prediction
data = pd.DataFrame([{
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
    "I": i,
    "B": b,
    "FORMATION": formation,
    "BASIN": basin,
    "RegionCluster": cluster
}])

# Prediction
if st.button("🔮 Predict Lithium Concentration"):
    prediction = model.predict(data)[0]
    st.success(f"🟢 Predicted Lithium Concentration: **{prediction:.2f} mg/L**")
