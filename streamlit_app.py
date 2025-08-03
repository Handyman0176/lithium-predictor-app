import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load model and setup
# Load in app.py
import cloudpickle
with open("lithium_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

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

formation_options = sorted([
    "Abo", "Abo Reef", "Atoka", "Bell Canyon", "Bend", "Blinebry Clearfork",
    "Blinebry Grayburg San And", "C Pecos Wolfcamp Lower", "Caddo", "Cambrian",
    "Canyon", "Canyon Sand", "Canyon Upper", "Cherry Canyon", "Chinle",
    "Cisco", "Clear Fork", "Coleman Junction", "Dean", "Delaware",
    "Delaware Bell Canyon", "Delaware Sd.", "Detrital Devonian", "Devonian",
    "Ellenberger", "Ellenburger", "Flippen", "Formation", "Fusselman",
    "Gardner", "Glorieta", "Grayburg", "Grayburg - San Andres", "Holt",
    "Jackson", "Limey Shale", "Lo Abo", "Lower Devonian", "Mckee",
    "Mid Delaware", "Mississippian", "Montoya Ellenburger", "Pnl", "Penn",
    "Pennsylvanian", "Pennsylvanian Odom", "Pennsylvanian Strawn",
    "Permian", "Permian Detrital", "Permian Lower", "San Andres",
    "San Angelo", "Santa Rosa", "Seven Rivers", "Silurian", "Simpson",
    "Spraberry", "Spraberry", "Strawn", "Strawn Reef", "Tubb Lower",
    "Unknown", "Waddell", "Waddell Simpson", "Wilberns", "Wolfcamp",
    "Wolfcamp Abo", "Wolfcamp Penn", "Wolfcamp Sterling", "Wichita",
    "Wichita Albany", "Yates"
])
# Fixed basin
basin = "Permian"

# Updated formation dropdown
formation = st.selectbox("Formation", formation_options)

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
