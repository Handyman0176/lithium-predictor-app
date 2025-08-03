import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import joblib
import pydeck as pdk

# ------------------
# Load model + kmeans
# ------------------
with open("lithium_model.pkl", "rb") as f:
    model = cloudpickle.load(f)
kmeans = joblib.load("region_kmeans.pkl")

# ------------------
# Color scale function
# ------------------
def lithium_to_color(value, min_val=0, max_val=200):
    normalized = min(max((value - min_val) / (max_val - min_val), 0), 1)
    red = int(255 * normalized)
    blue = int(255 * (1 - normalized))
    return [red, 50, blue, 160]

# ------------------
# UI
# ------------------
st.title("🧪 Lithium Concentration Predictor (Permian Basin)")
mode = st.radio("Choose Input Mode:", ["Manual Input", "Upload CSV"])

# ------------------
# Mode: Manual Input
# ------------------
if mode == "Manual Input":
    st.header("📥 Enter Sample Data Manually")

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
        "Spraberry", "Strawn", "Strawn Reef", "Tubb Lower", "Unknown", "Waddell",
        "Waddell Simpson", "Wilberns", "Wolfcamp", "Wolfcamp Abo", "Wolfcamp Penn",
        "Wolfcamp Sterling", "Wichita", "Wichita Albany", "Yates"
    ])
    formation = st.selectbox("Formation", formation_options)
    basin = "Permian"

    if st.button("🔮 Predict Lithium Concentration", key="manual_btn"):
        cluster = int(kmeans.predict(pd.DataFrame({"LATITUDE": [latitude], "LONGITUDE": [longitude]}))[0])
        data = pd.DataFrame([{
            "LATITUDE": latitude, "LONGITUDE": longitude, "Na": na, "K": k, "Mg": mg,
            "Ca": ca, "Sr": sr, "Cl": cl, "TDS": tds, "PH": ph, "I": i, "B": b,
            "FORMATION": formation, "BASIN": basin, "RegionCluster": cluster
        }])
        prediction = model.predict(data)[0]
        st.success(f"🟢 **Predicted Lithium Concentration: {prediction:.2f} mg/L**")

        df_map = pd.DataFrame({"lat": [latitude], "lon": [longitude], "Li": [prediction]})
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=latitude, longitude=longitude, zoom=6, pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position='[lon, lat]',
                    get_fill_color=lithium_to_color(prediction),
                    get_radius=5000 + (prediction * 20),
                    pickable=True
                )
            ],
            tooltip={"text": "Li: {Li} mg/L\nLat: {lat}\nLon: {lon}"}
        ))

# ------------------
# Mode: Upload CSV
# ------------------
elif mode == "Upload CSV":
    st.header("📤 Upload CSV for Batch Prediction")

    st.info("ℹ️ Your CSV should have the following columns (case-sensitive):")
    sample_data = pd.DataFrame({
        "LATITUDE": [31.5], "LONGITUDE": [-103.5], "Na": [10000], "K": [500], "Mg": [300],
        "Ca": [1500], "Sr": [100], "Cl": [18000], "TDS": [25000], "PH": [6.5],
        "I": [0.05], "B": [5.0], "FORMATION": ["Wolfcamp"]
    })
    st.dataframe(sample_data)
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="lithium_sample_template.csv",
        mime="text/csv"
    )
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a CSV with well data", type=["csv"])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        required_columns = ['LATITUDE', 'LONGITUDE', 'Na', 'K', 'Mg', 'Ca', 'Sr',
                            'Cl', 'TDS', 'PH', 'I', 'B', 'FORMATION']
        missing = [col for col in required_columns if col not in df_input.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            df_input = df_input.dropna(subset=required_columns)
            df_input["BASIN"] = "Permian"
            df_input["RegionCluster"] = kmeans.predict(df_input[["LATITUDE", "LONGITUDE"]])
            df_input["Predicted_Li_mg_L"] = model.predict(df_input)
            df_input["COLOR"] = df_input["Predicted_Li_mg_L"].apply(lithium_to_color)

            st.success(f"✅ Predicted lithium for {len(df_input)} wells.")
            st.dataframe(df_input[["LATITUDE", "LONGITUDE", "FORMATION", "Predicted_Li_mg_L"]])

            st.download_button(
                label="📥 Download Predictions as CSV",
                data=df_input.to_csv(index=False),
                file_name="lithium_predictions.csv",
                mime="text/csv"
            )

            st.subheader("📍 Predicted Lithium Map (Batch)")
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=df_input["LATITUDE"].mean(),
                    longitude=df_input["LONGITUDE"].mean(),
                    zoom=6,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=df_input,
                        get_position='[LONGITUDE, LATITUDE]',
                        get_fill_color='COLOR',
                        get_radius='5000 + Predicted_Li_mg_L * 20',
                        pickable=True
                    )
                ],
                tooltip={"text": "Li: {Predicted_Li_mg_L} mg/L\nFormation: {FORMATION}"}
            ))
            st.markdown("🟦 **Low Lithium** → 🟥 **High Lithium**")

# ------------------
# Model Info
# ------------------
if st.checkbox("🎓 Show Model Details"):
    st.markdown("""
    - Model: CatBoostRegressor
    - Trained on: 200+ cleaned Permian Basin samples
    - R²: 0.91
    """)
