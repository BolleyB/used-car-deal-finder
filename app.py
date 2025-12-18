# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

# -------------------------
# Config
# -------------------------
MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "vehicles.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(MODEL_DIR, "hgb_feature_importance.csv")
CURRENT_YEAR = 2025
TOP_MODELS = 200

st.set_page_config(
    page_title="Used Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load latest HGB model
# -------------------------
@st.cache_resource
def load_latest_model():
    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("hgb_batch_model_") and f.endswith(".joblib")
    ]
    if not model_files:
        st.warning("No HGB batch model found!")
        return None, None, None
    latest_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
    model_path = os.path.join(MODEL_DIR, latest_file)
    obj = joblib.load(model_path)
    return obj["model"], obj["encoder"], obj["features"]

with st.spinner("Loading model..."):
    model, encoder, features = load_latest_model()

if model is None:
    st.error("No model available. Train a model first!")
    st.stop()

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["log_odometer"] = np.log1p(df["odometer"])
    df = df[df["price"] > 0]
    df["log_price"] = np.log1p(df["price"])
    return df

df = load_dataset()

# -------------------------
# Prepare categorical columns
# -------------------------
cat_cols = ["manufacturer","model","condition","cylinders","fuel","drive","type","state"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.lower()

# Dropdown options
manufacturers = sorted(df["manufacturer"].dropna().unique())
conditions = sorted(df["condition"].dropna().unique())
cylinders_options = sorted(df["cylinders"].dropna().unique())
fuel_options = sorted(df["fuel"].dropna().unique())
drive_options = sorted(df["drive"].dropna().unique())
types = sorted(df["type"].dropna().unique())
states = sorted(df["state"].dropna().unique())
car_age_options = list(range(int(df["car_age"].min()), int(df["car_age"].max()) + 1))

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚗 Used Car Price Predictor")
st.write("Predict the price of a used car using the latest batch-trained HGB model.")
st.write("This app includes key metrics and comparisons to help you make informed decisions.")

with st.expander("🔧 Car Details Input", expanded=True):
    manufacturer = st.selectbox("Manufacturer", [m.title() for m in manufacturers])
    manufacturer_lower = manufacturer.lower()

    models_for_manufacturer = (
        df[df["manufacturer"] == manufacturer_lower]["model"]
        .value_counts()
        .head(TOP_MODELS)
        .index
        .tolist()
    )
    model_name = st.selectbox("Model", [m.title() for m in models_for_manufacturer])
    model_lower = model_name.lower()

    condition = st.selectbox("Condition", [c.title() for c in conditions]).lower()
    cylinders = st.selectbox("Cylinders", [c.title() for c in cylinders_options]).lower()
    fuel = st.selectbox("Fuel Type", [f.title() for f in fuel_options]).lower()
    odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=35000)
    drive = st.selectbox("Drive Type", [d.upper() for d in drive_options]).lower()
    car_type = st.selectbox("Car Type", [t.title() for t in types]).lower()
    car_age = st.selectbox("Car Age (years)", car_age_options)
    state = st.selectbox("State", [s.upper() for s in states]).lower()
    submitted = st.button("Predict Price")

# -------------------------
# Prepare input for prediction
# -------------------------
def prepare_input(df_input):
    df_copy = df_input.copy()
    if "log_odometer" not in df_copy.columns:
        df_copy["log_odometer"] = np.log1p(df_copy["odometer"])
    return df_copy

# -------------------------
# Prediction & Metrics
# -------------------------
if submitted:
    new_car = pd.DataFrame([{
        "manufacturer": manufacturer_lower,
        "model": model_lower,
        "condition": condition,
        "cylinders": cylinders,
        "fuel": fuel,
        "drive": drive,
        "type": car_type,
        "state": state,
        "car_age": car_age,
        "odometer": odometer
    }])
    new_car_prepared = prepare_input(new_car)

    # Ensure columns match training features
    cat_in_features = [f for f in features if f in cat_cols]
    X_cat = encoder.transform(new_car_prepared[cat_in_features])
    num_in_features = [f for f in features if f not in cat_cols]
    X_num = new_car_prepared[num_in_features].to_numpy()
    X_final = np.hstack([X_num, X_cat])

    with st.spinner("Predicting price..."):
        pred_log = model.predict(X_final)
        pred_price = np.expm1(pred_log)[0]

    st.success(f"💰 Predicted Price: ${pred_price:,.2f}")

    # Additional Metrics
    median_price = df["price"].median()
    price_percentile = (df["price"] < pred_price).mean() * 100
    price_per_mile = pred_price / max(1, odometer)
    depreciation_per_year = pred_price / max(1, car_age)

    st.markdown("### 📊 Additional Metrics")
    st.write(f"- Dataset Median Price: ${median_price:,.2f}")
    st.write(f"- Price Percentile: {price_percentile:.1f}th percentile")
    st.write(f"- Price per Mile: ${price_per_mile:,.2f}")
    st.write(f"- Approx. Depreciation per Year: ${depreciation_per_year:,.2f}")

# -------------------------
# Precomputed Feature Importance
# -------------------------
with st.expander("📊 Feature Importance (Precomputed)", expanded=False):
    if os.path.exists(FEATURE_IMPORTANCE_CSV):
        df_imp = pd.read_csv(FEATURE_IMPORTANCE_CSV).sort_values("importance", ascending=True)
        st.bar_chart(df_imp.set_index("feature")["importance"], horizontal=True)
    else:
        st.warning("Feature importance CSV not found. Run batch_feature_importance.py first.")

st.caption("🔹 Built with Python, Streamlit, and scikit-learn | Portfolio-ready")
