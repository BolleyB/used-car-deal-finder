# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import RegressorMixin

# -------------------------
# Config
# -------------------------
MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "cleaned_vehicles_sample.csv")

# -------------------------
# Safe model loader
# -------------------------
@st.cache_resource
def load_models_safe():
    models = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".joblib"):
            path = os.path.join(MODEL_DIR, file)
            try:
                obj = joblib.load(path)
                if isinstance(obj, RegressorMixin):
                    models.append(obj)
                elif isinstance(obj, list):
                    valid_models = [m for m in obj if isinstance(m, RegressorMixin)]
                    models.extend(valid_models)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
    return models

models = load_models_safe()
st.write(f"Loaded {len(models)} valid models.")

# -------------------------
# Load dataset for dropdowns
# -------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Normalize categorical columns
cat_cols = ["manufacturer", "model", "condition", "cylinders", "fuel",
            "transmission", "drive", "type", "state"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower()

# Dropdown options
manufacturers = sorted(df["manufacturer"].unique())
conditions = sorted(df["condition"].unique())
cylinders_options = sorted(df["cylinders"].unique())
fuel_options = sorted(df["fuel"].unique())
transmissions = sorted(df["transmission"].unique())
drive_options = sorted(df["drive"].unique())
types = sorted(df["type"].unique())
states = sorted(df["state"].unique())
min_age, max_age = int(df["car_age"].min()), int(df["car_age"].max())
car_age_options = list(range(min_age, max_age + 1))

# -------------------------
# Dynamic manufacturer -> model
# -------------------------
manufacturer = st.selectbox("Manufacturer", [m.title() for m in manufacturers])
manufacturer_lower = manufacturer.lower()
models_for_manufacturer = sorted(df[df["manufacturer"] == manufacturer_lower]["model"].unique())
model_name = st.selectbox("Model", [m.title() for m in models_for_manufacturer])
model_lower = model_name.lower()

# -------------------------
# Other car features
# -------------------------
with st.form("car_form"):
    st.subheader("Enter other car details:")

    condition = st.selectbox("Condition", [c.title() for c in conditions]).lower()
    cylinders = st.selectbox("Cylinders", [c.title() for c in cylinders_options]).lower()
    fuel = st.selectbox("Fuel Type", [f.title() for f in fuel_options]).lower()
    odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=35000)
    transmission = st.selectbox("Transmission", [t.title() for t in transmissions]).lower()
    drive = st.selectbox("Drive Type", [d.upper() for d in drive_options]).lower()
    car_type = st.selectbox("Car Type", [t.title() for t in types]).lower()
    car_age = st.selectbox("Car Age (years)", car_age_options)
    state = st.selectbox("State", [s.upper() for s in states]).lower()

    submitted = st.form_submit_button("Predict Price")

# -------------------------
# Prediction function
# -------------------------
def prepare_features(input_df, full_df):
    """
    Prepares features for prediction: log_odometer, car_age, model frequency.
    """
    df_copy = input_df.copy()

    # Log-transform odometer
    df_copy["log_odometer"] = np.log1p(df_copy["log_odometer"] if "log_odometer" in df_copy.columns else df_copy["odometer"])

    # Car age (already provided, but ensure column exists)
    if "car_age" not in df_copy.columns:
        df_copy["car_age"] = df_copy["car_age"]

    # High-cardinality feature: model frequency
    model_counts = full_df["model"].value_counts()
    df_copy["model_freq"] = df_copy["model"].map(model_counts).fillna(0)

    return df_copy

def predict_batch(models, df):
    """
    Predict with multiple models and average results.
    Assumes models were trained on log-target.
    """
    preds = []
    for m in models:
        try:
            pred = m.predict(df)
            preds.append(pred)
        except Exception as e:
            st.warning(f"Prediction failed for one model: {e}")
    if len(preds) == 0:
        return None
    avg_pred_log = np.mean(preds, axis=0)
    return np.expm1(avg_pred_log)  # reverse log transform

# -------------------------
# Make Prediction
# -------------------------
if submitted:
    if len(models) == 0:
        st.error("No valid models loaded!")
    else:
        new_car = pd.DataFrame([{
            "manufacturer": manufacturer_lower,
            "model": model_lower,
            "condition": condition,
            "cylinders": cylinders,
            "fuel": fuel,
            "transmission": transmission,
            "drive": drive,
            "type": car_type,
            "state": state,
            "car_age": car_age,
            "odometer": odometer
        }])

        new_car_prepared = prepare_features(new_car, df)

        preds = predict_batch(models, new_car_prepared)
        if preds is not None:
            st.success(f"Predicted Price: ${preds[0]:,.2f}")
        else:
            st.error("No predictions could be made. Check model features and input.")
