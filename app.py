# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -------------------------
# Load trained model
# -------------------------
model_path = os.path.join("models", "random_forest_used_car_model.joblib")
rf_model = joblib.load(model_path)

st.title("Used Car Price Predictor")
st.write("Predict the price of a used car using the trained Random Forest model.")

# -------------------------
# Load cleaned dataset for dropdown options
# -------------------------
data_path = os.path.join("data", "cleaned_vehicles_sample.csv")
df = pd.read_csv(data_path)

# Normalize columns and categorical values
df.columns = df.columns.str.strip().str.lower()
for col in ["manufacturer", "model", "condition", "cylinders", "fuel",
            "transmission", "drive", "type", "state"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower()

# -------------------------
# Dropdown options
# -------------------------
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
# Manufacturer & Model selection (dynamic)
# -------------------------
manufacturer = st.selectbox("Manufacturer", [m.title() for m in manufacturers])
manufacturer_lower = manufacturer.lower()

# Models update based on selected manufacturer
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
# Make Prediction
# -------------------------
if submitted:
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
        "log_odometer": np.log1p(odometer)
    }])

    try:
        predicted_price = rf_model.predict(new_car)[0]
        st.success(f"Predicted Price: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error predicting price: {e}")
        st.write("Make sure all required features are included and correctly typed.")
