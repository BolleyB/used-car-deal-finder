import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Paths
# -------------------------
DATA_PATH = os.path.join("data", "vehicles.csv")
MODEL_PATH = os.path.join("models", "random_forest_used_car_model.joblib")

os.makedirs("models", exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

print("Dataset shape (raw):", df.shape)

# -------------------------
# Feature engineering
# -------------------------
print("Engineering features...")
CURRENT_YEAR = 2025

df["car_age"] = CURRENT_YEAR - df["year"]
df["log_odometer"] = np.log1p(df["odometer"])

# -------------------------
# Select features
# -------------------------
target = "price"

categorical_features = [
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "transmission",
    "drive",
    "type",
    "state",
]

numeric_features = [
    "car_age",
    "log_odometer",
]

features = categorical_features + numeric_features

# -------------------------
# Clean data
# -------------------------
print("Cleaning data...")

df = df[features + [target]]

# Drop rows with missing target
df = df.dropna(subset=[target])

# Fill missing values
df[categorical_features] = df[categorical_features].fillna("unknown")
df[numeric_features] = df[numeric_features].fillna(0)

# Remove invalid or extreme values
df = df[(df["price"] > 0) & (df["price"] < 200000)]  # remove extreme outliers
df = df[df["car_age"].between(0, 50)]

print("Dataset shape (cleaned):", df.shape)

# -------------------------
# Log-transform target
# -------------------------
df["log_price"] = np.log1p(df["price"])
target_transformed = "log_price"

# -------------------------
# Train / test split
# -------------------------
print("Splitting data...")

X = df[features]
y = df[target_transformed]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Preprocessing + model pipeline
# -------------------------
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model),
    ]
)

# -------------------------
# Train model
# -------------------------
print("Training Random Forest model...")
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"MAE: ${mae:,.2f}")
print(f"R²: {r2:.4f}")

# -------------------------
# Save model
# -------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
