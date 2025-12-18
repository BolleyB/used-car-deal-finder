import pandas as pd
import numpy as np
import joblib
import os
from glob import glob
from sklearn.inspection import permutation_importance

# -------------------------
# Paths
# -------------------------
DATA_PATH = os.path.join("data", "vehicles.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load latest HGB model
# -------------------------
model_files = sorted(
    glob(os.path.join(MODEL_DIR, "hgb_batch_model_*.joblib")),
    key=os.path.getmtime,
    reverse=True
)

if not model_files:
    raise FileNotFoundError("No HGB batch model found in 'models/' directory!")

MODEL_PATH = model_files[0]
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoder = bundle["encoder"]
features = bundle["features"]

print(f"Loaded model: {MODEL_PATH}")
print("Features:", features)

# -------------------------
# Load & prep dataset
# -------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

CURRENT_YEAR = 2025
df["car_age"] = CURRENT_YEAR - df["year"]
df["log_odometer"] = np.log1p(df["odometer"])
df = df[df["price"] > 0]
df["log_price"] = np.log1p(df["price"])

categorical_features = [
    "manufacturer", "condition", "cylinders", "fuel",
    "drive", "type", "state", "model"
]
numeric_features = ["car_age", "log_odometer"]

df[categorical_features] = df[categorical_features].fillna("unknown")
df[numeric_features] = df[numeric_features].fillna(0)

# -------------------------
# Sample subset for memory safety
# -------------------------
sample_size = min(20000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

X = df_sample[features].copy()
y = df_sample["log_price"]

# -------------------------
# Encode all categoricals together in same order as training
# -------------------------
cols_to_encode = [c for c in categorical_features if c in X.columns]
if cols_to_encode:
    X[cols_to_encode] = encoder.transform(X[cols_to_encode])

# -------------------------
# Permutation importance
# -------------------------
print("Computing permutation importance...")

result = permutation_importance(
    model,
    X.to_numpy(),
    y.to_numpy(),
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
}).sort_values(by="importance", ascending=False)

# -------------------------
# Output
# -------------------------
print("\nTop Features by Permutation Importance:")
print(importance_df.head(15))

FI_PATH = os.path.join(MODEL_DIR, "hgb_feature_importance.csv")
importance_df.to_csv(FI_PATH, index=False)
print(f"\nSaved permutation importance to {FI_PATH}")
