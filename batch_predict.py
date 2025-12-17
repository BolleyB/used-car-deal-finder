import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Paths
# -------------------------
DATA_PATH = os.path.join("data", "vehicles.csv")
MODEL_PATH = os.path.join("models", "hgb_batch_model_log.joblib")
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
# Target & log transform
# -------------------------
df = df[df["price"] > 0]
df["log_price"] = np.log1p(df["price"])

# -------------------------
# Outlier removal (IQR) on log_price, car_age, log_odometer
# -------------------------
def remove_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.between(lower, upper)

mask = (
    remove_outliers_iqr(df["log_price"]) &
    remove_outliers_iqr(df["car_age"]) &
    remove_outliers_iqr(df["log_odometer"])
)
df = df[mask]

# -------------------------
# Select features
# -------------------------
categorical_features = ["manufacturer", "condition", "cylinders",
                        "fuel", "transmission", "drive", "type", "state"]

# Handle high-cardinality 'model'
top_models = df["model"].value_counts().nlargest(300).index
df["model"] = df["model"].where(df["model"].isin(top_models), other="other")
categorical_features.append("model")

numeric_features = ["car_age", "log_odometer"]
features = categorical_features + numeric_features
target = "log_price"

# Fill missing values
df[categorical_features] = df[categorical_features].fillna("unknown")
df[numeric_features] = df[numeric_features].fillna(0)

# -------------------------
# Train/test split
# -------------------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Encode categorical features
# -------------------------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[categorical_features] = encoder.fit_transform(X_train[categorical_features])
X_test[categorical_features] = encoder.transform(X_test[categorical_features])

# -------------------------
# Initialize incremental HistGradientBoosting model
# -------------------------
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=10,
    min_samples_leaf=20,
    warm_start=True
)

# -------------------------
# Batch training
# -------------------------
batch_size = 50000
n_batches = int(np.ceil(X_train.shape[0] / batch_size))
print(f"Training in {n_batches} batches...")

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, X_train.shape[0])
    X_batch = X_train.iloc[start:end]
    y_batch = y_train.iloc[start:end]
    
    print(f"Training batch {i+1}/{n_batches}...")
    model.max_iter += 10 if i > 0 else 0  # incrementally grow iterations
    model.fit(X_batch, y_batch)

# -------------------------
# Evaluate
# -------------------------
y_pred = np.expm1(model.predict(X_test))  # invert log
y_true = np.expm1(y_test)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MAE: ${mae:,.2f}")
print(f"R²: {r2:.4f}")

# -------------------------
# Save model & encoder
# -------------------------
joblib.dump({"model": model, "encoder": encoder, "features": features}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
