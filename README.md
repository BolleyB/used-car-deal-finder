
# 🚗 Used Car Price Predictor

A production-ready machine learning application that predicts used car prices from real-world listing data.

## 📌 Overview
This project uses **batch-trained machine learning** on 426,000+ vehicle listings to predict fair market prices.  
The system includes feature engineering, model versioning, explainability, and a clean Streamlit UI.

## 🧠 Model
- Algorithm: HistGradientBoostingRegressor
- R²: ~0.78
- MAE: ~$3,800
- Training: Memory-safe batch training

## 🏗️ System Architecture

```
Raw CSV Data (426k rows)
        │
        ▼
Feature Engineering
(log transforms, frequency encoding)
        │
        ▼
Batch Training Pipeline
(HistGradientBoosting)
        │
        ▼
Model Versioning
(hgb_batch_model_YYYYMMDD.joblib)
        │
        ▼
Permutation Feature Importance
(precomputed offline)
        │
        ▼
Streamlit Web App
(Predictions + Metrics + Explainability)
```

## 📊 Key Features
- Automatic loading of latest trained model
- Precomputed feature importance (no live compute overhead)
- Business-friendly metrics:
  - Price confidence range
  - Depreciation per year
  - Price per mile
  - Percentile ranking

## 🖥️ Tech Stack
- Python, pandas, NumPy
- scikit-learn
- Streamlit
- Joblib
- AWS-ready

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Future Improvements
- Hyperparameter tuning for 80%+ R²
- Model calibration
- Deployment via Docker + AWS

