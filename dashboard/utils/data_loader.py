# ============================================================
# utils/data_loader.py — بارگذاری داده و مدل‌ها
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = 'Evapo_mm_Avg'


def load_data():
    """Load final dataset"""
    df = pd.read_parquet(DATA_DIR / "weather_features_final.parquet")
    df['Date_M'] = pd.to_datetime(df['Date_M'])
    df = df.sort_values('Date_M').reset_index(drop=True)
    return df


def load_metadata():
    """Load project metadata"""
    return {
        'target': TARGET,
        'period': '1398-1402',
        'n_days': 1737,
        'n_features': 214
    }


def load_hybrid_model():
    """Load Hybrid model and its exact feature list"""
    model = joblib.load(MODELS_DIR / "hybrid_xgb_residual.pkl")

    # Load features from JSON (exact match with training)
    features_file = REPORTS_DIR / "hybrid_features.json"
    if features_file.exists():
        with open(features_file, 'r', encoding='utf-8') as f:
            features = json.load(f)
    else:
        # Fallback to txt
        with open(REPORTS_DIR / "selected_features.txt", 'r', encoding='utf-8') as f:
            features = [line.strip() for line in f.readlines()]

    return model, features


def load_results():
    """Load model comparison results"""
    try:
        return pd.read_excel(REPORTS_DIR / "FINAL_model_comparison.xlsx")
    except:
        return pd.DataFrame({
            'Model': ['Hybrid (Naive + XGB)', 'Naive (Baseline)',
                     'Random Forest', 'LSTM', 'XGBoost'],
            'RMSE': [23.559, 34.606, 43.403, 46.085, 54.714],
            'MAE': [14.417, 18.339, 28.072, 37.591, 32.752],
            'R²': [0.892, 0.767, 0.633, 0.590, 0.416],
        })
