# ============================================================
# Page 4: Report
# ============================================================

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Report", page_icon="📄", layout="wide")

st.markdown("# 📄 Project Report")

st.markdown("""
## 🌤️ Intelligent Weather Analytics & Forecasting System

### 📌 Executive Summary

This project focuses on **daily evaporation forecasting** using meteorological
data collected over a 5-year period (1398–1402 SH, 1,737 days).

### 🎯 Objectives

1. **Comprehensive exploratory analysis** of weather parameters
2. **214 engineered features** derived from raw data
3. **Comparison of 5 modeling approaches** for evaporation prediction
4. **Interactive dashboard** for monitoring and forecasting

### 📊 Data

- **Source:** Local weather station
- **Period:** March 2019 – March 2024
- **Records:** 1,737 days
- **Parameters:** Temperature, humidity, wind, rainfall, evaporation, pressure, solar radiation

### 🏆 Key Results

| Model | RMSE | R² |
|-------|------|-----|
| **Hybrid (Naive + XGB)** | **23.56** | **0.892** |
| Naive (Baseline) | 34.61 | 0.767 |
| Random Forest | 43.40 | 0.633 |
| LSTM | 46.09 | 0.590 |
| XGBoost | 54.71 | 0.416 |

**Conclusion:** The Hybrid model achieved a **32% improvement over the baseline**.

### 💡 Key Findings

1. **Strong autocorrelation of evaporation** (r ≈ 0.95) — Naive baseline is powerful
2. **Hybrid Model** combining simplicity with ML power gives the best results
3. **Pure ML models** overfit in time series with high autocorrelation
4. **Humidity** is the strongest inhibitor of evaporation (r = -0.68)

### 🔧 Technologies

- **Python 3.11** — Core language
- **Pandas, NumPy** — Data processing
- **Scikit-learn, XGBoost** — Machine learning
- **TensorFlow/Keras** — Deep learning
- **Streamlit, Plotly** — Interactive dashboard

### 📚 References

1. McKee, T. B., et al. (1993). The relationship of drought frequency and duration to time scales.
2. Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.

---

### 👤 Author

**Amin Mosallanejad**

Water/Wastewater Treatment | Senior Utility Engineer | Data Scientist (M.Sc.)
""")

# ============================================================
# Download Section
# ============================================================
st.markdown("---")
st.markdown("### 📥 Downloads")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "📄 Project Report (Markdown)",
        """# Weather Analytics Project Report

## Executive Summary
This project focuses on daily evaporation forecasting using meteorological
data collected over a 5-year period.

## Results
- Best Model: Hybrid (Naive + XGBoost)
- RMSE: 23.56
- R²: 0.892

## Author
Amin Mosallanejad
""",
        "weather_report.md",
        "text/markdown"
    )

with col2:
    st.download_button(
        "📊 Model Results (CSV)",
        """Model,RMSE,MAE,R2
Hybrid,23.56,14.42,0.892
Naive,34.61,18.34,0.767
Random Forest,43.40,28.07,0.633
LSTM,46.09,37.59,0.590
XGBoost,54.71,32.75,0.416
""",
        "model_results.csv",
        "text/csv"
    )

with col3:
    st.download_button(
        "📋 Feature List",
        "Feature list available in reports directory",
        "features.txt",
        "text/plain"
    )
