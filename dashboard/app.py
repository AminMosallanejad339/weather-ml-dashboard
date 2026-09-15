# ============================================================
# Weather ML Dashboard — Home Page
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import load_data, load_metadata

st.set_page_config(
    page_title="Weather ML Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        padding: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    .author-card {
        background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .author-name {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .author-title {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌤️ Weather Analytics & Evaporation Forecasting</div>',
            unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_metadata():
    return load_metadata()

try:
    df = get_data()
    metadata = get_metadata()
except Exception as e:
    st.error(f"❌ Data loading error: {e}")
    st.stop()

# ============================================================
# Project Overview
# ============================================================
st.markdown('<div class="section-header">📋 Project Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Days of Data</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Year_S'].nunique()}</div>
        <div class="metric-label">Years Covered</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Evapo_mm_Avg'].mean():.1f}</div>
        <div class="metric-label">Avg Evaporation (mm)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Temperature_Avg'].mean():.1f}°C</div>
        <div class="metric-label">Avg Temperature</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# About the Project
# ============================================================
st.markdown('<div class="section-header">🎯 About the Project</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📌 Objective

    Design an **end-to-end intelligent weather analytics system** focused on
    **evaporation** as a key parameter in water resource management.

    ### 🔬 Methodology

    - **Data Source:** Local weather station, 1398–1402 (5 years, 1,737 days)
    - **214 Engineered Features:** calendrical, physical, temporal, statistical, interaction
    - **5 Models Compared:** Naive, Random Forest, XGBoost, Hybrid, LSTM
    - **Best Model:** Hybrid (Naive + XGBoost) with R² = 0.89

    ### 📊 Key Finding

    The **Hybrid model**, combining Naive simplicity with XGBoost power,
    achieved **32% improvement over the baseline**.
    """)

with col2:
    st.markdown("### 📊 Model Performance")
    results = pd.DataFrame({
        'Model': ['Hybrid', 'Naive', 'RF', 'LSTM', 'XGB'],
        'R²': [0.892, 0.767, 0.633, 0.590, 0.416]
    })

    fig = px.bar(results, x='Model', y='R²',
                 color='R²', color_continuous_scale='RdYlGn')
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Data Overview
# ============================================================
st.markdown('<div class="section-header">📈 Data Overview</div>', unsafe_allow_html=True)

param_options = {
    'Evaporation (mm)': 'Evapo_mm_Avg',
    'Temperature (°C)': 'Temperature_Avg',
    'Humidity (%)': 'Humidity_Avg',
    'Rainfall (mm)': 'Rain_mm_Tot',
    'Wind Speed (km/h)': 'Wind_Kmh_Avg',
    'Solar Radiation (W/m²)': 'Watts_m2_Avg',
}

selected_param = st.selectbox('Select parameter:', list(param_options.keys()))
param_col = param_options[selected_param]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Date_M'], y=df[param_col],
    mode='lines', name=selected_param,
    line=dict(color='#1f77b4', width=1)
))

df['rolling_30'] = df[param_col].rolling(30).mean()
fig.add_trace(go.Scatter(
    x=df['Date_M'], y=df['rolling_30'],
    mode='lines', name='30-Day Moving Average',
    line=dict(color='red', width=2, dash='dash')
))

fig.update_layout(
    title=f'Time Series — {selected_param}',
    xaxis_title='Date',
    yaxis_title=selected_param,
    height=450,
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# AUTHOR SECTION
# ============================================================
st.markdown('<div class="section-header">👤 Author</div>', unsafe_allow_html=True)

st.markdown("""
<div class="author-card">
    <div class="author-name">Amin Mosallanejad</div>
    <div class="author-title">
        Water/Wastewater Treatment &nbsp;|&nbsp; Senior Utility Engineer &nbsp;|&nbsp; Data Scientist (M.Sc.)
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌤️ <strong>Weather Analytics Dashboard</strong> | Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Study Period: 1398–1402 (2019–2023) | 1,737 days of data</p>
</div>
""", unsafe_allow_html=True)
