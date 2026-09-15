# ============================================================
# Page 2: Models
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_results

st.set_page_config(page_title="Models", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Model Comparison & Evaluation")

@st.cache_data
def get_results():
    return load_results()

results = get_results()

# ============================================================
# Results Table
# ============================================================
st.markdown("### 📊 Model Comparison (Test Set)")

def highlight_best(row):
    if row['RMSE'] == results['RMSE'].min():
        return ['background-color: #d4edda; font-weight: bold'] * len(row)
    return [''] * len(row)

st.dataframe(
    results.style.apply(highlight_best, axis=1),
    use_container_width=True
)

# ============================================================
# Charts
# ============================================================
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(results.sort_values('RMSE'),
                x='Model', y='RMSE',
                color='RMSE', color_continuous_scale='RdYlGn_r',
                text='RMSE', title='RMSE Comparison (lower is better)')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(results.sort_values('R²', ascending=False),
                x='Model', y='R²',
                color='R²', color_continuous_scale='RdYlGn',
                text='R²', title='R² Comparison (higher is better)')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Radar Chart
# ============================================================
st.markdown("### 🎯 Multi-Criteria Comparison")

radar_data = results.copy()
radar_data['RMSE_norm'] = 1 - (radar_data['RMSE'] - radar_data['RMSE'].min()) / (radar_data['RMSE'].max() - radar_data['RMSE'].min())
radar_data['R²_norm'] = (radar_data['R²'] - radar_data['R²'].min()) / (radar_data['R²'].max() - radar_data['R²'].min())
radar_data['MAE_norm'] = 1 - (radar_data['MAE'] - radar_data['MAE'].min()) / (radar_data['MAE'].max() - radar_data['MAE'].min())

fig = go.Figure()
for _, row in radar_data.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['RMSE_norm'], row['MAE_norm'], row['R²_norm'], row['RMSE_norm']],
        theta=['RMSE', 'MAE', 'R²', 'RMSE'],
        fill='toself',
        name=row['Model']
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, height=500,
    title='Normalized Model Comparison'
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Key Findings
# ============================================================
st.markdown("### 💡 Key Findings")

best_model = results.sort_values('RMSE').iloc[0]
naive_rmse = results[results['Model'].str.contains('Naive', case=False)]['RMSE'].values
naive_rmse = naive_rmse[0] if len(naive_rmse) > 0 else 34.6

improvement = (naive_rmse - best_model['RMSE']) / naive_rmse * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🏆 Best Model", best_model['Model'])
with col2:
    st.metric("📉 Best RMSE", f"{best_model['RMSE']:.2f}")
with col3:
    st.metric("📈 Improvement vs Naive", f"{improvement:.1f}%")

st.info(f"""
**Conclusion:** The **{best_model['Model']}** model achieved the best performance
on the Test Set with RMSE = {best_model['RMSE']:.2f} and R² = {best_model['R²']:.3f},
representing a **{improvement:.1f}% improvement over the Naive baseline**.
""")
