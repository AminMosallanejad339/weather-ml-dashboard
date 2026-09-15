# ============================================================
# Page 3: Forecast
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data, load_hybrid_model, REPORTS_DIR

st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")

st.markdown("# 📈 Evaporation Forecast")
st.markdown("Predict daily evaporation for the next days using the Hybrid model.")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

try:
    model, features = load_hybrid_model()
    st.success(f"✅ Hybrid model loaded successfully with {len(features)} features")
    MODEL_AVAILABLE = True
except Exception as e:
    st.error(f"⚠️ Model could not be loaded: {e}")
    MODEL_AVAILABLE = False

# ============================================================
# Settings
# ============================================================
st.markdown("### ⚙️ Forecast Settings")

col1, col2 = st.columns(2)
with col1:
    horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

with col2:
    method = st.selectbox("Forecast method",
                          ["Naive (Baseline)", "Hybrid (Naive + XGB)"])

# ============================================================
# Run Forecast
# ============================================================
if st.button("🚀 Run Forecast", type="primary"):
    if not MODEL_AVAILABLE:
        st.error("Model not available")
    else:
        with st.spinner("Running forecast..."):
            last_evapo = df['Evapo_mm_Avg'].iloc[-1]
            naive_forecast = [last_evapo] * horizon

            if method == "Naive (Baseline)":
                forecast = naive_forecast
            else:
                df_work = df.copy()
                preds = []
                for h in range(horizon):
                    last = df_work.iloc[-1:].copy()

                    # Exact feature order (same as training)
                    X_next = last.reindex(columns=features, fill_value=0)

                    residual_pred = model.predict(X_next)[0]
                    naive_pred = last['Evapo_mm_Avg'].values[0]
                    y_next = naive_pred + residual_pred
                    preds.append(y_next)

                    new_row = last.copy()
                    new_row['Evapo_mm_Avg'] = y_next
                    new_row['Date_M'] = last['Date_M'].values[0] + pd.Timedelta(days=1)

                    for col in features:
                        if 'Evapo' in col and '_lag1' in col:
                            new_row[col] = y_next

                    df_work = pd.concat([df_work, new_row], ignore_index=True)

                forecast = preds

            last_date = df['Date_M'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast_Evapo_mm': np.round(forecast, 2)
            })

            # Chart
            fig = go.Figure()

            last_60 = df.tail(60)
            fig.add_trace(go.Scatter(
                x=last_60['Date_M'], y=last_60['Evapo_mm_Avg'],
                mode='lines', name='Actual (last 60 days)',
                line=dict(color='black', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Forecast_Evapo_mm'],
                mode='lines+markers', name=f'Forecast ({method})',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=10)
            ))

            fig.add_vline(x=last_date, line_dash='dot', line_color='gray')

            fig.update_layout(
                title=f'{horizon}-Day Evaporation Forecast — {method}',
                xaxis_title='Date', yaxis_title='Evaporation (mm)',
                height=500, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("### 📋 Forecast Table")
            st.dataframe(forecast_df, use_container_width=True)

            # Download
            csv = forecast_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"forecast_{horizon}days.csv",
                "text/csv"
            )

# ============================================================
# Saved Forecasts
# ============================================================
st.markdown("---")
st.markdown("### 📁 Saved Forecasts")

try:
    saved_forecast = pd.read_excel(REPORTS_DIR / "forecast_7days.xlsx")
    st.dataframe(saved_forecast, use_container_width=True)
except:
    st.info("No saved forecasts found")
