# ============================================================
# Page 1: Exploratory Data Analysis (EDA)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.markdown("# 📊 Exploratory Data Analysis")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Distributions", "🔥 Correlation", "🌡️ Seasonal Pattern", "🚨 Extreme Events"
])

# ============================================================
# Tab 1: Distributions
# ============================================================
with tab1:
    st.markdown("### Distribution of Weather Parameters")

    param = st.selectbox(
        'Select parameter:',
        ['Evapo_mm_Avg', 'Temperature_Avg', 'Humidity_Avg',
         'Wind_Kmh_Avg', 'Watts_m2_Avg', 'Rain_mm_Tot', 'VPD'],
        key='dist_param'
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x=param, nbins=60,
                          title=f'Distribution of {param}',
                          color_discrete_sequence=['steelblue'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y=param, title=f'Boxplot of {param}',
                    color_discrete_sequence=['#2ca02c'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Descriptive Statistics")
    stats = df[param].describe().round(3)
    st.dataframe(stats.to_frame().T, use_container_width=True)

# ============================================================
# Tab 2: Correlation
# ============================================================
with tab2:
    st.markdown("### Correlation Matrix")

    corr_vars = ['Temperature_Avg', 'Humidity_Avg', 'Evapo_mm_Avg',
                 'Wind_Kmh_Avg', 'Watts_m2_Avg', 'Baro_KPa_Avg',
                 'Rain_mm_Tot', 'VPD', 'PET_Hargreaves']

    corr_vars = [c for c in corr_vars if c in df.columns]
    corr_matrix = df[corr_vars].corr()

    fig = px.imshow(corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect='auto',
                    title='Correlation Matrix of Weather Parameters')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation with Evaporation")
    corr_target = df.corr(numeric_only=True)['Evapo_mm_Avg'].drop('Evapo_mm_Avg').sort_values()

    col1, col2 = st.columns(2)
    with col1:
        top_pos = corr_target.tail(10)
        fig = px.bar(x=top_pos.values, y=top_pos.index, orientation='h',
                    title='Top 10 Positive Correlations',
                    color=top_pos.values, color_continuous_scale='Greens')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_neg = corr_target.head(10)
        fig = px.bar(x=top_neg.values, y=top_neg.index, orientation='h',
                    title='Top 10 Negative Correlations',
                    color=top_neg.values, color_continuous_scale='Reds_r')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 3: Seasonal Pattern
# ============================================================
with tab3:
    st.markdown("### Monthly Pattern")

    month_names = ['Apr','May','Jun','Jul','Aug','Sep',
                   'Oct','Nov','Dec','Jan','Feb','Mar']

    monthly = df.groupby('Month_S').agg({
        'Evapo_mm_Avg': 'mean',
        'Temperature_Avg': 'mean',
        'Humidity_Avg': 'mean',
        'Rain_mm_Tot': 'sum',
    }).round(2)
    monthly.index = month_names

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=monthly.index, y=monthly['Rain_mm_Tot']/5,
               name='Rainfall (avg)', marker_color='steelblue'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=monthly.index, y=monthly['Temperature_Avg'],
                   name='Temperature', mode='lines+markers',
                   line=dict(color='red', width=3)),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=monthly.index, y=monthly['Evapo_mm_Avg'],
                   name='Evaporation', mode='lines+markers',
                   line=dict(color='orange', width=3)),
        secondary_y=False
    )

    fig.update_layout(title='Seasonal Pattern of Parameters', height=500)
    fig.update_yaxes(title_text="Rainfall / Evaporation (mm)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(monthly, use_container_width=True)

# ============================================================
# Tab 4: Extreme Events
# ============================================================
with tab4:
    st.markdown("### Detected Extreme Events")

    events = {
        'Hot Day (P98)': df['Hot_Day'].sum() if 'Hot_Day' in df.columns else 0,
        'Heat Wave': df['Heat_Wave'].sum() if 'Heat_Wave' in df.columns else 0,
        'Heavy Rain (P98)': df['Heavy_Rain'].sum() if 'Heavy_Rain' in df.columns else 0,
        'Windy Day (P98)': df['Windy_Day'].sum() if 'Windy_Day' in df.columns else 0,
        'High Evapo (P95)': df['High_Evapo'].sum() if 'High_Evapo' in df.columns else 0,
        'Dry Spell (>30d)': df['Dry_Spell'].sum() if 'Dry_Spell' in df.columns else 0,
    }

    events_df = pd.DataFrame({
        'Event': list(events.keys()),
        'Days': list(events.values()),
        'Percent (%)': [round(v/len(df)*100, 1) for v in events.values()]
    })

    fig = px.bar(events_df, x='Event', y='Days',
                color='Days', color_continuous_scale='Reds',
                text='Days', title='Frequency of Extreme Events')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(events_df, use_container_width=True)
