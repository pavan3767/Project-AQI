import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="AQI Forecaster", layout="wide")

st.title("🏭 Chennai AQI Monitor & 7-Day Forecast")
st.markdown("Live data from Perungudi Station | Powered by VECM Model")

# Load Data
try:
    history_df = pd.read_csv("Cleaned AQI Bulk data (6th Dec).csv")
    forecast_df = pd.read_csv("latest_forecast.csv")
    
    # Convert dates
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    # --- KPI METRICS ---
    latest_aqi = history_df['AQI_calculated'].iloc[-1]
    latest_date = history_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Date", latest_date)
    col2.metric("Current AQI", int(latest_aqi))
    
    # Determine Status
    if latest_aqi <= 50: status = "Good 🟢"
    elif latest_aqi <= 100: status = "Moderate 🟡"
    elif latest_aqi <= 200: status = "Poor 🟠"
    else: status = "Hazardous 🔴"
    col3.metric("Status", status)

    # --- MAIN PLOT ---
    st.subheader("Historical Trend + 7 Day Prediction")
    
    fig = go.Figure()
    
    # Historical Data (Last 30 days for clarity)
    recent_history = history_df.tail(30)
    fig.add_trace(go.Scatter(
        x=recent_history['Date'], 
        y=recent_history['AQI_calculated'],
        mode='lines+markers',
        name='Actual History',
        line=dict(color='blue')
    ))
    
    # Forecast Data
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Predicted_AQI'],
        mode='lines+markers',
        name='VECM Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("Detailed Forecast")
    st.dataframe(forecast_df)

except FileNotFoundError:

    st.error("Data files not found. The automation script might not have run yet.")

