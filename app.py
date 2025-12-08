import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="AQI Forecaster", layout="wide")

st.title("Chennai AQI Monitor & 7-Day Forecast")
st.markdown("Live data from Perungudi Station | Powered by VECM Model")

# Load Data
try:
    history_df = pd.read_csv("Cleaned AQI Bulk data (6th Dec).csv")
    forecast_df = pd.read_csv("latest_forecast.csv")
    
    # Convert dates to datetime objects initially for calculation/plotting
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    # --- KPI METRICS ---
    latest_aqi = history_df['AQI_calculated'].iloc[-1]
    latest_date = history_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Date", latest_date)
    col2.metric("Current AQI", int(latest_aqi))
    
    # Determine Status for KPI
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

    # --- DATA TABLE (Formatted) ---
    st.subheader("Detailed Forecast")

    # 1. Clean the 'Date' column (Remove 00:00:00 time component)
    forecast_df['Date'] = forecast_df['Date'].dt.date

    # 2. Round AQI to whole numbers
    forecast_df['Predicted_AQI'] = forecast_df['Predicted_AQI'].round(0).astype(int)

    # 3. Add the 'Status' Column with Emojis
    def get_aqi_status(aqi):
        if aqi <= 50:
            return "Good 🟢"
        elif aqi <= 100:
            return "Moderate 🟡"
        elif aqi <= 200:
            return "Poor 🟠"
        else:
            return "Hazardous 🔴"

    forecast_df['Status'] = forecast_df['Predicted_AQI'].apply(get_aqi_status)
    # 1. Create a Styler object to center text
    styled_df = forecast_df.style.set_properties(**{'text-align': 'center'})
    
    # 2. Also center the headers (th) specifically
    styled_df = styled_df.set_table_styles(
    [dict(selector='th', props=[('text-align', 'center')])]
    )
    
    # Display the final formatted table
    st.dataframe(
    styled_df,
    hide_index=True,  # Removes the 0,1,2,3 column
    use_container_width=True # Optional: stretches table to fill width
    )

except FileNotFoundError:
    st.error("Data files not found. The automation script might not have run yet.")



