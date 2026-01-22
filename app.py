import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="AQI Forecaster", layout="wide")

st.title("Chennai AQI Monitor & 7-Day Forecast")
st.markdown("Live data from Perungudi Station | Powered by VECM Model")

# Load Data
try:
    history_df = pd.read_csv("Cleaned AQI Bulk data (22nd Jan).csv")
    forecast_df = pd.read_csv("latest_forecast.csv")
    
    # Convert dates to datetime objects initially for calculation/plotting
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    # --- KPI METRICS ---
    latest_aqi = history_df['AQI'].iloc[-1]
    latest_date = history_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    col1, col2, col3,col4 = st.columns(4)
    col1.metric("Latest Date", latest_date)
    col2.metric("Current AQI", int(latest_aqi))
    
    # 3. Add the 'Status' Column with Emojis
    def get_aqi_status(aqi):
        if aqi <= 50:
            return "Good 🟢"
        elif aqi <= 100:
            return "Moderate 🟡"
        elif aqi <= 150:
            return "Poor 🟠"
        elif aqi <= 200:
            return "Unhealthy 🔴"
        elif aqi <= 300:
            return "Very Unhealthy 🔴"
        else:
            return "Hazardous 🟤"
    col3.metric("Status", get_aqi_status(latest_aqi))

    def display_aqi_recommendation(aqi_value):
        st.subheader("Health Recommendations")
    
        # 1. Good (0-50) -> Green Box
        if aqi_value <= 50:
            st.success(
                "🌳 **Action:** The air is fresh! Enjoy the outdoors, open your windows, and take a walk in the park."
            )
    
        # 2. Moderate (51-100) -> Blue Box
        elif 50 < aqi_value <= 100:
            st.info(
                "⚠️ **Action:** Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
            )
    
        # 3. Poor (101-200) -> Yellow/Amber Box (Fixed the Red Error issue)
        elif 100 < aqi_value <= 200:
            st.warning(
                "😷 **Action:** Everyone may begin to experience health effects. Limit outdoor activities and wear a mask if you need to go out."
            )
    
        # 4. Hazardous (201+) -> Red Box (Only for extreme emergencies)
        else:
            st.error(
                "🏠 **Action:** Avoid ALL outdoor activities. Keep windows closed, use an Air Purifier, and stay hydrated."
            )
    # Call the function where you want the box to appear
    display_aqi_recommendation(latest_aqi)

    # --- MAIN PLOT ---
    st.subheader("Historical Trend + 7 Day Prediction")
        
    fig = go.Figure()
        
    # Historical Data (Last 30 days for clarity)
    recent_history = history_df.tail(30)
    fig.add_trace(go.Scatter(
        x=recent_history['Date'], 
        y=recent_history['AQI'],
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


    forecast_df['Status'] = forecast_df['Predicted_AQI'].apply(get_aqi_status)
    
    # 1. Create a Styler object to center text and show the table
    st.dataframe(forecast_df,hide_index=True)

except FileNotFoundError:
    st.error("Data files not found. The automation script might not have run yet.")























