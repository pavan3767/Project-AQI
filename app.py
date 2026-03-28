import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AQI Forecaster", layout="wide")

# --- GEN AI SETUP ---
# Make sure to set GEMINI_API_KEY in your Streamlit secrets or environment variables
# .streamlit/secrets.toml -> GEMINI_API_KEY = "your_api_key_here"
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash') # Fast model for quick UI loads
else:
    st.warning("⚠️ Gemini API Key not found. Please set GEMINI_API_KEY in secrets. AI features will be disabled.")

@st.cache_data(ttl=3600) # Cache the AI response for 1 hour so you don't burn API quota on page reloads
def get_ai_insights(current_aqi, dom_pollutant, tomorrow_aqi):
    if not API_KEY:
        return "AI recommendations unavailable (No API Key).", "Planner unavailable."
    
    prompt = f"""
    You are an expert environmental health advisor. 
    The current AQI in Chennai is {current_aqi} and the dominant pollutant is {dom_pollutant}.
    Tomorrow's forecasted AQI is {tomorrow_aqi}.

    Please provide:
    1. A short, empathetic health recommendation for today (2-3 sentences).
    2. A practical 'Tomorrow Planner' for citizens based on tomorrow's forecast (2-3 sentences).

    Format the response strictly as:
    RECOMMENDATION: [Your recommendation here]
    PLANNER: [Your planner here]
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse the response
        rec_part = text.split("PLANNER:")[0].replace("RECOMMENDATION:", "").strip()
        plan_part = text.split("PLANNER:")[1].strip() if "PLANNER:" in text else "Have a safe day tomorrow."
        return rec_part, plan_part
    except Exception as e:
        return f"Could not generate recommendation: {e}", "Could not generate planner."

# --- DATA LOADING ---
@st.cache_data(ttl=300) # Cache data for 5 mins
def load_data():
    history_df = pd.read_csv("Cleaned AQI Bulk data (26th Jan).csv")
    forecast_df = pd.read_csv("latest_forecast.csv")
    
    # Load model log to see which model was selected by the drift detection
    try:
        model_logs = pd.read_csv("model_log.csv")
        active_model = model_logs.iloc[-1]['Selected_Model']
    except FileNotFoundError:
        active_model = "VECM" # Fallback if log doesn't exist yet
        
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    
    return history_df, forecast_df, active_model

try:
    history_df, forecast_df, active_model = load_data()

    # --- METRICS & VARIABLES ---
    latest_row = history_df.iloc[-1]
    latest_aqi = latest_row['AQI']
    latest_date = latest_row['Date'].strftime('%Y-%m-%d')
    tomorrow_aqi = forecast_df.iloc[0]['Predicted_AQI']
    
    # Identify Dominant Pollutant (Finding the max among SubIndex columns)
    subindex_cols = [c for c in history_df.columns if 'SubIndex' in c]
    if subindex_cols:
        dom_pollutant_col = latest_row[subindex_cols].astype(float).idxmax()
        dom_pollutant = dom_pollutant_col.replace('_SubIndex', '')
    else:
        dom_pollutant = "Unknown"

    # --- HEADER ---
    st.title("Chennai AQI Monitor & 7-Day Forecast")
    st.markdown(f"**Live data from Perungudi Station | Powered by dynamically selected {active_model} Model**")

    # --- KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Date", latest_date)
    col2.metric("Current AQI", int(latest_aqi))
    col3.metric("Dominant Pollutant", dom_pollutant)
    
    def get_aqi_status(aqi):
        if aqi <= 50: return "Good 🟢"
        elif aqi <= 100: return "Moderate 🟡"
        elif aqi <= 150: return "Poor 🟠"
        elif aqi <= 200: return "Unhealthy 🔴"
        else: return "Hazardous 🟤"
        
    col4.metric("Status", get_aqi_status(latest_aqi))

    # --- AI RECOMMENDATIONS & PLANNER ---
    st.subheader("🤖 AI Health Insights & Planner")
    
    with st.spinner("Generating insights..."):
        recommendation, planner = get_ai_insights(int(latest_aqi), dom_pollutant, int(tomorrow_aqi))
    
    ai_col1, ai_col2 = st.columns(2)
    
    with ai_col1:
        st.info(f"**Today's Health Advice:**\n\n{recommendation}", icon="🩺")
        
    with ai_col2:
        # Change color based on tomorrow's forecast
        if tomorrow_aqi <= 100:
            st.success(f"**Tomorrow's Planner (Predicted AQI: {int(tomorrow_aqi)}):**\n\n{planner}", icon="📅")
        elif tomorrow_aqi <= 200:
            st.warning(f"**Tomorrow's Planner (Predicted AQI: {int(tomorrow_aqi)}):**\n\n{planner}", icon="📅")
        else:
            st.error(f"**Tomorrow's Planner (Predicted AQI: {int(tomorrow_aqi)}):**\n\n{planner}", icon="📅")

    # --- MAIN PLOT ---
    st.subheader(f"Historical Trend + 7 Day Prediction ({active_model})")
        
    fig = go.Figure()
        
    # Historical Data (Last 30 days)
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
        name=f'{active_model} Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="AQI", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("Detailed Forecast")

    # Clean display dataframe
    display_forecast = forecast_df.copy()
    display_forecast['Date'] = display_forecast['Date'].dt.date
    display_forecast['Predicted_AQI'] = display_forecast['Predicted_AQI'].round(0).astype(int)
    display_forecast['Status'] = display_forecast['Predicted_AQI'].apply(get_aqi_status)
    
    st.dataframe(display_forecast, hide_index=True, use_container_width=True)

except FileNotFoundError:
    st.error("Data files not found. Ensure your `daily_runner.py` script has successfully generated the CSVs.")
except Exception as e:
    st.error(f"An error occurred: {e}")
