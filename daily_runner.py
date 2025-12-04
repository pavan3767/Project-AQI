import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import select_coint_rank

# --- CONFIGURATION ---
FILE_NAME = "Cleaned AQI Bulk data (2nd Dec).csv" # Your existing filename
FORECAST_FILE = "latest_forecast.csv"
STATION_ID = 13738  # Perungudi
TOKEN = os.environ.get("WAQI_TOKEN") # Securely loaded from secrets

# --- STEP 1: LOAD HISTORY ---
try:
    df = pd.read_csv(FILE_NAME)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
except FileNotFoundError:
    print("Error: Historical data file not found.")
    exit(1)

# --- STEP 2: FETCH LIVE DATA ---
print("Fetching live data...")
url = f"https://api.waqi.info/feed/@{STATION_ID}/?token={TOKEN}"
try:
    response = requests.get(url)
    data = response.json()
    
    if data['status'] != 'ok':
        raise ValueError("API returned error")
        
    iaqi = data['data']['iaqi']
    
    # Extract available metrics (WAQI keys -> Your CSV columns)
    new_row = {
        'Date': pd.to_datetime(datetime.now().date()),
        'PM2.5': iaqi.get('pm25', {}).get('v', np.nan),
        'PM10': iaqi.get('pm10', {}).get('v', np.nan),
        'NO2': iaqi.get('no2', {}).get('v', np.nan),
        'NH3': iaqi.get('nh3', {}).get('v', np.nan),
        'SO2': iaqi.get('so2', {}).get('v', np.nan),
        'CO': iaqi.get('co', {}).get('v', np.nan),
        'Ozone': iaqi.get('o3', {}).get('v', np.nan),
        'RH': iaqi.get('h', {}).get('v', np.nan),
        'Temp': iaqi.get('t', {}).get('v', np.nan),
        'AQI_calculated': data['data']['aqi'] # Map 'aqi' to your target column
    }
    
    # Append new row
    new_df = pd.DataFrame([new_row])
    
    # Check if today's date already exists to avoid duplicates
    if new_df['Date'].iloc[0] not in df['Date'].values:
        df = pd.concat([df, new_df], ignore_index=True)
        # Forward fill missing values (like NO, NOx if API doesn't provide them)
        df = df.ffill()
        # Save updated history
        df.to_csv(FILE_NAME, index=False)
        print(f"Added data for {new_df['Date'].iloc[0]}")
    else:
        print("Data for today already exists.")

except Exception as e:
    print(f"API Fetch failed: {e}")
    # We continue to forecasting even if API fails, using old data

# --- STEP 3: VECM FORECASTING ---
print("Training VECM model...")

# Prepare data for VECM (Numeric only, drop non-numeric columns like 'Checks')
# We select columns that have valid data
train_df = df.select_dtypes(include=[np.number]).dropna()
# Ensure AQI_calculated is included
if 'AQI_calculated' not in train_df.columns:
    print("Error: AQI_calculated column missing")
    exit(1)

# Fit Model (Auto-detect lag order or use fixed)
order_res = select_order(train_vecm, maxlags=10, deterministic="li")
lag_order = order_res.aic
rank_res = select_coint_rank(train_vecm, det_order=0, k_ar_diff=lag_order, method='trace')
rank = rank_res.rank
model = VECM(train_vecm, k_ar_diff=lag_order, coint_rank=rank, deterministic='li')
vecm_fit = model.fit()

# Predict next 7 days
prediction = vecm_fit.predict(steps=7)

# Create Forecast DataFrame
last_date = df['Date'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

# We are interested mainly in 'AQI_calculated' column index
aqi_col_index = train_df.columns.get_loc('AQI_calculated')
predicted_aqi = prediction[:, aqi_col_index]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_AQI': predicted_aqi
})

# Save forecast for the Frontend
forecast_df.to_csv(FORECAST_FILE, index=False)

print("Forecast generated and saved.")

