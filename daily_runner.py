import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from AQI_SubIndex import get_all_subindices_single

# --- CONFIGURATION ---
FILE_NAME = "Cleaned AQI Bulk data (26th Jan).csv"
FORECAST_FILE = "latest_forecast.csv"
STATION_ID = 13738
TOKEN = os.environ.get("WAQI_TOKEN")

# ---------------- STEP 1: LOAD HISTORY ----------------
try:
    df = pd.read_csv(FILE_NAME)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print("Loaded history. Latest date:", df['Date'].max())
except FileNotFoundError:
    raise RuntimeError("Historical data file not found")

# ---------------- STEP 2: FETCH LIVE DATA ----------------
print("Fetching live data...")

url = f"https://api.waqi.info/feed/@{STATION_ID}/?token={TOKEN}"

try:
    response = requests.get(url, timeout=30)
    data = response.json()

    if data.get('status') != 'ok':
        raise RuntimeError(f"WAQI API Error: {data}")

    iaqi = data['data'].get('iaqi', {})

    today_date = pd.to_datetime(datetime.now().date())

    new_row = {
        'Date': today_date,
        'PM2.5': iaqi.get('pm25', {}).get('v', np.nan),
        'PM10': iaqi.get('pm10', {}).get('v', np.nan),
        'NO2': iaqi.get('no2', {}).get('v', np.nan),
        'NH3': iaqi.get('nh3', {}).get('v', np.nan),
        'SO2': iaqi.get('so2', {}).get('v', np.nan),
        'CO': iaqi.get('co', {}).get('v', np.nan),
        'Ozone': iaqi.get('o3', {}).get('v', np.nan),
        'RH': iaqi.get('h', {}).get('v', np.nan),
        'Temp': iaqi.get('t', {}).get('v', np.nan),
        'AQI': data['data'].get('aqi', np.nan)
    }

    new_row_final = get_all_subindices_single(new_row)
    new_df = pd.DataFrame([new_row_final])

    # --- SAFE DUPLICATE DATE CHECK ---
    already_exists = (df['Date'].dt.date == today_date.date()).any()

    if not already_exists:

        print("Appending new row for:", today_date.date())

        # ⭐ FIX: UNION CONCAT (Never shrink schema)
        df = pd.concat([df, new_df], ignore_index=True, sort=False)

        df = df.ffill()

        df.to_csv(FILE_NAME, index=False)

        print("New data appended and saved")

    else:
        print("Today already exists in dataset")

except Exception as e:
    print("CRITICAL: API ingestion failed:", e)
    raise  # ⭐ FAIL WORKFLOW (VERY IMPORTANT)

# ---------------- STEP 3: VALIDATE REQUIRED COLUMNS ----------------
required_cols = [
    'AQI',
    'PM2.5_SubIndex',
    'PM10_SubIndex',
    'Temp',
    'CO_SubIndex'
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise RuntimeError(f"Missing required columns for VECM: {missing_cols}")

# ---------------- STEP 4: VECM FORECASTING ----------------
print("Training VECM model...")

train_df = df[['Date', 'AQI', 'PM2.5_SubIndex', 'PM10_SubIndex', 'Temp', 'CO_SubIndex']].copy()

train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.set_index('Date')

train_df = train_df.ffill().bfill().dropna()

if len(train_df) < 50:
    raise RuntimeError("Not enough data for VECM training")

# --- MODEL FIT ---
order_res = select_order(train_df, maxlags=10, deterministic="li")
lag_order = order_res.aic

rank_res = select_coint_rank(
    train_df,
    det_order=0,
    k_ar_diff=lag_order,
    method='trace'
)

rank = rank_res.rank

model = VECM(
    train_df,
    k_ar_diff=lag_order,
    coint_rank=rank,
    deterministic='li'
)

vecm_fit = model.fit()

# ---------------- FORECAST ----------------
prediction = vecm_fit.predict(steps=7)

last_date = df['Date'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

aqi_idx = train_df.columns.get_loc('AQI')
predicted_aqi = prediction[:, aqi_idx]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_AQI': predicted_aqi
})

forecast_df.to_csv(FORECAST_FILE, index=False)

print("Forecast generated and saved")
print("Pipeline completed successfully")
