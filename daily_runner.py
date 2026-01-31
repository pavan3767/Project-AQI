import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

from AQI_SubIndex import get_all_subindices_single
from drift_detection import check_adwin_drift, check_psi_drift, compute_rolling_mape
from model_selector import calculate_drift_score, select_model
from forecast_models import vecm_forecast, prophet_forecast


# ---------------- CONFIG ----------------
FILE_NAME = "Cleaned AQI Bulk data (26th Jan).csv"
FORECAST_FILE = "latest_forecast.csv"
DRIFT_LOG_FILE = "drift_log.csv"
MODEL_LOG_FILE = "model_log.csv"

STATION_ID = 13738
TOKEN = os.environ.get("WAQI_TOKEN")

FORECAST_STEPS = 7


# ---------------- STEP 1: LOAD HISTORY ----------------
try:
    df = pd.read_csv(FILE_NAME)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print("History Loaded. Latest Date:", df['Date'].max())

except FileNotFoundError:
    raise RuntimeError("Historical AQI file not found")


# ---------------- STEP 2: FETCH LIVE DATA ----------------
print("Fetching live data...")

url = f"https://api.waqi.info/feed/@{STATION_ID}/?token={TOKEN}"

try:
    response = requests.get(url, timeout=30)
    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError("WAQI API returned error")

    iaqi = data["data"].get("iaqi", {})

    today = pd.to_datetime(datetime.now().date())

    new_row = {
        "Date": today,
        "PM2.5": iaqi.get("pm25", {}).get("v", np.nan),
        "PM10": iaqi.get("pm10", {}).get("v", np.nan),
        "NO2": iaqi.get("no2", {}).get("v", np.nan),
        "NH3": iaqi.get("nh3", {}).get("v", np.nan),
        "SO2": iaqi.get("so2", {}).get("v", np.nan),
        "CO": iaqi.get("co", {}).get("v", np.nan),
        "Ozone": iaqi.get("o3", {}).get("v", np.nan),
        "RH": iaqi.get("h", {}).get("v", np.nan),
        "Temp": iaqi.get("t", {}).get("v", np.nan),
        "AQI": data["data"].get("aqi", np.nan)
    }

    new_row_final = get_all_subindices_single(new_row)
    new_df = pd.DataFrame([new_row_final])

    exists = (df["Date"].dt.date == today.date()).any()

    if not exists:
        print("Appending new row for:", today.date())

        df = pd.concat([df, new_df], ignore_index=True, sort=False)
        df = df.ffill()

        df.to_csv(FILE_NAME, index=False)

        print("New AQI row added successfully")

    else:
        print("Today already exists — skipping append")

except Exception as e:
    print("CRITICAL: API ingestion failed:", e)
    raise


# ---------------- STEP 3: DRIFT DETECTION ----------------

print("Running Drift Detection...")

latest = df.iloc[-1]

adwin_flags = check_adwin_drift({
    "AQI": latest["AQI"],
    "PM25": latest["PM2.5_SubIndex"],
    "PM10": latest["PM10_SubIndex"],
    "CO": latest["CO_SubIndex"]
})

psi_scores = {
    "AQI": check_psi_drift(df, "AQI"),
    "PM25": check_psi_drift(df, "PM2.5_SubIndex")
}

if os.path.exists("forecast_history.csv"):
    forecast_hist = pd.read_csv("forecast_history.csv")
    forecast_hist["Target_Date"] = pd.to_datetime(forecast_hist["Target_Date"])

    if len(forecast_hist) >= 7:
        mape_score = compute_rolling_mape(
            df[["Date", "AQI"]],
            forecast_hist,
            window=7
        )
    else:
        mape_score = None
    
if mape_score is None: mape_score = 0

drift_score = calculate_drift_score(adwin_flags, psi_scores, mape_score)
model_choice = select_model(drift_score)

print("Drift Score:", drift_score)
print("Model Selected:", model_choice)


# ---------------- STEP 4: FORECAST ----------------

print("Generating Forecast...")

if model_choice == "PROPHET":
    predicted_aqi = prophet_forecast(df, FORECAST_STEPS)

else:
    predicted_aqi = vecm_forecast(df, FORECAST_STEPS)


last_date = df["Date"].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, FORECAST_STEPS + 1)]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_AQI": predicted_aqi
})

forecast_df.to_csv(FORECAST_FILE, index=False)

print("Forecast Saved")


# ---------------- STEP 5: LOG DRIFT ----------------

drift_log_entry = pd.DataFrame([{
    "Date": datetime.now(),
    "ADWIN_AQI": adwin_flags["AQI"],
    "ADWIN_PM25": adwin_flags["PM25"],
    "ADWIN_PM10": adwin_flags["PM10"],
    "PSI_AQI": psi_scores["AQI"],
    "PSI_PM25": psi_scores["PM25"],
    "MAPE": mape_score,
    "Drift_Score": drift_score
}])

if os.path.exists(DRIFT_LOG_FILE):
    drift_log_entry.to_csv(DRIFT_LOG_FILE, mode="a", header=False, index=False)
else:
    drift_log_entry.to_csv(DRIFT_LOG_FILE, index=False)


# ---------------- STEP 6: LOG MODEL DECISION ----------------

model_log_entry = pd.DataFrame([{
    "Date": datetime.now(),
    "Selected_Model": model_choice,
    "Drift_Score": drift_score
}])

if os.path.exists(MODEL_LOG_FILE):
    model_log_entry.to_csv(MODEL_LOG_FILE, mode="a", header=False, index=False)
else:
    model_log_entry.to_csv(MODEL_LOG_FILE, index=False)


print("Drift and Model logs updated")
print("Pipeline Completed Successfully")


