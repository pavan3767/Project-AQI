import numpy as np
from river.drift import ADWIN

# ---------- ADWIN DETECTORS ----------
adwin_detectors = {
    "AQI": ADWIN(),
    "PM25": ADWIN(),
    "PM10": ADWIN(),
    "CO": ADWIN()
}

# ---------- ADWIN CHECK ----------
def check_adwin_drift(latest_row):

    drift_flags = {}

    for col, detector in adwin_detectors.items():

        val = latest_row.get(col, np.nan)

        if np.isnan(val):
            drift_flags[col] = False
            continue

        detector.update(val)
        drift_flags[col] = detector.drift_detected

    return drift_flags


# ---------- PSI ----------
def calculate_psi(expected, actual, buckets=10):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, buckets + 1)

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi


def check_psi_drift(df, column, window=30):

    if len(df) < window * 2:
        return 0

    expected = df[column].iloc[-2*window:-window]
    actual = df[column].iloc[-window:]

    return calculate_psi(expected, actual)


# ---------- MAPE ----------
def rolling_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    return np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
