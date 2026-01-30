import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank


# ---------- VECM ----------
def vecm_forecast(df, steps=7):

    train_df = df[['Date','AQI','PM2.5_SubIndex','PM10_SubIndex','Temp','CO_SubIndex']].copy()
    train_df = train_df.set_index('Date')
    train_df = train_df.ffill().bfill().dropna()

    order_res = select_order(train_df, maxlags=10, deterministic="li")
    lag_order = order_res.aic

    rank_res = select_coint_rank(train_df, det_order=0, k_ar_diff=lag_order, method='trace')
    rank = rank_res.rank

    model = VECM(train_df, k_ar_diff=lag_order, coint_rank=rank, deterministic='li')
    fit = model.fit()

    prediction = fit.predict(steps=steps)

    idx = train_df.columns.get_loc("AQI")
    return prediction[:, idx]


# ---------- PROPHET ----------
def prophet_forecast(df, steps=7):

    temp = df[['Date','AQI']].rename(columns={'Date':'ds','AQI':'y'})

    model = Prophet()
    model.fit(temp)

    future = model.make_future_dataframe(periods=steps)
    fc = model.predict(future)

    return fc.tail(steps)['yhat'].values
