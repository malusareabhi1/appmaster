import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from datetime import date, timedelta

st.set_page_config(page_title="Stock Trend Reversal Detector", layout="wide")

st.title("📉 Trend Reversal Detector for Multiple Stocks")

# Input Tickers
ticker_input = st.text_area("Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS, INFY.NS)", value="RELIANCE.NS, INFY.NS, TCS.NS")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Date Range
start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date = st.date_input("End Date", value=date.today())

reversal_signals = []

def detect_reversal(df):
    close_series = df['Close']
    df['SMA9'] = SMAIndicator(close=close_series, window=9).sma_indicator()
    df['SMA21'] = SMAIndicator(close=close_series, window=21).sma_indicator()
    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()

    df['Signal'] = None
    for i in range(1, len(df)):
        # Bullish Reversal
        if (df['SMA9'].iloc[i-1] < df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] > df['SMA21'].iloc[i]) and df['RSI'].iloc[i] > 40:
            df.loc[df.index[i], 'Signal'] = 'Bullish Reversal'
        # Bearish Reversal
        elif (df['SMA9'].iloc[i-1] > df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] < df['SMA21'].iloc[i]) and df['RSI'].iloc[i] < 60:
            df.loc[df.index[i], 'Signal'] = 'Bearish Reversal'
    return df

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty or 'Close' not in df.columns:
            st.warning(f"No valid data for {ticker}")
            continue

        df = detect_reversal(df)
        last_signal = df[df['Signal'].notnull()].iloc[-1] if not df[df['Signal'].notnull()].empty else None

        if last_signal is not None:
            reversal_signals.append({
                "Stock": ticker,
                "Date": last_signal.name.date(),
