# 📊 Streamlit App – 9 EMA + VWAP Reversal Intraday Strategy

import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import datetime

st.set_page_config(layout="wide", page_title="9 EMA + VWAP Reversal Scanner")
st.title("⚡ Intraday Scanner – 9 EMA + VWAP Reversal Strategy")

# ------------------- Sidebar --------------------
st.sidebar.header("📅 Scanner Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (NSE)", value="INFY.NS")
date_range = st.sidebar.date_input("Select Date Range", [datetime.date.today() - datetime.timedelta(days=5), datetime.date.today()])
timeframe = st.sidebar.selectbox("Time Interval", ["5m", "15m"], index=0)

# ------------------- Fetch Data --------------------
def load_data(ticker, interval, start, end):
    df = yf.download(ticker, interval=interval, start=start, end=end)
    if df.empty:
        st.warning("No data found. Try a different symbol or date.")
        return pd.DataFrame()
    df.dropna(inplace=True)

    # Technical indicators
    df["9EMA"] = ta.trend.EMAIndicator(close=df["Close"], window=9).ema_indicator()
    vwap = ta.volume.VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    df["VWAP"] = vwap.vwap()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    return df

# ------------------- Strategy Logic --------------------
def find_reversals(df):
    signals = []
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]

        # Conditions for BUY Reversal near VWAP
        cond1 = curr["Close"] > curr["VWAP"]
        cond2 = curr["9EMA"] > curr["VWAP"]
        cond3 = curr["RSI"] > 50
        cond4 = curr["Volume"] > prev["Volume"]
        cond5 = curr["Low"] <= curr["VWAP"] <= curr["High"]  # touched VWAP

        if cond1 and cond2 and cond3 and cond4 and cond5:
            signals.append({
                "Datetime": curr.name,
                "Close": curr["Close"],
                "VWAP": curr["VWAP"],
                "9EMA": curr["9EMA"],
                "RSI": curr["RSI"],
                "Volume": curr["Volume"]
            })
    return pd.DataFrame(signals)

# ------------------- Run Scanner --------------------
start_date = date_range[0]
end_date = date_range[1] + datetime.timedelta(days=1)

if st.button("🔍 Run Scan"):
    with st.spinner("Fetching and analyzing data..."):
        df = load_data(ticker, timeframe, start_date, end_date)
        if not df.empty:
            signal_df = find_reversals(df)
            st.subheader("📈 Reversal Signals")
            st.dataframe(signal_df, use_container_width=True)

            st.subheader("📊 Full Chart Data")
            st.dataframe(df.tail(50), use_container_width=True)
