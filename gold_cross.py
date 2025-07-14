import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.title("📈 20/50 EMA Crossover Scanner")

# List of NIFTY 50 stocks (can be customized)
nifty50_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "WIPRO.NS", "MARUTI.NS", "BAJFINANCE.NS"
]

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=200)

# Result list
signals = []

st.write("🔍 Scanning for EMA 20/50 crossover...")

for stock in nifty50_stocks:
    try:
        df = yf.download(stock, start=start_date, end=end_date)
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        print(type(prev["EMA20"]), type(prev["EMA50"]))  # should be float
        print(prev["EMA20"] < prev["EMA50"])             # should be True/False


        # Check for crossover in last 2 days
        if df.shape[0] < 2:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Golden Cross
        if prev["EMA20"] < prev["EMA50"] and latest["EMA20"] > latest["EMA50"]:
            signals.append({"Stock": stock, "Signal": "📈 Golden Cross"})

        # Death Cross
        elif prev["EMA20"] > prev["EMA50"] and latest["EMA20"] < latest["EMA50"]:
            signals.append({"Stock": stock, "Signal": "📉 Death Cross"})

    except Exception as e:
        st.error(f"Error for {stock}: {e}")

# Show results
if signals:
    result_df = pd.DataFrame(signals)
    st.success(f"✅ {len(result_df)} stocks found with crossover")
    st.dataframe(result_df)
else:
    st.warning("No crossover signals found today.")
