import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.title("📈 20/50 EMA Crossover Scanner")

# List of some NIFTY 50 stocks (customize as you want)
nifty50_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "WIPRO.NS", "MARUTI.NS", "BAJFINANCE.NS"
]

# User input: select stocks to scan
selected_stocks = st.multiselect("Select Stocks to Scan", nifty50_stocks, default=nifty50_stocks)

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=200)

signals = []

if st.button("Run EMA Crossover Scan"):
    progress_text = "Scanning stocks..."
    progress_bar = st.progress(0)

    for i, stock in enumerate(selected_stocks):
        try:
            df = yf.download(stock, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 2:
                st.warning(f"{stock}: Not enough data")
                continue

            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

            prev = df.iloc[-2]
            latest = df.iloc[-1]

            # Extract scalar values safely
            prev_ema20 = prev["EMA20"].item() if hasattr(prev["EMA20"], "item") else prev["EMA20"]
            prev_ema50 = prev["EMA50"].item() if hasattr(prev["EMA50"], "item") else prev["EMA50"]
            latest_ema20 = latest["EMA20"].item() if hasattr(latest["EMA20"], "item") else latest["EMA20"]
            latest_ema50 = latest["EMA50"].item() if hasattr(latest["EMA50"], "item") else latest["EMA50"]

            # Check Golden Cross
            if prev_ema20 < prev_ema50 and latest_ema20 > latest_ema50:
                signals.append({"Stock": stock, "Signal": "📈 Golden Cross"})

            # Check Death Cross
            elif prev_ema20 > prev_ema50 and latest_ema20 < latest_ema50:
                signals.append({"Stock": stock, "Signal": "📉 Death Cross"})

        except Exception as e:
            st.error(f"Error for {stock}: {e}")

        progress_bar.progress((i + 1) / len(selected_stocks))

    progress_bar.empty()

    if signals:
        st.success(f"✅ Found {len(signals)} crossover signals:")
        st.table(pd.DataFrame(signals))
    else:
        st.info("No crossover signals found.")
