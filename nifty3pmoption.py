# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("🕒 3PM Breakout Nifty Options Strategy")

# Step 1: Load NIFTY 15-min data
ticker = "^NSEI"
today = datetime.now()
start = today - timedelta(days=5)

data = yf.download(ticker, interval="15m", start=start.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
data.index = data.index.tz_convert('Asia/Kolkata')

# Filter only last trading day's 3:00 PM candle
data["Date"] = data.index.date
data["Time"] = data.index.time
last_day = data["Date"].unique()[-2]
last_day_data = data[data["Date"] == last_day]
candle_3pm = last_day_data.between_time("15:00", "15:00")

if not candle_3pm.empty:
    high_3pm = candle_3pm["High"].values[0]
    low_3pm = candle_3pm["Low"].values[0]
    st.info(f"📌 3PM Candle on {last_day}: High = {high_3pm}, Low = {low_3pm}")

    # Step 2: Get today's 9:30 AM candle
    today_data = data[data["Date"] == data["Date"].unique()[-1]]
    candle_930 = today_data.between_time("09:30", "09:30")

    if not candle_930.empty:
        price_930 = candle_930["Close"].values[0]
        st.success(f"✅ Today 9:30 AM Price = {price_930}")

        # Step 3: Trade Logic
        if price_930 > high_3pm:
            st.markdown("🟢 **BUY CALL OPTION**")
            st.code("Strike ≈ ATM + 100")
        elif price_930 < low_3pm:
            st.markdown("🔴 **BUY PUT OPTION**")
            st.code("Strike ≈ ATM - 100")
        else:
            st.warning("📍 No breakout. No trade today.")
    else:
        st.warning("⏳ 9:30 AM candle not yet available.")
else:
    st.error("❌ No 3PM candle found for last day.")

# Optional enhancements: 
# 1. Integrate NSE Option Chain for strike selection
# 2. Add paper trading log & PnL
# 3. Telegram alert at 9:31 AM
