import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("NIFTY Option Trading Strategy (Live Data)")

# User inputs on UI
trade_date = st.date_input("Select trade date", datetime.datetime.now().date())

# Ticker symbol for NIFTY on Yahoo Finance
# "^NSEI" is commonly used for Nifty 50 index
ticker = "^NSEI"

# Fetch last 2 trading days of data at 15-min interval
# We fetch extra day so we can get prev day's 3:00 PM candle

data = yf.download(
    tickers=ticker,
    period="5d",             # last 5 days just to be safe
    interval="15m",
    progress=False,
)

data.reset_index(inplace=True)

# Convert to india timezone (if required)
#data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
# Convert to india timezone ONLY IF not already tz-aware
if data['Datetime'].dt.tz is None:  # tz-naive
    data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
else:  # tz-aware
    data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')

# Extract last trading day for strategy candle
last_trading_date = trade_date - datetime.timedelta(days=1)

last_day_data = data[data['Datetime'].dt.date == last_trading_date]

if last_day_data.empty:
    st.error(f"No data available for previous trading day: {last_trading_date}")
    st.stop()

# Find the 3:00 PM candle of last trading day
candle_3pm = last_day_data[last_day_data['Datetime'].dt.time == datetime.time(15, 0)]

if candle_3pm.empty:
    st.error("No 3:00 PM candle found for last trading day")
    st.stop()

if len(candle_3pm) == 0:
    st.error("No 3:00 PM candle found for last trading day.")
    st.stop()
elif len(candle_3pm) > 1:
    st.warning("More than one 3:00 PM candle found, taking the first one.")

ref_high = float(candle_3pm['High'].iloc[0])
ref_low = float(candle_3pm['Low'].iloc[0])


st.write(f"Reference candle on {last_trading_date} 3:00 PM — High: {ref_high}, Low: {ref_low}")

# Get today's data after 9:30 AM
today_data = data[data['Datetime'].dt.date == trade_date]
today_after_930 = today_data[today_data['Datetime'].dt.time >= datetime.time(9, 30)].reset_index(drop=True)

if today_after_930.empty:
    st.warning("No data available for today after 9:30 AM yet.")
    st.stop()

signal = None
trigger_time = None
trigger_price = None

# Check signals in chronological order
for idx, row in today_after_930.iterrows():
    # Check breakout above last day 3pm high
    if row['High'] > ref_high:
        signal = "CALL (Buy CE)"
        trigger_time = row['Datetime']
        trigger_price = row['Close']
        break
    # Check breakdown below last day 3pm low
    elif row['Low'] < ref_low:
        signal = "PUT (Buy PE)"
        trigger_time = row['Datetime']
        trigger_price = row['Close']
        break

if signal:
    st.success(
        f"Trade Signal: {signal}\n"
        f"Triggered at: {trigger_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Price: {trigger_price:.2f}"
    )
else:
    st.info("No breakout/breakdown signal detected after 9:30 AM today.")

# Show data tables (optional)
with st.expander("View Today's Data After 9:30 AM"):
    st.write(today_after_930[['Datetime', 'Open', 'High', 'Low', 'Close']].tail(10))

with st.expander("View Last Trading Day Data"):
    st.write(last_day_data[['Datetime', 'Open', 'High', 'Low', 'Close']].tail(10))
