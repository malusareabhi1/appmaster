import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("NIFTY Option Trading Strategy (Last Day 3PM Candle Breakout)")

# User selects the trading date
trade_date = st.date_input("Select trade date", datetime.date.today())

ticker = "^NSEI"  # Nifty index symbol on Yahoo Finance

# Download last 5 days 15-min data (more robust for weekends/holidays)
data = yf.download(
    tickers=ticker,
    period="5d",
    interval="15m",
    progress=False
)

# Reset index and rename
data = data.reset_index()
data = data.rename(columns={"Datetime": "datetime"})

# Convert to India time if not already tz-aware
if data["datetime"].dt.tz is None:
    data["datetime"] = data["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
else:
    data["datetime"] = data["datetime"].dt.tz_convert("Asia/Kolkata")

# Find previous trading day with at least one 3:00 PM candle
curr = trade_date - datetime.timedelta(days=1)
looked_back = 0
while looked_back < 7:
    last_day_data = data[data['datetime'].dt.date == curr]
    # Check if 3:00 PM candle exists
    candle_3pm = last_day_data[last_day_data['datetime'].dt.time == datetime.time(15, 0)]
    if not candle_3pm.empty:
        break
    curr -= datetime.timedelta(days=1)
    looked_back += 1

if candle_3pm.empty:
    st.error("No 3:00 PM candle found for any recent trading day!")
    st.stop()

# Extract previous day's 3:00 PM high and low as FLOATS
ref_high = float(candle_3pm['High'].iloc[0])
ref_low = float(candle_3pm['Low'].iloc[0])
ref_date = curr

st.info(f"Reference date: {ref_date} | 3:00 PM High: {ref_high:.2f} Low: {ref_low:.2f}")

# Get current day after 9:30 AM
today_data = data[data['datetime'].dt.date == trade_date]
today_data = today_data[today_data['datetime'].dt.time >= datetime.time(9, 30)].reset_index(drop=True)

if today_data.empty:
    st.warning("No data for selected trade date after 9:30 AM yet.")
    st.stop()

# Strategy logic
signal = None
signal_time = None
signal_price = None

for i, row in today_data.iterrows():
    high = float(row['High'])
    low = float(row['Low'])
    if high > ref_high:
        signal = "CALL (Buy CE)"
        signal_time = row['datetime']
        signal_price = row['Close']
        break
    elif low < ref_low:
        signal = "PUT (Buy PE)"
        signal_time = row['datetime']
        signal_price = row['Close']
        break

if signal:
    # Ensure signal_time is a scalar
    if isinstance(signal_time, pd.Series):
        signal_time = signal_time.iloc[0]
    signal_time = pd.to_datetime(signal_time)
    # Ensure signal_price is a scalar
    if isinstance(signal_price, pd.Series):
        signal_price = float(signal_price.iloc[0])
    else:
        signal_price = float(signal_price)
    st.success(f"Trade Signal: **{signal}**")
    st.write(f"Triggered at: {signal_time.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"Price: {signal_price:.2f}")
else:
    st.info("No breakout/breakdown after 9:30 AM on selected date.")



# Optional: Show table of today's data after 9:30 AM
with st.expander("See today's (after 9:30AM) data"):
    st.dataframe(today_data[['datetime', 'Open', 'High', 'Low', 'Close']].tail(10))

# Optional: Show previous day's 3:00 PM candle for verification
with st.expander("See previous day's 3PM candle details"):
    st.dataframe(candle_3pm)
