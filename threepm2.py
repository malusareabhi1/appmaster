import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("NIFTY 15-min Breakout Strategy with Target & Trailing SL")

# User parameters
trade_date = st.date_input("Select trade date", datetime.date.today())
target_pts = st.number_input("Target in points (from entry)", min_value=10, value=30, step=1)
trailing_sl_pts = st.number_input("Trailing Stop Loss in points", min_value=5, value=20, step=1)

ticker = "^NSEI"

# Download data
data = yf.download(
    tickers=ticker,
    period="5d",
    interval="15m",
    progress=False
)
data = data.reset_index().rename(columns={"Datetime":"datetime"})
if data["datetime"].dt.tz is None:
    data["datetime"] = data["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
else:
    data["datetime"] = data["datetime"].dt.tz_convert("Asia/Kolkata")

# Find previous day with a 3:00PM candle
curr = trade_date - datetime.timedelta(days=1)
looked_back = 0
while looked_back < 7:
    last_day_data = data[data['datetime'].dt.date == curr]
    candle_3pm = last_day_data[last_day_data['datetime'].dt.time == datetime.time(15, 0)]
    if not candle_3pm.empty:
        break
    curr -= datetime.timedelta(days=1)
    looked_back += 1
if candle_3pm.empty:
    st.error("No 3:00PM candle found in last week!")
    st.stop()
ref_high, ref_low = float(candle_3pm['High'].iloc[0]), float(candle_3pm['Low'].iloc[0])
ref_date = curr

st.info(f"Reference: {ref_date} 3:00 PM | High: {ref_high:.2f} Low: {ref_low:.2f}")

today_data = data[data['datetime'].dt.date == trade_date]
today_data = today_data[today_data['datetime'].dt.time >= datetime.time(9, 30)].reset_index(drop=True)
if today_data.empty:
    st.warning("No data for selected trade date after 9:30 AM yet.")
    st.stop()

# Strategy: Entry logic
signal = None; signal_time = None; entry_price = None; entry_idx = None; side = None
for i, row in today_data.iterrows():
    high, low = float(row['High']), float(row['Low'])
    if high > ref_high:
        signal = "CALL (Buy CE)"
        signal_time = row['datetime']
        entry_price = float(row['Close'])
        entry_idx = i
        side = "LONG"
        break
    elif low < ref_low:
        signal = "PUT (Buy PE)"
        signal_time = row['datetime']
        entry_price = float(row['Close'])
        entry_idx = i
        side = "SHORT"
        break

if not signal:
    st.info("No breakout/breakdown signal today after 9:30 AM.")
    st.stop()

# --- Trade Management: Target and Trailing SL ---
trade_outcome = "Open"
exit_price = None
exit_time = None

prices_checked = []

if side == "LONG":
    peak_price = entry_price
    for j in range(entry_idx+1, len(today_data)):
        price = float(today_data.at[j, 'Close'])
        prices_checked.append((today_data.at[j, "datetime"], price))
        # Update peak for trailing SL
        if price > peak_price:
            peak_price = price
        # Fixed target reached
        if price >= entry_price + target_pts:
            trade_outcome = f"Target Hit (+{target_pts} pts)"
            exit_price = price
            exit_time = today_data.at[j, "datetime"]
            break
        # Trailing stop loss hit
        if price <= peak_price - trailing_sl_pts:
            trade_outcome = f"Trailing SL Hit ({trailing_sl_pts} pts from peak)"
            exit_price = price
            exit_time = today_data.at[j, "datetime"]
            break
    else: # If not broken out of loop
        trade_outcome = "Open till EOD"
        exit_price = float(today_data.at[len(today_data)-1, 'Close'])
        exit_time = today_data.at[len(today_data)-1, "datetime"]

elif side == "SHORT":
    trough_price = entry_price
    for j in range(entry_idx+1, len(today_data)):
        price = float(today_data.at[j, 'Close'])
        prices_checked.append((today_data.at[j, "datetime"], price))
        # Update trough for trailing SL
        if price < trough_price:
            trough_price = price
        # Fixed target reached
        if price <= entry_price - target_pts:
            trade_outcome = f"Target Hit (-{target_pts} pts)"
            exit_price = price
            exit_time = today_data.at[j, "datetime"]
            break
        # Trailing stop loss hit
        if price >= trough_price + trailing_sl_pts:
            trade_outcome = f"Trailing SL Hit ({trailing_sl_pts} pts from trough)"
            exit_price = price
            exit_time = today_data.at[j, "datetime"]
            break
    else:
        trade_outcome = "Open till EOD"
        exit_price = float(today_data.at[len(today_data)-1, 'Close'])
        exit_time = today_data.at[len(today_data)-1, "datetime"]

# Display trade signals and result
st.success(f"Trade Signal: **{signal}**")
st.write(f"Entry: {pd.to_datetime(signal_time).strftime('%Y-%m-%d %H:%M')}, Price: {entry_price:.2f}")
st.write(f"Exit: {pd.to_datetime(exit_time).strftime('%Y-%m-%d %H:%M')}, Price: {exit_price:.2f}")
st.write(f"Result: {trade_outcome}")
st.write(f"PnL: {(exit_price-entry_price) if side=='LONG' else (entry_price-exit_price):.2f} pts")

with st.expander("Trades Checked"):
    st.write(pd.DataFrame(prices_checked, columns=["Datetime", "Close"]))

with st.expander("Today's candles after 9:30AM"):
    st.dataframe(today_data[['datetime', 'Open', 'High', 'Low', 'Close']].tail(10))

with st.expander("Previous day's 3:00 PM candle"):
    st.dataframe(candle_3pm)
