import streamlit as st
import pandas as pd
import datetime

st.title("NIFTY Option Trading Strategy: Last Day 3PM Candle Breakout")

# Upload your 15-min NIFTY data (CSV format)
uploaded_file = st.file_uploader("Upload NIFTY 15m OHLC csv", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['datetime'])

    # User option to pick date
    trading_date = st.date_input("Today's date (Strategy day)", datetime.date.today())
    last_trading_day = trading_date - datetime.timedelta(days=1)
    st.write(f'Backtest for Trade Date: {trading_date}, Reference Candle: {last_trading_day} 3:00 PM')

    # Extract last day's 3PM candle (15:00:00)
    mask_last_day = (df['datetime'].dt.date == last_trading_day) & (df['datetime'].dt.time == datetime.time(hour=15, minute=0))
    ref_candle = df[mask_last_day]

    if not ref_candle.empty:
        ref_high = ref_candle['high'].values[0]
        ref_low = ref_candle['low'].values[0]
        st.write(f"Reference Candle High: {ref_high}, Low: {ref_low}")

        # Today's data after 9:30am
        mask_today = (df['datetime'].dt.date == trading_date) & (df['datetime'].dt.time >= datetime.time(hour=9, minute=30))
        today_data = df[mask_today].reset_index(drop=True)

        signal = ""
        trigger_row = None

        # Check breakout logic
        for idx, row in today_data.iterrows():
            if row['high'] > ref_high:
                signal = "CALL (Buy CE)"
                trigger_row = row
                break
            elif row['low'] < ref_low:
                signal = "PUT (Buy PE)"
                trigger_row = row
                break

        if signal:
            st.success(f"Signal: {signal} at {trigger_row['datetime']} [Price: {trigger_row['close']}]")
        else:
            st.info("No breakout/trade signal so far today.")
    else:
        st.error(f"No data for last trading day 3:00 PM candle.")

st.markdown("""
**How To Use:**
1. Export/download your NIFTY 15-min OHLC data as CSV with columns: `datetime`, `open`, `high`, `low`, `close`.
2. Upload the file above and select the intended date.
3. The app will show if/when a breakout trade signal is triggered as per your strategy.
""")
