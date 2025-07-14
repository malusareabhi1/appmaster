import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.title("📊 Breakout with Volume Scanner")

# Sidebar inputs
lookback = st.sidebar.slider("Lookback Candles", 2, 10, 3)
vol_multiplier = st.sidebar.slider("Volume Multiplier", 1.0, 5.0, 1.5)

# Sample stocks (use your list as needed)
stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'SBIN.NS', 'ICICIBANK.NS']
selected = st.multiselect("Select Stocks", stocks, default=stocks)

start = datetime.today() - timedelta(days=60)
end = datetime.today()

results = []

if st.button("🔍 Run Breakout Scan"):
    for stock in selected:
        try:
            df = yf.download(stock, start=start, end=end, interval='1d', progress=False)
            df.dropna(inplace=True)

            # Create breakout and volume columns
            df['Prev_High'] = df['High'].shift(1).rolling(window=lookback).max()
            df['Avg_Vol'] = df['Volume'].shift(1).rolling(window=lookback).mean()

            df.dropna(inplace=True)  # Removes rows with NaNs from rolling

            # Check latest row for breakout signal
            latest = df.iloc[-1]
            if (
                latest['Close'] > latest['Prev_High']
                and latest['Volume'] > latest['Avg_Vol'] * vol_multiplier
            ):
                results.append({
                    'Stock': stock,
                    'Date': latest.name.date(),
                    'Close': round(latest['Close'], 2),
                    'Breakout >': round(latest['Prev_High'], 2),
                    'Volume': int(latest['Volume']),
                    'Avg Volume': int(latest['Avg_Vol']),
                    'Signal': '✅ Breakout + Volume'
                })

        except Exception as e:
            st.error(f"Error with {stock}: {e}")

    st.subheader("📈 Breakout Results")
    if results:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
    else:
        st.info("No breakout signals found.")
