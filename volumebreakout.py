import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Breakout with Volume Scanner")

# Sidebar Settings
lookback = st.sidebar.slider("Lookback Candles", 2, 10, 3)
vol_multiplier = st.sidebar.slider("Volume Multiplier", 1.0, 5.0, 1.5)

stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'SBIN.NS', 'ICICIBANK.NS']
selected = st.multiselect("Select Stocks", stocks, default=stocks)

start = datetime.today() - timedelta(days=90)
end = datetime.today()

results = []

if st.button("🔍 Scan for Breakouts"):
    for symbol in selected:
        try:
            df = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
            if df.empty or len(df) < lookback + 2:
                st.warning(f"{symbol}: Not enough data.")
                continue

            # Calculate previous high and avg volume
            df['Prev_High'] = df['High'].shift(1).rolling(window=lookback).max()
            df['Avg_Volume'] = df['Volume'].shift(1).rolling(window=lookback).mean()

            # Drop all rows with NaNs in any comparison column
            df = df.dropna(subset=['Close', 'Prev_High', 'Volume', 'Avg_Volume'])

            # Ensure last row is valid
            if df.empty:
                st.warning(f"{symbol}: Data not ready after cleaning.")
                continue

            latest = df.iloc[-1]

            # Perform scalar (safe) comparison
            if (
                latest['Close'] > latest['Prev_High'] and
                latest['Volume'] > latest['Avg_Volume'] * vol_multiplier
            ):
                results.append({
                    "Stock": symbol,
                    "Date": latest.name.date(),
                    "Close": round(latest["Close"], 2),
                    "Breakout Above": round(latest["Prev_High"], 2),
                    "Volume": int(latest["Volume"]),
                    "Avg Volume": int(latest["Avg_Volume"]),
                    "Signal": "✅ Breakout + Volume"
                })

        except Exception as e:
            st.error(f"{symbol}: {e}")

    st.subheader("📈 Breakout Results")
    if results:
        df_result = pd.DataFrame(results)
        st.dataframe(df_result)
        st.download_button("📥 Download CSV", df_result.to_csv(index=False), "breakout_volume.csv")
    else:
        st.info("No breakout signals found.")
