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
selected = st.multiselect("Select Stocks to Scan", stocks, default=stocks)

start = datetime.today() - timedelta(days=90)
end = datetime.today()

results = []

if st.button("🔍 Scan for Breakouts"):
    for symbol in selected:
        try:
            df = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
            if df.empty or len(df) < lookback + 2:
                continue

            # Rolling calculations
            df['Prev_High'] = df['High'].shift(1).rolling(lookback).max()
            df['Avg_Volume'] = df['Volume'].shift(1).rolling(lookback).mean()

            # Drop rows with NaNs
            df = df.dropna(subset=["Close", "Prev_High", "Volume", "Avg_Volume"])

            # Align operands before comparison
            close_aligned, high_aligned = df["Close"].align(df["Prev_High"], join="inner")
            vol_aligned, avg_vol_aligned = df["Volume"].align(df["Avg_Volume"], join="inner")

            # Perform aligned comparisons
            breakout_condition = (close_aligned > high_aligned) & (vol_aligned > avg_vol_aligned * vol_multiplier)
            breakout_df = df.loc[breakout_condition.index].copy()
            breakout_df = breakout_df[breakout_condition]

            if not breakout_df.empty:
                latest = breakout_df.iloc[-1]
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
