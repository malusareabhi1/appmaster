import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Breakout with Volume Scanner (NSE Stocks)")

# Sidebar Inputs
lookback_candles = st.sidebar.slider("Breakout Lookback (N candles)", 2, 10, 3)
volume_multiplier = st.sidebar.slider("Volume Spike Multiplier", 1.0, 5.0, 1.5)

# Sample list (customize with your full list as needed)
nifty_stocks = [
    'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'LT.NS', 'AXISBANK.NS', 'ITC.NS', 'BHARTIARTL.NS'
]
selected_stocks = st.multiselect("Select Stocks to Scan", nifty_stocks, default=nifty_stocks)

# Date range for fetching data
end_date = datetime.today()
start_date = end_date - timedelta(days=60)

breakout_results = []

if st.button("🔍 Scan for Breakouts"):
    progress_bar = st.progress(0, text="Scanning...")

    for i, stock in enumerate(selected_stocks):
        try:
            df = yf.download(stock, start=start_date, end=end_date, interval="1d", progress=False)
            df.dropna(inplace=True)

            if len(df) < lookback_candles + 1:
                st.warning(f"{stock}: Not enough data.")
                continue

            # Calculate breakout and volume stats
            df["Prev_High"] = df["High"].shift(1).rolling(window=lookback_candles).max()
            df["Avg_Volume"] = df["Volume"].shift(1).rolling(window=lookback_candles).mean()

            latest = df.iloc[-1]

            if latest["Close"] > latest["Prev_High"] and latest["Volume"] > latest["Avg_Volume"] * volume_multiplier:
                breakout_results.append({
                    "Stock": stock,
                    "Close": round(latest["Close"], 2),
                    "Breakout Above": round(latest["Prev_High"], 2),
                    "Volume": int(latest["Volume"]),
                    "Avg Volume": int(latest["Avg_Volume"]),
                    "Signal": "✅ Breakout + Volume"
                })

        except Exception as e:
            st.error(f"Error for {stock}: {e}")

        progress_bar.progress((i + 1) / len(selected_stocks))

    progress_bar.empty()

    st.subheader("📈 Breakout with Volume Results")
    if breakout_results:
        df_result = pd.DataFrame(breakout_results)
        st.dataframe(df_result)
        st.download_button("📥 Download CSV", df_result.to_csv(index=False), "breakout_volume_signals.csv", "text/csv")
    else:
        st.info("No breakout signals found.")
