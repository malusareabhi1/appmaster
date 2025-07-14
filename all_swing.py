import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator

# ------------------------------------
# Page Title and Strategy Explanation
# ------------------------------------
st.set_page_config(page_title="EMA Crossover + Pullback Scanner", layout="wide")
st.title("📈 EMA Crossover & Pullback to 20 EMA Scanner")

with st.expander("ℹ️ Strategy Explanation"):
    st.markdown("""
### 🔶 Golden Cross 📈
- 20 EMA crosses above 50 EMA → **Bullish** (entry signal)

### 🔷 Death Cross 📉
- 20 EMA crosses below 50 EMA → **Bearish** (exit signal)

### 🟢 Pullback to EMA20 (Buy the Dip)
- Price above EMA20 and EMA50 (Uptrend)
- Pullback to near EMA20 (within 1%)
- **Reversal candlestick** (Bullish Engulfing or Hammer)
- RSI > 40 for confirmation
""")

# ------------------------------------
# Stock selection and date range
# ------------------------------------
nifty50_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'SBIN.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'AXISBANK.NS', 'ITC.NS', 'LT.NS', 'MARUTI.NS']
selected_stocks = st.multiselect("Select Stocks to Scan", nifty50_stocks, default=nifty50_stocks)

end_date = datetime.today()
start_date = end_date - timedelta(days=200)

scan_crossover = st.checkbox("Scan for Golden/Death Cross", value=True)
scan_pullback = st.checkbox("Scan for Pullback to EMA20 (Buy the Dip)", value=True)

# ------------------------------------
# Main Scan Logic
# ------------------------------------
signals = []
pullback_signals = []

if st.button("🔍 Run Scan"):
    progress_bar = st.progress(0, text="Starting scan...")

    for i, stock in enumerate(selected_stocks):
        try:
            df = yf.download(stock, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 60:
                st.warning(f"{stock}: Not enough data")
                continue

            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # ------------- EMA Crossover -------------
            if scan_crossover:
                prev_ema20 = prev["EMA20"]
                prev_ema50 = prev["EMA50"]
                latest_ema20 = latest["EMA20"]
                latest_ema50 = latest["EMA50"]

                if prev_ema20 < prev_ema50 and latest_ema20 > latest_ema50:
                    signals.append({"Stock": stock, "Signal": "📈 Golden Cross"})

                elif prev_ema20 > prev_ema50 and latest_ema20 < latest_ema50:
                    signals.append({"Stock": stock, "Signal": "📉 Death Cross"})

            # ------------- Buy the Dip -------------
            if scan_pullback:
                # Condition 1: Uptrend
                in_uptrend = latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA50"]

                # Condition 2: Close near EMA20 (within 1%)
                near_ema20 = abs(latest["Close"] - latest["EMA20"]) / latest["Close"] < 0.01

                # Condition 3: Reversal candle
                is_bullish_engulfing = (
                    prev["Close"] < prev["Open"] and
                    latest["Close"] > latest["Open"] and
                    latest["Close"] > prev["Open"] and
                    latest["Open"] < prev["Close"]
                )

                is_hammer = (
                    latest["Close"] > latest["Open"] and
                    (latest["Open"] - latest["Low"]) > 2 * (latest["Close"] - latest["Open"])
                )

                # Condition 4: RSI > 40
                rsi_ok = latest["RSI"] > 40

                if in_uptrend and near_ema20 and rsi_ok and (is_bullish_engulfing or is_hammer):
                    pullback_signals.append({"Stock": stock, "Signal": "🟢 Pullback Buy"})

        except Exception as e:
            st.error(f"Error with {stock}: {e}")

        progress_bar.progress((i + 1) / len(selected_stocks), text=f"Scanning {stock}...")

    progress_bar.empty()

    # ----------------- Results -----------------
    if scan_crossover:
        st.subheader("📊 EMA Crossover Results")
        if signals:
            st.dataframe(pd.DataFrame(signals))
        else:
            st.info("No EMA crossover signals found.")

    if scan_pullback:
        st.subheader("📉 Pullback to 20 EMA (Buy the Dip)")
        if pullback_signals:
            st.dataframe(pd.DataFrame(pullback_signals))
        else:
            st.info("No pullback buy signals found.")
