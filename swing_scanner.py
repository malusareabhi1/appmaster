import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

# ========== Sidebar ==========
st.set_page_config(page_title="Swing Strategy Scanner", layout="wide")
st.title("📊 Swing Trade Strategy Scanner")

strategy = st.sidebar.selectbox("Choose Strategy", [
    "EMA Crossover (9/21)",
    "Pullback to EMA20",
    "Breakout with Volume",
    "Bollinger Squeeze Breakout",
    "Support Bounce + RSI Divergence",
    "RSI 60-40 Strategy"
])

tickers = st.sidebar.text_area("Paste NSE Tickers (e.g. INFY.NS)", "RELIANCE.NS\nTCS.NS\nINFY.NS\nSBIN.NS\nHDFCBANK.NS").splitlines()

start = datetime.today() - timedelta(days=60)
end = datetime.today()

@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df["Ticker"] = ticker
        df = df.dropna()
        return df
    except:
        return pd.DataFrame()

signals = []

for stock in tickers:
    df = get_data(stock)
    if df.empty or len(df) < 30:
        continue

    df["EMA9"] = EMAIndicator(df["Close"], window=9).ema_indicator()
    df["EMA21"] = EMAIndicator(df["Close"], window=21).ema_indicator()
    df["EMA20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    bb = BollingerBands(df["Close"])
    df["BB_Width"] = bb.bollinger_hband() - bb.bollinger_lband()
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Volume_avg"] = df["Volume"].rolling(window=20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ========== Strategy Checks ==========

    if strategy == "EMA Crossover (9/21)":
        if prev["EMA9"] < prev["EMA21"] and last["EMA9"] > last["EMA21"]:
            signals.append((stock, "🟢 EMA Bullish Crossover"))

    elif strategy == "Pullback to EMA20":
        price = last["Close"]
        near_ema20 = abs(price - last["EMA20"]) / price < 0.01
        uptrend = last["EMA20"] > last["EMA50"]
        is_hammer = last["Close"] > last["Open"] and (last["Open"] - last["Low"]) > 2 * (last["Close"] - last["Open"])
        if near_ema20 and uptrend and last["RSI"] > 40 and is_hammer:
            signals.append((stock, "🟢 Pullback Buy"))

    elif strategy == "Breakout with Volume":
        if last["Close"] > df["Close"].rolling(window=20).max().iloc[-2] and last["Volume"] > 1.5 * last["Volume_avg"]:
            signals.append((stock, "🟢 Breakout + Volume"))

    elif strategy == "Bollinger Squeeze Breakout":
        bb_min = df["BB_Width"].rolling(window=20).min().iloc[-1]
        if last["BB_Width"] <= bb_min * 1.1 and last["Close"] > last["BB_Upper"]:
            signals.append((stock, "🟢 BB Squeeze Breakout"))

    elif strategy == "Support Bounce + RSI Divergence":
        support = df["Close"].rolling(window=10).min().iloc[-2]
        if last["Close"] > prev["Close"] and last["Close"] > support and last["RSI"] > prev["RSI"]:
            signals.append((stock, "🟢 Support Bounce"))

    elif strategy == "RSI 60-40 Strategy":
        if prev["RSI"] < 60 and last["RSI"] > 60:
            signals.append((stock, "🟢 RSI Cross Above 60 (Buy)"))
        elif prev["RSI"] > 40 and last["RSI"] < 40:
            signals.append((stock, "🔻 RSI Cross Below 40 (Exit)"))

# ========== Show Results ==========

st.subheader(f"📈 Strategy: {strategy}")
if signals:
    df_result = pd.DataFrame(signals, columns=["Ticker", "Signal"])
    st.success(f"✅ {len(df_result)} Signals Found")
    st.dataframe(df_result)
else:
    st.warning("No signals found based on current strategy and stock list.")
