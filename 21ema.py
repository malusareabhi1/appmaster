import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# --- Data Fetch ---
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    return df

df = load_data(ticker, period, interval)

# --- Validation ---
if df.empty or 'Close' not in df.columns:
    st.error("⚠️ Failed to load data or 'Close' column missing. Please check symbol or timeframe.")
    st.stop()

df.dropna(inplace=True)

# --- EMA Calculation ---
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df.dropna(inplace=True)

# --- Initialize Signals ---
df['Signal'] = 0

# --- Ensure Series and Align ---
close = df['Close'].copy()
ema = df['EMA21'].copy()

# Align both series explicitly
close, ema = close.align(ema, join="inner")

# Shifted series
close_prev = close.shift(1)
ema_prev = ema.shift(1)

# Buy & Sell Conditions
buy_condition = (close > ema) & (close_prev <= ema_prev)
sell_condition = (close < ema) & (close_prev >= ema_prev)

df['Signal'] = 0
df.loc[buy_condition, 'Signal'] = 1
df.loc[sell_condition, 'Signal'] = -1

# --- Plotting ---
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Candles'
))

# EMA21 line
fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA21'], mode='lines',
    name='EMA21', line=dict(color='orange')
))

# Buy signal markers
buy_signals = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals['Close'],
    mode='markers', name='Buy Signal',
    marker=dict(color='green', size=10, symbol='triangle-up')
))

# Sell signal markers
sell_signals = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals['Close'],
    mode='markers', name='Sell Signal',
    marker=dict(color='red', size=10, symbol='triangle-down')
))

fig.update_layout(
    title=f"{ticker} | {interval.upper()} | 21 EMA Strategy",
    xaxis_title="Date", yaxis_title="Price",
    xaxis_rangeslider_visible=False, height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- Signal Log ---
st.subheader("📋 Signal Log")
signal_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(signal_df)

# --- CSV Download ---
st.download_button(
    "📥 Download Signal CSV",
    data=signal_df.to_csv().encode(),
    file_name=f"{ticker}_21ema_signals.csv",
    mime="text/csv"
)
