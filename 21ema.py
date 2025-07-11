import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar ---
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# --- Fetch data ---
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

df = load_data(ticker, period, interval)

# --- Validate & Clean ---
if df.empty or 'Close' not in df.columns:
    st.error("⚠️ Data not found or symbol incorrect.")
    st.stop()

df.dropna(inplace=True)
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df.dropna(inplace=True)

# --- Signal Logic ---
close = df['Close'].copy()
ema = df['EMA21'].copy()

# Align and calculate
close, ema = close.align(ema, join="inner", axis=0)
close_prev = close.shift(1)
ema_prev = ema.shift(1)

# Conditions
buy_condition = (close > ema) & (close_prev <= ema_prev)
sell_condition = (close < ema) & (close_prev >= ema_prev)

# Initialize Signal Column
df['Signal'] = 0
df.loc[buy_condition.index, 'Signal'] = 1
df.loc[sell_condition.index, 'Signal'] = -1

# --- Plotting ---
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Candles'))

fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA21'], mode='lines', name='EMA21',
    line=dict(color='orange')))

# Buy Signals
buys = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(
    x=buys.index, y=buys['Close'], mode='markers', name='Buy',
    marker=dict(color='green', symbol='triangle-up', size=10)))

# Sell Signals
sells = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(
    x=sells.index, y=sells['Close'], mode='markers', name='Sell',
    marker=dict(color='red', symbol='triangle-down', size=10)))

fig.update_layout(
    title=f"{ticker} | {interval.upper()} | 21 EMA Strategy",
    xaxis_title="Date", yaxis_title="Price",
    xaxis_rangeslider_visible=False, height=600)

# --- Show Chart ---
st.plotly_chart(fig, use_container_width=True)

# --- Signal Log ---
st.subheader("📋 Signal Log")
signal_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(signal_df)

# --- Download CSV ---
st.download_button(
    label="📥 Download Signal Log",
    data=signal_df.to_csv().encode(),
    file_name=f"{ticker}_21EMA_signals.csv",
    mime="text/csv"
)
