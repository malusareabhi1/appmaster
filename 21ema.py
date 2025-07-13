import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Setup ---
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar Inputs ---
st.sidebar.header("📌 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "5y"])

# --- Fetch Data ---
@st.cache_data
def load_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

df = load_data(ticker, period, interval)

# --- Validation ---
if df.empty:
    st.error("⚠️ No data found. Please check the symbol or timeframe.")
    st.stop()

# --- Strategy Logic ---
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['Close_prev'] = df['Close'].shift(1)
df['EMA21_prev'] = df['EMA21'].shift(1)

df['Signal'] = 0
df.loc[(df['Close'] > df['EMA21']) & (df['Close_prev'] <= df['EMA21_prev']), 'Signal'] = 1  # Buy
df.loc[(df['Close'] < df['EMA21']) & (df['Close_prev'] >= df['EMA21_prev']), 'Signal'] = -1  # Sell

df.dropna(inplace=True)

# --- Plotting ---
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'],
                             name='Candles'))

# EMA Line
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'],
                         mode='lines', name='EMA21',
                         line=dict(color='orange')))

# Buy Markers
buy_signals = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                         mode='markers', name='Buy',
                         marker=dict(color='green', symbol='triangle-up', size=10)))

# Sell Markers
sell_signals = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                         mode='markers', name='Sell',
                         marker=dict(color='red', symbol='triangle-down', size=10)))

fig.update_layout(title=f"{ticker} | {interval} | 21 EMA Strategy",
                  xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

# --- Show Chart ---
st.plotly_chart(fig, use_container_width=True)

# --- Signal Log Table ---
st.subheader("📋 Signal Log")
log_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
log_df['Signal'] = log_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(log_df)

# --- Download ---
st.download_button("📥 Download Signal CSV",
                   data=log_df.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_signals.csv",
                   mime="text/csv")
