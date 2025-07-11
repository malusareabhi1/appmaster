import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page setup
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# Sidebar
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# Load data
@st.cache_data
def load_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

df = load_data(ticker, period, interval)

# Validate
if df.empty or 'Close' not in df.columns:
    st.error("⚠️ No data found or symbol incorrect.")
    st.stop()

# Clean
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['Close_prev'] = df['Close'].shift(1)
df['EMA21_prev'] = df['EMA21'].shift(1)
df.dropna(inplace=True)

# Signal logic using apply
def get_signal(row):
    if row['Close'] > row['EMA21'] and row['Close_prev'] <= row['EMA21_prev']:
        return 1  # Buy
    elif row['Close'] < row['EMA21'] and row['Close_prev'] >= row['EMA21_prev']:
        return -1  # Sell
    else:
        return 0

df['Signal'] = df.apply(get_signal, axis=1)

# Plot
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'],
                             name='Candles'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='orange')))

# Buy signals
buys = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
                         name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)))

# Sell signals
sells = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
                         name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)))

fig.update_layout(title=f"{ticker} | {interval} | 21 EMA Strategy",
                  xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

st.plotly_chart(fig, use_container_width=True)

# Signal log
st.subheader("📋 Signal Log")
log_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
log_df['Signal'] = log_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(log_df)

# Download button
st.download_button("📥 Download Signal CSV",
                   data=log_df.to_csv().encode(),
                   file_name=f"{ticker}_21ema_signals.csv",
                   mime="text/csv")
