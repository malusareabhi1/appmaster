import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Streamlit UI ---
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

st.sidebar.header("🔍 Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Select Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Period", ["1d", "5d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"])

# --- Load data ---
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

df = load_data(ticker, period, interval)

# --- Check Data ---
if df.empty:
    st.error("❌ No data found. Check symbol or timeframe.")
    st.stop()

# --- Calculate EMA and Signals ---
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['Close_prev'] = df['Close'].shift(1)
df['EMA21_prev'] = df['EMA21'].shift(1)

def signal(row):
    if row['Close'] > row['EMA21'] and row['Close_prev'] <= row['EMA21_prev']:
        return 1  # Buy
    elif row['Close'] < row['EMA21'] and row['Close_prev'] >= row['EMA21_prev']:
        return -1  # Sell
    return 0

df['Signal'] = df.apply(signal, axis=1)
df.dropna(inplace=True)

# --- Plotting ---
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name='Candles'))

# EMA21 line
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21',
                         line=dict(color='orange')))

# Buy markers
buy = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='Buy',
                         marker=dict(color='green', symbol='triangle-up', size=10)))

# Sell markers
sell = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='Sell',
                         marker=dict(color='red', symbol='triangle-down', size=10)))

fig.update_layout(title=f"{ticker} | {interval} | 21 EMA Strategy",
                  xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

st.plotly_chart(fig, use_container_width=True)

# --- Signal Table ---
st.subheader("📋 Signal Log")
log = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
log['Signal'] = log['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(log)

# --- Download Button ---
st.download_button("📥 Download Signal CSV",
                   data=log.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_Signals.csv",
                   mime="text/csv")
