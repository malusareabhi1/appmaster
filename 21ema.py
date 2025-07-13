import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go






# Page Setup
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# Sidebar Inputs
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.radio("Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Data Period", ["1d", "5d", "7d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "5y"])

# Load data
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    return df

df = load_data(ticker, period, interval)

if df.empty:
    st.error("❌ No data found. Check the stock symbol or internet.")
    st.stop()

# Strategy logic
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['Close_prev'] = df['Close'].shift(1)
df['EMA21_prev'] = df['EMA21'].shift(1)
df['Signal'] = 0

# Buy: close > ema and previous close <= previous ema
df.loc[(df['Close'] > df['EMA21']) & (df['Close_prev'] <= df['EMA21_prev']), 'Signal'] = 1
# Sell: close < ema and previous close >= previous ema
df.loc[(df['Close'] < df['EMA21']) & (df['Close_prev'] >= df['EMA21_prev']), 'Signal'] = -1

df.dropna(inplace=True)

# Plotting
fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name='Candles'))

# EMA line
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21',
                         line=dict(color='orange')))

# Buy markers
buy = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='Buy',
                         marker=dict(symbol='triangle-up', color='green', size=10)))

# Sell markers
sell = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='Sell',
                         marker=dict(symbol='triangle-down', color='red', size=10)))

fig.update_layout(title=f"{ticker} | {interval} Chart | 21 EMA Strategy",
                  xaxis_title="Date", yaxis_title="Price",
                  xaxis_rangeslider_visible=False,
                  height=600)

# Show chart
st.plotly_chart(fig, use_container_width=True)

# Signal log
st.subheader("📋 Trade Signals")
log_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
log_df['Signal'] = log_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(log_df)

# Download
st.download_button("📥 Download Signal Log", log_df.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_signals.csv",
                   mime="text/csv")
