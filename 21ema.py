import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page Setup ---
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar ---
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Period", ["1d", "5d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"])

# --- Load Data ---
@st.cache_data
def load_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

df = load_data(ticker, period, interval)

# --- Validation ---
if df.empty or 'Close' not in df.columns:
    st.error("❌ No data found. Check symbol or timeframe.")
    st.stop()

# --- Strategy Calculation ---
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['Signal'] = 0

# Use iterrows for safety
previous_row = None
for i, row in df.iterrows():
    if previous_row is not None:
        if row['Close'] > row['EMA21'] and previous_row['Close'] <= previous_row['EMA21']:
            df.at[i, 'Signal'] = 1  # Buy
        elif row['Close'] < row['EMA21'] and previous_row['Close'] >= previous_row['EMA21']:
            df.at[i, 'Signal'] = -1  # Sell
    previous_row = row

# Drop any NaNs
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
buys = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
                         name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))

# Sell markers
sells = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
                         name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{ticker} | {interval} | 21 EMA Strategy",
                  xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

st.plotly_chart(fig, use_container_width=True)

# --- Signal Table ---
st.subheader("📋 Signal Log")
signal_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(signal_df)

# --- Download CSV ---
st.download_button("📥 Download Signal CSV",
                   data=signal_df.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_Signals.csv",
                   mime="text/csv")

