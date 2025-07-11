import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Streamlit Page Config ---
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# --- Fetch Data ---
@st.cache_data
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

df = load_data(ticker, period, interval)

# --- Strategy Logic ---
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

df['Signal'] = 0
df.loc[(df['Close'] > df['EMA21']) & (df['Close'].shift(1) <= df['EMA21'].shift(1)), 'Signal'] = 1  # Buy
df.loc[(df['Close'] < df['EMA21']) & (df['Close'].shift(1) >= df['EMA21'].shift(1)), 'Signal'] = -1 # Sell

# --- Plotting ---
fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'],
                             name='Price'))

# EMA21
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'],
                         mode='lines', name='EMA21', line=dict(color='orange')))

# Buy Signals
buy_signals = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                         mode='markers', name='Buy',
                         marker=dict(color='green', size=10, symbol='triangle-up')))

# Sell Signals
sell_signals = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                         mode='markers', name='Sell',
                         marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{ticker} | {interval.upper()} | 21 EMA Strategy",
                  xaxis_title="Date", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

# --- Display Chart ---
st.plotly_chart(fig, use_container_width=True)

# --- Signal Table ---
st.subheader("📋 Buy/Sell Signal Log")
signal_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']]
signal_df['Signal'] = signal_df['Signal'].replace({1: 'Buy', -1: 'Sell'})
st.dataframe(signal_df)

# --- Download CSV ---
st.download_button("📥 Download Signal CSV", data=signal_df.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_signals.csv", mime="text/csv")
