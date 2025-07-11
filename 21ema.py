import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# --- Sidebar ---
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
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
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

df = load_data(ticker, period, interval)

# --- Validation ---
if df.empty or 'Close' not in df.columns:
    st.error("⚠️ No data found or incorrect symbol.")
    st.stop()

df = df.dropna()
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df = df.dropna()

# --- Signal Logic ---
close = df['Close'].astype(float)
ema21 = df['EMA21'].astype(float)
close_prev = close.shift(1)
ema_prev = ema21.shift(1)

# Safe Signal Calculation
signal = []
for c, e, cp, ep in zip(close, ema21, close_prev, ema_prev):
    if pd.isna(cp) or pd.isna(ep):
        signal.append(0)
    elif c > e and cp <= ep:
        signal.append(1)  # Buy
    elif c < e and cp >= ep:
        signal.append(-1)  # Sell
    else:
        signal.append(0)
df['Signal'] = signal

# --- Plotting ---
fig = go.Figure()

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name='Price'))

fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines',
                         name='EMA21', line=dict(color='orange')))

# Buy markers
buy_df = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy_df.index, y=buy_df['Close'], mode='markers',
                         name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))

# Sell markers
sell_df = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell_df.index, y=sell_df['Close'], mode='markers',
                         name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{ticker} | {interval} | 21 EMA Strategy",
                  xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, height=600)

# --- Show Output ---
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Buy/Sell Signal Log")
log_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']].copy()
log_df['Signal'] = log_df['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(log_df)

# --- Download ---
st.download_button("📥 Download Signal Log",
                   data=log_df.to_csv().encode(),
                   file_name=f"{ticker}_21EMA_signals.csv",
                   mime="text/csv")
