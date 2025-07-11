import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config("📈 21 EMA Strategy", layout="wide")
st.title("📊 21 EMA Trading Strategy – Intraday & Swing")

# Sidebar
st.sidebar.header("🔍 Strategy Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS")
mode = st.sidebar.selectbox("Select Mode", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])
    period = st.sidebar.selectbox("Intraday Period", ["1d", "5d", "7d", "10d", "1mo"])
else:
    interval = "1d"
    period = st.sidebar.selectbox("Swing Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# Data loader
@st.cache_data
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data(ticker, period, interval)

if df.empty:
    st.warning("⚠️ No data found for the selected symbol and period.")
    st.stop()

# Calculate EMA21
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

# Initialize Signal column
df['Signal'] = 0

# Safe comparison only after enough rows
if len(df) > 21:
    condition_buy = (df['Close'] > df['EMA21']) & (df['Close'].shift(1) <= df['EMA21'].shift(1))
    condition_sell = (df['Close'] < df['EMA21']) & (df['Close'].shift(1) >= df['EMA21'].shift(1))

    df.loc[condition_buy, 'Signal'] = 1
    df.loc[condition_sell, 'Signal'] = -1

    # --- Plot ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candles'))

    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'],
                             mode='lines', name='EMA21',
                             line=dict(color='orange')))

    # Buy markers
    buys = df[df['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'],
                             mode='markers', name='Buy',
                             marker=dict(color='green', symbol='triangle-up', size=10)))

    # Sell markers
    sells = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'],
                             mode='markers', name='Sell',
                             marker=dict(color='red', symbol='triangle-down', size=10)))

    fig.update_layout(title=f"{ticker} | {interval.upper()} | 21 EMA Strategy",
                      xaxis_title="Date", yaxis_title="Price",
                      xaxis_rangeslider_visible=False, height=600)

    st.plotly_chart(fig, use_container_width=True)

    # Show signal log
    st.subheader("📋 Signal Log")
    signal_df = df[df['Signal'] != 0][['Close', 'EMA21', 'Signal']]
    signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell'})
    st.dataframe(signal_df)

    # CSV Download
    st.download_button("📥 Download Signal CSV",
                       data=signal_df.to_csv().encode(),
                       file_name=f"{ticker}_21EMA_signals.csv",
                       mime="text/csv")
else:
    st.warning("⚠️ Not enough data to calculate EMA. Try increasing period or using a higher timeframe.")
