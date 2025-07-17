import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# Detect candlestick patterns
def detect_patterns(df):
    df["Bullish Engulfing"] = (df["Open"].shift(1) > df["Close"].shift(1)) & \
                              (df["Open"] < df["Close"]) & \
                              (df["Open"] < df["Close"].shift(1)) & \
                              (df["Close"] > df["Open"].shift(1))

    df["Bearish Engulfing"] = (df["Open"].shift(1) < df["Close"].shift(1)) & \
                              (df["Open"] > df["Close"]) & \
                              (df["Open"] > df["Close"].shift(1)) & \
                              (df["Close"] < df["Open"].shift(1))

    df["Hammer"] = ((df["High"] - df["Low"]) > 3 * (df["Open"] - df["Close"])) & \
                   ((df["Close"] - df["Low"]) / (.001 + df["High"] - df["Low"]) > 0.6) & \
                   ((df["Open"] - df["Low"]) / (.001 + df["High"] - df["Low"]) > 0.6)

    df["Doji"] = abs(df["Close"] - df["Open"]) <= (0.1 * (df["High"] - df["Low"]))

    return df

# UI
st.title("📊 Candlestick Pattern Finder")
symbol = st.text_input("Enter stock symbol (e.g. RELIANCE.NS, INFY.NS, ^NSEI)", "TCS.NS")
interval = st.selectbox("Interval", ["1d", "1h", "15m"], index=0)
period = st.selectbox("History Period", ["5d", "1mo", "3mo", "6mo"], index=0)

# Load data
df = yf.download(symbol, interval=interval, period=period)
if df.empty:
    st.error("Failed to fetch data.")
    st.stop()

df = df.reset_index()
df = detect_patterns(df)

# Plot
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['Datetime'] if 'Datetime' in df else df['Date'],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candlestick"
))

# Add markers for patterns
for pattern in ["Bullish Engulfing", "Bearish Engulfing", "Hammer", "Doji"]:
    pattern_df = df[df[pattern]]
    fig.add_trace(go.Scatter(
        x=pattern_df["Datetime"] if 'Datetime' in pattern_df else pattern_df["Date"],
        y=pattern_df["High"] * 1.01,
        mode="markers+text",
        marker=dict(size=10, symbol="triangle-up", color="green" if "Bullish" in pattern or pattern=="Hammer" else "red"),
        name=pattern,
        text=[pattern]*len(pattern_df),
        textposition="top center"
    ))

fig.update_layout(title=f"{symbol} - Candle Pattern Detection",
                  xaxis_rangeslider_visible=False,
                  height=600)

st.plotly_chart(fig, use_container_width=True)
