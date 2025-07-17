import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# --- Detect candlestick patterns ---
def detect_patterns(df):
    df["Bullish Engulfing"] = (df["Open"].shift(1) > df["Close"].shift(1)) & \
                              (df["Open"] < df["Close"]) & \
                              (df["Open"] < df["Close"].shift(1)) & \
                              (df["Close"] > df["Open"].shift(1))

    df["Bearish Engulfing"] = (df["Open"].shift(1) < df["Close"].shift(1)) & \
                              (df["Open"] > df["Close"]) & \
                              (df["Open"] > df["Close"].shift(1)) & \
                              (df["Close"] < df["Open"].shift(1))

    df["Hammer"] = ((df["High"] - df["Low"]) > 3 * abs(df["Open"] - df["Close"])) & \
                   ((df["Close"] - df["Low"]) / (.001 + df["High"] - df["Low"]) > 0.6) & \
                   ((df["Open"] - df["Low"]) / (.001 + df["High"] - df["Low"]) > 0.6)

    df["Doji"] = abs(df["Close"] - df["Open"]) <= (0.1 * (df["High"] - df["Low"]))
    return df

# --- Streamlit UI ---
st.title("📊 Candlestick Pattern Finder")

symbol = st.text_input("Enter stock symbol (e.g. RELIANCE.NS, INFY.NS, ^NSEI)", "TCS.NS")
interval = st.selectbox("Interval", ["1d", "1h", "15m"], index=0)
period = st.selectbox("History Period", ["5d", "1mo", "3mo", "6mo"], index=0)

# --- Load data ---
df = yf.download(symbol, interval=interval, period=period)

if df.empty:
    st.error("❌ Could not fetch data. Try different symbol or period.")
    st.stop()

df.reset_index(inplace=True)

# Detect patterns
df = detect_patterns(df)

# Identify datetime column
x_col = "Datetime" if "Datetime" in df.columns else "Date"

# --- Plot candlestick chart ---
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df[x_col],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Candlesticks"
))

# Add pattern markers
for pattern in ["Bullish Engulfing", "Bearish Engulfing", "Hammer", "Doji"]:
    match = df[df[pattern]]
    fig.add_trace(go.Scatter(
        x=match[x_col],
        y=match["High"] * 1.01,
        mode="markers+text",
        text=[pattern] * len(match),
        textposition="top center",
        marker=dict(size=10, symbol="triangle-up", color="green" if "Bullish" in pattern or pattern=="Hammer" else "red"),
        name=pattern
    ))

fig.update_layout(
    title=f"{symbol} - Detected Candle Patterns",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
