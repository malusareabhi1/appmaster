import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="NIFTY 3-Day Chart with 3PM Candle", layout="wide")
st.title("📊 NIFTY 3-Day 15-Min Chart with 3PM Candle Highlight")

# --- Download NIFTY 15-min data for last 3 days ---
@st.cache_data(ttl=300)
def get_nifty_15min():
    df = yf.download("^NSEI", interval="15m", period="5d", progress=False)
   

    df = df[df.index.time >= datetime.strptime("09:15", "%H:%M").time()]
    df = df[df.index.time <= datetime.strptime("15:30", "%H:%M").time()]
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['Date'] = df['Datetime'].dt.date
    df['Time'] = df['Datetime'].dt.time
    return df

df = get_nifty_15min()
st.write(df.head())
    # Flatten column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(0)  # drop 'NIFTYBEES.NS'
st.write(df.head())
# Filter only last 3 trading days
last_dates = df['Date'].unique()[-3:]
df = df[df['Date'].isin(last_dates)]

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['Datetime'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="NIFTY",
    increasing_line_color='green',
    decreasing_line_color='red'
))

# --- Highlight 3PM candles ---
for date in last_dates:
    candle_3pm = df[(df['Date'] == date) & (df['Time'] == datetime.strptime("15:00", "%H:%M").time())]
    if not candle_3pm.empty:
        row = candle_3pm.iloc[0]
        # Vertical line at 3PM
        fig.add_vline(x=row['Datetime'], line_color="blue", line_dash="dot", annotation_text="3PM", annotation_position="top right")
        # Open/Close markers
        fig.add_trace(go.Scatter(
            x=[row['Datetime']],
            y=[row['Open']],
            mode="markers+text",
            text=["Open"],
            textposition="top center",
            marker=dict(color="blue", size=10),
            name="3PM Open"
        ))
        fig.add_trace(go.Scatter(
            x=[row['Datetime']],
            y=[row['Close']],
            mode="markers+text",
            text=["Close"],
            textposition="bottom center",
            marker=dict(color="orange", size=10),
            name="3PM Close"
        ))

fig.update_layout(
    title="NIFTY 15-Min Chart with 3PM Candle Marked",
    xaxis_title="Datetime",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
