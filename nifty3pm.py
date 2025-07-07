import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="NIFTY 3-Day Chart with 3PM Candle", layout="wide")
st.title("📊 NIFTY 3-Day 15-Min Chart with 3PM Candle Highlight")

# --- Download NIFTY 15-min data for last 3 days ---
@st.cache_data(ttl=3600)
def get_nifty_15min(ticker="^NSEI", interval="15m", period="3d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.error("❌ No data returned from yfinance.")
            st.stop()

        df.reset_index(inplace=True)

        # ✅ Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # ✅ Find datetime column automatically
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)

        if not datetime_col:
            st.error("❌ No datetime column found after reset_index().")
            st.write("📋 Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={datetime_col: 'datetime'}, inplace=True)

        # ✅ Convert to datetime and localize
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

        # ✅ Now lowercase column names
        df.columns = [col.lower() for col in df.columns]

        # ✅ Filter NSE market hours (9:15 to 15:30)
        df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) &
                (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = get_nifty_15min()
st.write(df.head())
    # Flatten column names
#if isinstance(df.columns, pd.MultiIndex):
    #df.columns = df.columns.droplevel(0)  # drop 'NIFTYBEES.NS'
#st.write(df.head())
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
