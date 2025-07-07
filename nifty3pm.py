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
        #df.columns = [col.lower() for col in df.columns]
        df.reset_index(inplace=True)

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip().lower() if isinstance(col, tuple) else col.strip().lower() for col in df.columns]
        else:
            df.columns = [col.strip().lower() for col in df.columns]
        
        # Rename to standard OHLC
        rename_map = {
            'open_^nsei': 'open',
            'high_^nsei': 'high',
            'low_^nsei': 'low',
            'close_^nsei': 'close',
            'volume_^nsei': 'volume'
        }
        df.rename(columns=rename_map, inplace=True)


        # ✅ Filter NSE market hours (9:15 to 15:30)
        df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) &
                (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = get_nifty_15min()
st.write(df.head())
st.write("Columns:", df.columns.tolist())


def plot_nifty_15min_chart(df):
    if df.empty:
        st.warning("No data to display.")
        return

    st.subheader("📉 NIFTY 15-Min Candlestick Chart (Last 3 Days)")
    fig = go.Figure()

    # Plot the candlesticks
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="15-min Candle",
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Extract last 3 dates
    last_dates = df['datetime'].dt.date.unique()[-3:]

    for date in last_dates:
        # Find the exact 3:00 PM candle
        dt_3pm = pd.Timestamp(f"{date} 15:00").tz_localize("Asia/Kolkata")
        candle_3pm = df[df['datetime'] == dt_3pm]

        if not candle_3pm.empty:
            row = candle_3pm.iloc[0]
            vline_time = row['datetime'].to_pydatetime()  # ✅ Convert to native Python datetime

            # Add vertical dotted line at 3PM
            fig.add_vline(
                x=vline_time,
                line_color="blue",
                line_dash="dot",
                annotation_text="3PM",
                annotation_position="top right"
            )

            # Add marker for Open
            fig.add_trace(go.Scatter(
                x=[vline_time],
                y=[row['open']],
                mode="markers+text",
                marker=dict(color="blue", size=10),
                text=["Open"],
                textposition="top center",
                name="3PM Open"
            ))

            # Add marker for Close
            fig.add_trace(go.Scatter(
                x=[vline_time],
                y=[row['close']],
                mode="markers+text",
                marker=dict(color="orange", size=10),
                text=["Close"],
                textposition="bottom center",
                name="3PM Close"
            ))

    # Update layout
    fig.update_layout(
        title="NIFTY 15-Min Chart with 3PM Candle Highlighted",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


plot_nifty_15min_chart(df)
