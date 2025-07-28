import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from datetime import date, timedelta
import numpy as np

st.set_page_config(page_title="Trend Reversal Detector", layout="wide")
st.title("🔁 Trend Reversal Detector for NSE Stocks")

# Get stock list
ticker_input = st.text_area("Enter NSE stock symbols (comma-separated)", "RELIANCE.NS, INFY.NS, TCS.NS")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Date selection
start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date = st.date_input("End Date", value=date.today())

reversal_signals = []

def detect_reversal(df):
    # Convert Close to proper 1D numeric Series
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    close_series = df['Close']

    # Ensure it's a proper numeric 1D Series
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series(close_series.flatten(), index=df.index)

    if not pd.api.types.is_numeric_dtype(close_series):
        raise ValueError("Close prices must be numeric.")

    # Calculate Indicators
    df['SMA9'] = SMAIndicator(close=close_series, window=9).sma_indicator()
    df['SMA21'] = SMAIndicator(close=close_series, window=21).sma_indicator()
    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()

    df['Signal'] = None

    for i in range(1, len(df)):
        if (
            df['SMA9'].iloc[i-1] < df['SMA21'].iloc[i-1] and
            df['SMA9'].iloc[i] > df['SMA21'].iloc[i] and
            df['RSI'].iloc[i] > 40
        ):
            df.loc[df.index[i], 'Signal'] = 'Bullish Reversal'
        elif (
            df['SMA9'].iloc[i-1] > df['SMA21'].iloc[i-1] and
            df['SMA9'].iloc[i] < df['SMA21'].iloc[i] and
            df['RSI'].iloc[i] < 60
        ):
            df.loc[df.index[i], 'Signal'] = 'Bearish Reversal'

    return df

# Process each stock
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for {ticker}")
            continue

        df = detect_reversal(df)

        signal_df = df[df['Signal'].notnull()]
        last_signal = signal_df.iloc[-1] if not signal_df.empty else None

        if last_signal is not None:
            reversal_signals.append({
                "Stock": ticker,
                "Date": last_signal.name.date(),
                "Close": round(last_signal['Close'], 2),
                "Signal": last_signal['Signal']
            })

            st.subheader(f"{ticker} - {last_signal['Signal']} on {last_signal.name.date()}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA9'], name='SMA9'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA21'], name='SMA21'))
            fig.add_trace(go.Scatter(
                x=signal_df.index,
                y=signal_df['Close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Signal'
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No reversal signal found for {ticker}")

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")

# Final results table
if reversal_signals:
    st.markdown("### 📊 Reversal Summary")
    st.dataframe(pd.DataFrame(reversal_signals))
else:
    st.warning("No reversal signals found.")
