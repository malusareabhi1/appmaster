import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from datetime import date, timedelta

st.set_page_config(page_title="Trend Reversal Detector", layout="wide")
st.title("🔁 Trend Reversal Detector for NSE Stocks")

# User input
ticker_input = st.text_area("Enter NSE stock symbols (comma-separated)", "RELIANCE.NS, INFY.NS, TCS.NS")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date = st.date_input("End Date", value=date.today())

reversal_signals = []

# Helper function
def detect_reversal(df):
    # Clean Close column
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    close = df['Close']

    df['SMA9'] = SMAIndicator(close=close, window=9).sma_indicator()
    df['SMA21'] = SMAIndicator(close=close, window=21).sma_indicator()
    df['RSI'] = RSIIndicator(close=close, window=14).rsi()
    df['Signal'] = None

    for i in range(1, len(df)):
        if (df['SMA9'].iloc[i-1] < df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] > df['SMA21'].iloc[i]) and df['RSI'].iloc[i] > 40:
            df.loc[df.index[i], 'Signal'] = 'Bullish Reversal'
        elif (df['SMA9'].iloc[i-1] > df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] < df['SMA21'].iloc[i]) and df['RSI'].iloc[i] < 60:
            df.loc[df.index[i], 'Signal'] = 'Bearish Reversal'

    return df

# Process stocks
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data for {ticker}")
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
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA9'], name='SMA9'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA21'], name='SMA21'))
            fig.add_trace(go.Scatter(
                x=signal_df.index,
                y=signal_df['Close'],
                mode='markers',
                name='Signal',
                marker=dict(color='red', size=10, symbol='x')
            ))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No reversal signal detected for {ticker}")

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")

# Show table
if reversal_signals:
    st.markdown("### 📊 Reversal Summary")
    st.dataframe(pd.DataFrame(reversal_signals))
else:
    st.warning("No trend reversal signals found.")
