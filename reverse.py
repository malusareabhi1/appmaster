import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from datetime import date, timedelta

st.set_page_config(page_title="Stock Trend Reversal Detector", layout="wide")

st.title("📈 Trend Reversal Detector for Multiple Stocks")

# Input Tickers
ticker_input = st.text_area("Enter Stock Tickers (comma-separated, NSE stocks use .NS)", value="RELIANCE.NS, INFY.NS, TCS.NS")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date = st.date_input("End Date", value=date.today())

reversal_signals = []

def detect_reversal(df):
    df['SMA9'] = SMAIndicator(df['Close'], window=9).sma_indicator()
    df['SMA21'] = SMAIndicator(df['Close'], window=21).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

    df['Signal'] = None
    for i in range(1, len(df)):
        # Bullish Reversal: SMA9 crosses above SMA21 and RSI rising from <40
        if (df['SMA9'].iloc[i-1] < df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] > df['SMA21'].iloc[i]) and df['RSI'].iloc[i] > 40:
            df.loc[df.index[i], 'Signal'] = 'Bullish Reversal'
        # Bearish Reversal: SMA9 crosses below SMA21 and RSI falling from >60
        elif (df['SMA9'].iloc[i-1] > df['SMA21'].iloc[i-1]) and (df['SMA9'].iloc[i] < df['SMA21'].iloc[i]) and df['RSI'].iloc[i] < 60:
            df.loc[df.index[i], 'Signal'] = 'Bearish Reversal'
    return df

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            continue
        df = detect_reversal(df)

        last_signal = df[df['Signal'].notnull()].iloc[-1] if not df[df['Signal'].notnull()].empty else None

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

            reversal_dates = df[df['Signal'].notnull()]
            fig.add_trace(go.Scatter(x=reversal_dates.index, y=reversal_dates['Close'],
                                     mode='markers', marker=dict(color='red', size=10),
                                     name='Reversal Signal'))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No recent reversal signals for {ticker}")
    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")

# Summary Table
if reversal_signals:
    st.markdown("### 🧾 Summary of Detected Reversals")
    st.dataframe(pd.DataFrame(reversal_signals))
else:
    st.warning("No trend reversal signals found in the selected date range.")

