import streamlit as st
import yfinance as yf
import pandas as pd
import ta

# Sample NIFTY 50 tickers — you can replace/add more
NSE_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'ITC.NS', 'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS'
]

def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period='3mo', interval='1d', progress=False)
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

def analyze_trend_reversal(df):
    try:
        df['SMA9'] = df['Close'].rolling(window=9).mean()
        df['SMA21'] = df['Close'].rolling(window=21).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

        # Reversal condition: crossover today vs yesterday + RSI confirmation
        if (
            df['SMA9'].iloc[-2] < df['SMA21'].iloc[-2] and 
            df['SMA9'].iloc[-1] > df['SMA21'].iloc[-1] and 
            df['RSI'].iloc[-1] > 40
        ):
            return "Bullish Reversal"
        elif (
            df['SMA9'].iloc[-2] > df['SMA21'].iloc[-2] and 
            df['SMA9'].iloc[-1] < df['SMA21'].iloc[-1] and 
            df['RSI'].iloc[-1] < 60
        ):
            return "Bearish Reversal"
        else:
            return None
    except Exception as e:
        return None

def run_scanner(tickers):
    results = []
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if df is not None:
            signal = analyze_trend_reversal(df)
            if signal:
                results.append({
                    'Stock': ticker.replace(".NS", ""),
                    'Signal': signal,
                    'Last Price': round(df['Close'].iloc[-1], 2),
                    'Date': df.index[-1].strftime('%Y-%m-%d')
                })
    return pd.DataFrame(results)

# Streamlit UI
st.set_page_config(page_title="NSE Trend Reversal Scanner", layout="wide")
st.title("📈 NSE Trend Reversal Scanner")
st.write("This app scans selected NSE stocks for potential **trend reversal signals** using SMA crossover and RSI.")

if st.button("🔍 Run Scan"):
    with st.spinner("Scanning stocks... Please wait."):
        df_signals = run_scanner(NSE_TICKERS)

    if not df_signals.empty:
        st.success("Scan complete! Here are the signals:")
        st.dataframe(df_signals, use_container_width=True)

        csv = df_signals.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="trend_reversal_signals.csv", mime="text/csv")
    else:
        st.info("No trend reversal signals found today.")

st.caption("Powered by yfinance, TA-Lib, and Streamlit | Strategy: SMA(9/21) crossover + RSI filter")
