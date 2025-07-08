# ✅ Cleaned & Fixed Streamlit App for NIFTY 3PM Breakout Strategy with Option Chain Support

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="NIFTY 15-Min Chart with 3PM Breakout Strategy", layout="wide")

st.title("\U0001F4C8 NIFTY 15-Min Chart – 3PM Breakout/Breakdown Strategy")

st.sidebar.header("Settings")
offset_points = st.sidebar.number_input("Offset Points for Breakout/Breakdown", value=100, step=10)
analysis_days = st.sidebar.slider("Number of Days to Analyze", min_value=3, max_value=60, value=5, step=1)

# ✅ Load Data from Yahoo Finance
@st.cache_data(ttl=3600)
def load_nifty_data(ticker="^NSEI", interval="15m", period="3d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.error("❌ No data returned from yfinance.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        #df.columns = [col.lower() for col in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        
        df.columns = [col.lower() for col in df.columns]

        #df.rename(columns={'datetime': 'datetime'}, inplace=True)
        # Try to detect the datetime column name automatically
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        
        if not datetime_col:
            st.error("❌ No datetime column found after reset_index().")
            st.write("📋 Available columns:", df.columns.tolist())
            return pd.DataFrame()
        
        df.rename(columns={datetime_col: 'datetime'}, inplace=True)

        
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
        df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) &
                (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_last_n_days(df, n_days):
    df['date'] = df['datetime'].dt.date
    last_days = sorted(df['date'].unique())[-n_days:]
    return df[df['date'].isin(last_days)].copy()

# ✅ Option Chain (Safe Parsing)
def get_nifty_option_chain_simple():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com/option-chain"
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        response = session.get(url, timeout=5)
        data = response.json()
        records = data['records']['data']

        rows = []
        for rec in records:
            strike = rec.get('strikePrice')
            ce = rec.get('CE', {})
            pe = rec.get('PE', {})
            rows.append({
                'strikePrice': strike,
                'CE_LTP': ce.get('lastPrice'),
                'CE_IV': ce.get('impliedVolatility'),
                'PE_LTP': pe.get('lastPrice'),
                'PE_IV': pe.get('impliedVolatility')
            })
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"❌ Error fetching option chain: {e}")
        return pd.DataFrame()

# ✅ Safe Strike Extract
@st.cache_data(ttl=300)
def get_strike_list(option_chain_df):
    if 'strikePrice' not in option_chain_df.columns:
        st.error("❌ 'strikePrice' column missing in option chain data.")
        st.write("Columns available:", option_chain_df.columns.tolist())
        return []
    return sorted(option_chain_df['strikePrice'].dropna().unique().tolist())

def find_nearest_strike(strikes, spot_price):
    return min(strikes, key=lambda x: abs(x - spot_price)) if strikes else None

# ✅ Main Trade Logic
def generate_trade_logs(df, offset, option_chain_df):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
    strikes = get_strike_list(option_chain_df)
    if not strikes:
        return pd.DataFrame(), pd.DataFrame()

    breakout_logs = []
    breakdown_logs = []

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()

        threepm_high = current['high']
        threepm_close = current['close']
        spot = current['close']

        entry_breakout = threepm_high + offset
        entry_breakdown = threepm_close - offset

        strike = find_nearest_strike(strikes, spot)
        ce_symbol = f"NIFTY11JUL24{strike}CE"
        pe_symbol = f"NIFTY11JUL24{strike}PE"

        # Breakout Log
        breakout_logs.append({
            '3PM Date': current['datetime'].date(),
            'Spot': round(spot, 2),
            '3PM High': round(threepm_high, 2),
            'Breakout Entry': round(entry_breakout, 2),
            'Option Buy': ce_symbol,
            'Type': 'CALL'
        })

        # Breakdown Log
        breakdown_logs.append({
            '3PM Date': current['datetime'].date(),
            'Spot': round(spot, 2),
            '3PM Close': round(threepm_close, 2),
            'Breakdown Entry': round(entry_breakdown, 2),
            'Option Buy': pe_symbol,
            'Type': 'PUT'
        })

    return pd.DataFrame(breakout_logs), pd.DataFrame(breakdown_logs)



# ✅ Step 1: Load data
df = load_nifty_data(period=f"{analysis_days}d")

# ✅ Step 2: Exit if empty
if df.empty:
    st.stop()

# ✅ Step 3: Rename ^nsei columns
df = df.rename(columns={
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})

# ✅ Step 4: Filter by last N days
df = filter_last_n_days(df, analysis_days)

# ✅ Step 5: Generate trades
trade_log_df, breakdown_df = generate_trade_logs(df, offset_points, option_chain_df)


# ✅ Display results
st.subheader("\U0001F4C4 Breakout Trade Log (CALLS)")
st.dataframe(trade_log_df)

st.subheader("\U0001F4C4 Breakdown Trade Log (PUTS)")
st.dataframe(breakdown_df)

if not option_chain_df.empty:
    st.subheader("\U0001F4CA NSE Option Chain Snapshot")
    st.dataframe(option_chain_df.head(20))
