# live_paper_trading.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Live 3PM Breakout Paper Trading", layout="wide")
st.title("📡 Live Paper Trading – NIFTY 3PM Breakout/Breakdown Strategy")

# Sidebar inputs
st.sidebar.header("⚙️ Settings")
offset = st.sidebar.number_input("Breakout/Breakdown Offset (Points)", value=100, step=10)
capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, step=0.5)
ticker = st.sidebar.selectbox("Symbol", ["^NSEI"], index=0)
refresh_rate = st.sidebar.slider("Refresh Interval (sec)", 30, 300, 60, step=10)

# Session state
if "position" not in st.session_state:
    st.session_state.position = None
if "capital" not in st.session_state:
    st.session_state.capital = capital
if "logs" not in st.session_state:
    st.session_state.logs = []

# Fetch today's 15m candles
@st.cache_data(ttl=60)
def fetch_today_data():
    today = datetime.now().date()
    df = yf.download(ticker, interval="15m", period="2d", progress=False)
    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    df = df[df['Datetime'].dt.date == today]
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Paper trade logic
def check_trade(df, three_pm_high, three_pm_close, three_pm_low):
    last_candle = df.iloc[-1]
    now = last_candle['Datetime']
    price = last_candle['Close']

    # Check if already in trade
    if st.session_state.position:
        pos = st.session_state.position
        entry = pos['Entry']
        sl = pos['SL']
        target = pos['Target']
        qty = pos['Qty']

        if price >= target:
            pnl = round((target - entry) * qty, 2)
            msg = f"🎯 Target Hit at {price}"
            exit_pos(now, price, pnl, msg)
        elif price <= sl:
            pnl = round((sl - entry) * qty, 2)
            msg = f"🛑 Stop Loss Hit at {price}"
            exit_pos(now, price, pnl, msg)
        elif now.time() >= datetime.strptime("15:25", "%H:%M").time():
            pnl = round((price - entry) * qty, 2)
            msg = f"⏰ Time Exit at {price}"
            exit_pos(now, price, pnl, msg)
    else:
        # Try to enter trade
        risk_amt = st.session_state.capital * (risk_pct / 100)

        # Breakout Entry
        if price >= three_pm_high + offset:
            sl = three_pm_low
            risk_per_unit = abs((three_pm_high + offset) - sl)
            qty = int(risk_amt / risk_per_unit) if risk_per_unit else 0
            entry = three_pm_high + offset
            target = entry + (entry - sl) * 1.5
            enter_pos(now, entry, sl, target, qty, "Breakout")

        # Breakdown Entry
        elif price <= three_pm_close - offset:
            sl = three_pm_high
            entry = three_pm_close - offset
            risk_per_unit = abs(sl - entry)
            qty = int(risk_amt / risk_per_unit) if risk_per_unit else 0
            target = entry - (sl - entry) * 1.5
            enter_pos(now, entry, sl, target, qty, "Breakdown")

# Entry
def enter_pos(time, entry, sl, target, qty, typ):
    st.session_state.position = {
        "Time": time,
        "Entry": entry,
        "SL": sl,
        "Target": target,
        "Qty": qty,
        "Type": typ
    }
    st.toast(f"✅ Entered {typ} at ₹{entry:.2f} | SL: ₹{sl:.2f} | Target: ₹{target:.2f} | Qty: {qty}", icon="🚀")

# Exit
def exit_pos(time, exit_price, pnl, reason):
    pos = st.session_state.position
    log = {
        "Type": pos["Type"],
        "Entry Time": pos["Time"],
        "Exit Time": time,
        "Entry": pos["Entry"],
        "Exit": exit_price,
        "SL": pos["SL"],
        "Target": pos["Target"],
        "Qty": pos["Qty"],
        "P&L": pnl,
        "Reason": reason
    }
    st.session_state.logs.append(log)
    st.session_state.capital += pnl
    st.session_state.position = None
    st.toast(f"{reason} | P&L: ₹{pnl:.2f}", icon="📉" if pnl < 0 else "💰")

# Fetch data
df = fetch_today_data()

if df.empty or len(df) < 20:
    st.warning("Waiting for enough candles to begin live paper trading...")
    st.stop()

# Find 3PM candle
three_pm_candle = df[df['Datetime'].dt.time == datetime.strptime("15:00", "%H:%M").time()]
if three_pm_candle.empty:
    st.warning("🔔 Waiting for today's 3:00 PM candle to form...")
    st.dataframe(df.tail(5))
else:
    row = three_pm_candle.iloc[0]
    st.success("✅ 3PM Candle Captured")
    st.write(f"🕒 3PM High: ₹{row['High']:.2f} | Low: ₹{row['Low']:.2f} | Close: ₹{row['Close']:.2f}")

    # Monitor current candle
    check_trade(df, row['High'], row['Close'], row['Low'])

    # Show current status
    if st.session_state.position:
        pos = st.session_state.position
        st.subheader("📌 Current Open Trade")
        st.write(pos)

# Logs
st.subheader("📋 Trade Log")
log_df = pd.DataFrame(st.session_state.logs)
if not log_df.empty:
    st.dataframe(log_df)
    st.success(f"💼 Capital: ₹{st.session_state.capital:,.2f} | Net P&L: ₹{log_df['P&L'].sum():,.2f}")
else:
    st.info("No trades yet.")

# Auto refresh every X seconds
st.markdown(f"⏳ Auto-refreshing every {refresh_rate} seconds...")
time.sleep(refresh_rate)
st.experimental_rerun()
