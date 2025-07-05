import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
import math

st.set_page_config(page_title="Live Option Paper Trading", layout="wide")
st.title("📈 NIFTY Option Paper Trading – 3PM Breakout Strategy")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, step=0.5)
offset = st.sidebar.number_input("Breakout/Breakdown Offset", value=100, step=10)
refresh_rate = st.sidebar.slider("Refresh Interval (sec)", 30, 300, 60)

# Session State Init
if "capital" not in st.session_state:
    st.session_state.capital = capital
if "position" not in st.session_state:
    st.session_state.position = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# ---- Fetch NIFTY spot data ----
@st.cache_data(ttl=60)
def fetch_nifty_spot():
    df = yf.download("^NSEI", interval="15m", period="2d", progress=False)
    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    today = datetime.now().date()
    return df[df['Datetime'].dt.date == today]

# ---- Get nearest ITM strike ----
def get_itm_strike(price, direction='CE'):
    step = 50
    base = math.floor(price / step) * step
    return base if direction == 'CE' else base + step

# ---- Generate current week expiry ----
def get_current_expiry():
    today = datetime.today()
    weekday = today.weekday()
    thursday = today + timedelta((3 - weekday) % 7)
    return thursday.strftime('%y%b%d').upper()

# ---- Get Option Symbol ----
def get_option_symbol(strike, typ):
    expiry = get_current_expiry()  # e.g., 24JUL11
    return f"NIFTY{expiry}{strike}{typ}"

# ---- Fetch option premium ----
def fetch_option_ltp(symbol):
    try:
        df = yf.download(symbol + ".NS", interval="1m", period="1d", progress=False)
        if not df.empty:
            return round(df['Close'].iloc[-1], 2)
    except:
        return None
    return None

# ---- Entry logic ----
def enter_option_trade(signal_type, spot_price, strike):
    opt_type = 'CE' if signal_type == 'Breakout' else 'PE'
    symbol = get_option_symbol(strike, opt_type)
    ltp = fetch_option_ltp(symbol)
    if not ltp:
        st.warning(f"❌ Unable to fetch option price for {symbol}")
        return

    risk_amt = st.session_state.capital * (risk_pct / 100)
    qty = int(risk_amt / (ltp * 0.3))  # 30% SL
    sl = ltp * 0.7
    target = ltp + (ltp - sl) * 1.5

    st.session_state.position = {
        "Time": datetime.now(),
        "Signal": signal_type,
        "Strike": strike,
        "Symbol": symbol,
        "Entry": ltp,
        "Qty": qty,
        "SL": round(sl, 2),
        "Target": round(target, 2)
    }

    st.toast(f"✅ Entered {signal_type} - {symbol} at ₹{ltp}", icon="🚀")

# ---- Exit logic ----
def check_exit():
    pos = st.session_state.position
    ltp = fetch_option_ltp(pos['Symbol'])
    if ltp is None:
        return

    entry = pos['Entry']
    sl = pos['SL']
    target = pos['Target']
    qty = pos['Qty']

    if ltp <= sl:
        reason = '🛑 Stop Loss Hit'
        pnl = (ltp - entry) * qty
    elif ltp >= target:
        reason = '🎯 Target Hit'
        pnl = (ltp - entry) * qty
    elif datetime.now().time() >= datetime.strptime("15:25", "%H:%M").time():
        reason = '⏰ Time Exit'
        pnl = (ltp - entry) * qty
    else:
        return  # Still active

    log = {
        "Time": datetime.now(),
        "Symbol": pos['Symbol'],
        "Signal": pos['Signal'],
        "Entry": entry,
        "Exit": ltp,
        "Qty": qty,
        "P&L": round(pnl, 2),
        "Reason": reason
    }
    st.session_state.logs.append(log)
    st.session_state.capital += pnl
    st.session_state.position = None
    st.toast(f"{reason} – Exit at ₹{ltp:.2f} | P&L: ₹{pnl:.2f}", icon="📉" if pnl < 0 else "💰")

# ---- Main live logic ----
df = fetch_nifty_spot()
if df.empty or len(df) < 20:
    st.warning("Waiting for today's candles...")
    st.stop()

three_pm_candle = df[df['Datetime'].dt.time == datetime.strptime("15:00", "%H:%M").time()]
if three_pm_candle.empty:
    st.info("⏳ Waiting for 3PM candle...")
    st.dataframe(df.tail())
else:
    row = three_pm_candle.iloc[0]
    threepm_high = row['High']
    threepm_low = row['Low']
    threepm_close = row['Close']

    st.success(f"3PM High: ₹{threepm_high:.2f} | Low: ₹{threepm_low:.2f} | Close: ₹{threepm_close:.2f}")

    latest = df.iloc[-1]
    now_price = latest['Close']

    if not st.session_state.position:
        if now_price >= threepm_high + offset:
            strike = get_itm_strike(now_price, direction='CE')
            enter_option_trade('Breakout', now_price, strike)
        elif now_price <= threepm_close - offset:
            strike = get_itm_strike(now_price, direction='PE')
            enter_option_trade('Breakdown', now_price, strike)
    else:
        check_exit()
        st.subheader("📌 Current Open Option Position")
        st.write(st.session_state.position)

# ---- Show Logs ----
st.subheader("📋 Trade Log")
log_df = pd.DataFrame(st.session_state.logs)
if not log_df.empty:
    st.dataframe(log_df)
    st.success(f"💼 Capital: ₹{st.session_state.capital:,.2f} | Net P&L: ₹{log_df['P&L'].sum():,.2f}")
else:
    st.info("No trades yet.")

# ---- Auto Refresh ----
st.markdown(f"⏳ Refreshing every {refresh_rate} seconds...")
time.sleep(refresh_rate)
st.experimental_rerun()
