import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

st.set_page_config(page_title="NIFTY Options Paper Trading", layout="wide")
st.title("📈 NIFTY Options Full Paper Trade Simulator")

# Sidebar configuration
st.sidebar.header("⚙️ Trade Configuration")
capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)
offset_pts = st.sidebar.number_input("Breakout Offset (pts)", value=0)
target_pts = st.sidebar.number_input("Target Move (pts from 9:30 Close)", value=50)
sl_pct = st.sidebar.slider("Stop Loss % on Option", 1.0, 10.0, 5.0)
analysis_days = st.sidebar.slider("Number of Days to Analyze", min_value=10, max_value=90, value=30)

# Fetch NIFTY 15m data
def load_nifty_data():
    df = yf.download("^NSEI", interval="15m", period=f"{analysis_days+2}d", progress=False)
    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    df = df[df['Datetime'].dt.time >= datetime.strptime("09:15", "%H:%M").time()]
    df = df[df['Datetime'].dt.time <= datetime.strptime("15:30", "%H:%M").time()]
    return df

# Get current week expiry
def get_current_expiry(d):
    weekday = d.weekday()
    thursday = d + timedelta((3 - weekday) % 7)
    return thursday.strftime('%y%b%d').upper()

# Get ATM strike
def get_atm_strike(price):
    return int(round(price / 50) * 50)

# Get option symbol
def get_option_symbol(price, date, opttype):
    expiry = get_current_expiry(date)
    strike = get_atm_strike(price)
    return f"NIFTY{expiry}{strike}{opttype}"

# Get option LTP at 9:30
def get_option_price_at_930(symbol):
    try:
        df = yf.download(symbol + ".NS", interval="1m", period="1d", progress=False)
        df = df.reset_index()
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
        df_930 = df[df['Datetime'].dt.time == datetime.strptime("09:30", "%H:%M").time()]
        if not df_930.empty:
            return round(df_930['Close'].iloc[0], 2)
    except:
        return None
    return None

# Simulate trades
def generate_trade_logs(df):
    df['date'] = df['Datetime'].dt.date
    grouped = df.groupby('date')
    dates = sorted(grouped.groups.keys())
    logs = []

    for i in range(1, len(dates)):
        prev_day = grouped.get_group(dates[i - 1])
        today = grouped.get_group(dates[i])

        # Find 3PM candle of previous day
        candle_3pm = prev_day[prev_day['Datetime'].dt.time == datetime.strptime("15:00", "%H:%M").time()]
        if candle_3pm.empty:
            continue
        candle = candle_3pm.iloc[0]
        open_3pm = candle['Open']
        close_3pm = candle['Close']

        # Find 9:30AM candle of current day
        today_930 = today[today['Datetime'].dt.time == datetime.strptime("09:30", "%H:%M").time()]
        if today_930.empty:
            continue

        current = today_930.iloc[0]
        nifty_930_close = current['Close']

        # Entry criteria
        if nifty_930_close > open_3pm and nifty_930_close > close_3pm:
            symbol = get_option_symbol(nifty_930_close, current['Datetime'], 'CE')
            option_price = get_option_price_at_930(symbol)
            if option_price is None:
                continue
            sl_price = round(option_price * (1 - sl_pct / 100), 2)
            target_price = round(nifty_930_close + target_pts, 2)

            # Check 15m candles till 11:00AM
            trade_candles = today[(today['Datetime'].dt.time >= datetime.strptime("09:30", "%H:%M").time()) &
                                  (today['Datetime'].dt.time <= datetime.strptime("11:00", "%H:%M").time())]

            entry_time = current['Datetime']
            exit_time = None
            exit_reason = '⏰ Time Exit'

            for _, row in trade_candles.iterrows():
                if row['High'] >= target_price:
                    exit_time = row['Datetime']
                    exit_reason = '🎯 Target Hit'
                    break
                elif row['Low'] <= sl_price:
                    exit_time = row['Datetime']
                    exit_reason = '🛑 Stop Loss Hit'
                    break

            final_exit_price = target_price if exit_reason == '🎯 Target Hit' else sl_price if exit_reason == '🛑 Stop Loss Hit' else row['Close']
            pnl = round(final_exit_price - option_price, 2)

            logs.append({
                'Date': dates[i],
                'NIFTY@9:30': nifty_930_close,
                'Option': symbol,
                'Entry Price': option_price,
                'Exit Price': final_exit_price,
                'Entry Time': entry_time.time(),
                'Exit Time': exit_time.time() if exit_time else row['Datetime'].time(),
                'Result': exit_reason,
                'P&L': pnl
            })

    return pd.DataFrame(logs)

# Load and simulate
df_nifty = load_nifty_data()
st.write("Actual Columns:", df_nifty.columns.tolist())

trade_logs = generate_trade_logs(df_nifty)

# Show results
st.subheader("📋 Trade Log")
if not trade_logs.empty:
    st.dataframe(trade_logs)
    st.success(f"✅ Total Trades: {len(trade_logs)} | Net P&L: ₹{trade_logs['P&L'].sum():,.2f}")
else:
    st.warning("No trades met criteria.")


#df_nifty.columns = df_nifty.columns.get_level_values(-1)
st.write("Before Fix - Columns:", df_nifty.columns.tolist())
# ---- Candle Chart Visualization for All Days ----
st.subheader("📊 NIFTY 15-min Candlestick Chart (All Analyzed Days)")
st.write("Actual Columns:", df_nifty.columns.tolist())
plot_data = df_nifty.copy()
#st.write("Actual Columns:", df_nifty.columns.tolist())
#st.write("Preview of Plot Data", plot_data.head())
# Ensure 'Datetime' column exists and is datetime type
if 'Datetime' in plot_data.columns:
    plot_data['Datetime'] = pd.to_datetime(plot_data['Datetime'])

    fig = go.Figure(data=[go.Candlestick(
        x=plot_data['Datetime'],
        open=plot_data['Open'],
        high=plot_data['High'],
        low=plot_data['Low'],
        close=plot_data['Close'],
        name="NIFTY 15m"
    )])

    # Add horizontal lines for 3PM candle Open/Close
    candle_time = datetime.strptime("15:00", "%H:%M").time()
    three_pm_candles = plot_data[plot_data['Datetime'].dt.time == candle_time]

    for _, row in three_pm_candles.iterrows():
        dt = row['Datetime']
        o = row['Open']
        c = row['Close']
        fig.add_hline(y=o, line=dict(dash="dot", color="blue"),
                      annotation_text=f"3PM Open ({dt.date()})", annotation_position="top left")
        fig.add_hline(y=c, line=dict(dash="dot", color="green"),
                      annotation_text=f"3PM Close ({dt.date()})", annotation_position="top right")

    fig.update_layout(
        title=f"NIFTY 15-min Candlestick Chart – Last {analysis_days} Days",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ 'Datetime' column not found in data.")



