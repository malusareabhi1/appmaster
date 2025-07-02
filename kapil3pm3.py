# ✅ Updated Streamlit App with User-defined Stop Loss % and Trailing SL Logic

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="NIFTY 15-Min Chart with 3PM Breakout Strategy", layout="wide")

st.title("📈 NIFTY 15-Min Chart – 3PM Breakout/Breakdown Strategy")

st.sidebar.header("Settings")
offset_points = st.sidebar.number_input("Offset Points for Breakout/Breakdown", value=100, step=10)
analysis_days = st.sidebar.slider("Number of Days to Analyze", min_value=10, max_value=90, value=60, step=5)
sl_percent = st.sidebar.slider("Stop Loss % (from Entry Price)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

st.markdown("""
## 📘 Strategy Explanation

This intraday breakout/backtest strategy is based on the NIFTY 15-minute chart.

- 🔼 **Breakout**: If price crosses 3PM High + offset, we enter Long next day. SL is based on % from Entry.
- 🔽 **Breakdown**: If price crosses below 3PM Close and drops offset, we go Short. SL is also % based.
- 🧠 **Trailing SL**: If the position moves in profit, SL trails based on maximum favorable price.

---
""")

@st.cache_data(ttl=3600)
def load_nifty_data(ticker="^NSEI", interval="15m", period="60d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.error("❌ No data returned from yfinance.")
            st.stop()

        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        if not datetime_col:
            st.error("❌ No datetime column found.")
            st.write("📋 Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={datetime_col: 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

        df.columns = [col.lower() for col in df.columns]

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

def generate_trade_logs(df, offset, sl_percent):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
    breakout_logs, breakdown_logs = [], []

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()
        next_day = df[df['datetime'].dt.date == next_day_date].copy()

        entry_breakout = current['high'] + offset
        sl_breakout = entry_breakout * (1 - sl_percent / 100)
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * 1.5

        entry_row = next_day[next_day['high'] >= entry_breakout]
        result, exit_time, exit_price = '❌ No Entry', '-', 0

        if not entry_row.empty:
            entry_time = entry_row.iloc[0]['datetime']
            after_entry = next_day[next_day['datetime'] >= entry_time].copy()
            after_entry['max_price'] = after_entry['high'].cummax()
            after_entry['trailing_sl'] = after_entry['max_price'] * (1 - sl_percent / 100)

            hit_rows = after_entry[(after_entry['low'] <= after_entry['trailing_sl']) | (after_entry['high'] >= target_breakout)]
            if not hit_rows.empty:
                first_hit = hit_rows.iloc[0]
                if first_hit['high'] >= target_breakout:
                    result = '🎯 Target Hit'
                    exit_price = target_breakout
                else:
                    result = '🛑 Trailing SL Hit'
                    exit_price = first_hit['trailing_sl']
                exit_time = first_hit['datetime']
            else:
                result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']
            pnl = round(exit_price - entry_breakout, 2)
        else:
            pnl = 0

        breakout_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            'Entry': round(entry_breakout, 2),
            'SL': round(sl_breakout, 2),
            'Target': round(target_breakout, 2),
            'Exit Time': exit_time if isinstance(exit_time, str) else exit_time.time(),
            'Result': result,
            'raw_pnl': pnl
        })

        entry_breakdown = current['close'] - offset
        sl_breakdown = entry_breakdown * (1 + sl_percent / 100)
        target_breakdown = entry_breakdown - (sl_breakdown - entry_breakdown) * 1.5

        entry_row = next_day[next_day['low'] <= entry_breakdown]
        result, exit_time, exit_price = '❌ No Entry', '-', 0

        if not entry_row.empty:
            entry_time = entry_row.iloc[0]['datetime']
            after_entry = next_day[next_day['datetime'] >= entry_time].copy()
            after_entry['min_price'] = after_entry['low'].cummin()
            after_entry['trailing_sl'] = after_entry['min_price'] * (1 + sl_percent / 100)

            hit_rows = after_entry[(after_entry['high'] >= after_entry['trailing_sl']) | (after_entry['low'] <= target_breakdown)]
            if not hit_rows.empty:
                first_hit = hit_rows.iloc[0]
                if first_hit['low'] <= target_breakdown:
                    result = '🎯 Target Hit'
                    exit_price = target_breakdown
                else:
                    result = '🛑 Trailing SL Hit'
                    exit_price = first_hit['trailing_sl']
                exit_time = first_hit['datetime']
            else:
                result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']
            pnl = round(entry_breakdown - exit_price, 2)
        else:
            pnl = 0

        breakdown_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            'Entry': round(entry_breakdown, 2),
            'SL': round(sl_breakdown, 2),
            'Target': round(target_breakdown, 2),
            'Exit Time': exit_time if isinstance(exit_time, str) else exit_time.time(),
            'Result': result,
            'raw_pnl': pnl
        })

    return pd.DataFrame(breakout_logs), pd.DataFrame(breakdown_logs), df_3pm

def color_pnl_text(pnl):
    if pnl > 0:
        return f"🟢 {pnl}"
    elif pnl < 0:
        return f"🔴 {pnl}"
    else:
        return f"{pnl}"

def show_trade_metrics(df, label):
    total = len(df)
    wins = df[df['Result'] == '🎯 Target Hit'].shape[0]
    pnl = df['raw_pnl'].sum()
    st.success(f"{label}: {total} trades | 🎯 Wins: {wins} | 💰 Total P&L: ₹{pnl:.2f}")

# ----- Load Data and Generate Trades -----
df = load_nifty_data(period=f"{analysis_days}d")
if df.empty:
    st.error("No data loaded.")
    st.stop()

df = filter_last_n_days(df, analysis_days)
rename_map = {col: col.lower() for col in df.columns}
df.rename(columns=rename_map, inplace=True)

breakout_df, breakdown_df, df_3pm = generate_trade_logs(df, offset_points, sl_percent)

fig = plot_candlestick_chart(df, df_3pm)

st.subheader("🕯️ NIFTY Candlestick Chart")
st.plotly_chart(fig, use_container_width=True)

# ----- Breakout Logs -----
filtered_breakout_df = breakout_df[breakout_df['Result'] != '❌ No Entry'].copy()
filtered_breakout_df['P&L'] = filtered_breakout_df['raw_pnl'].apply(color_pnl_text)

st.subheader("📘 Breakout Logs")
st.dataframe(filtered_breakout_df)
show_trade_metrics(filtered_breakout_df, "Breakout Trades")

st.download_button("📥 Download Breakout Log", filtered_breakout_df.to_csv(index=False), file_name="breakout_log.csv")

# ----- Breakdown Logs -----
filtered_breakdown_df = breakdown_df[breakdown_df['Result'] != '❌ No Entry'].copy()
filtered_breakdown_df['P&L'] = filtered_breakdown_df['raw_pnl'].apply(color_pnl_text)

st.subheader("📕 Breakdown Logs")
st.dataframe(filtered_breakdown_df)
show_trade_metrics(filtered_breakdown_df, "Breakdown Trades")

st.download_button("📥 Download Breakdown Log", filtered_breakdown_df.to_csv(index=False), file_name="breakdown_log.csv")
