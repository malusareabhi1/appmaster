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

        # ✅ If datetime is in index, move it to column
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)

        # ✅ Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # ✅ Find datetime column automatically
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        if not datetime_col:
            st.error("❌ No datetime column found.")
            st.write("📋 Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={datetime_col: 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # ✅ Localize to India time if needed
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

        # ✅ Lowercase all column names
        df.columns = [col.lower() for col in df.columns]

        # ✅ Filter NSE market hours (9:15 to 15:30)
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
        threepm_high, threepm_close = current['high'], current['close']

        # Long Entry Setup
        entry_breakout = threepm_high + offset
        sl_breakout = entry_breakout * (1 - sl_percent / 100)
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * 1.5

        next_day = df[df['datetime'].dt.date == next_day_date].copy()
        entry_row = next_day[next_day['high'] >= entry_breakout]

        #result, exit_time, exit_price = '❌ No Entry', '-', '-', 0
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
            'P&L': pnl
        })

    return pd.DataFrame(breakout_logs), df_3pm

def plot_candlestick_chart(df, df_3pm):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="NIFTY",
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Horizontal rays from 3PM candle till next day's 3PM
    for i in range(len(df_3pm) - 1):
        ref_dt = df_3pm.iloc[i]['datetime']
        next_day_dt = df_3pm.iloc[i + 1]['datetime']
        high = df_3pm.iloc[i]['high']
        low = df_3pm.iloc[i]['low']

        # Time window from next day 9:15 to next day 15:00
        next_day_data = df[(df['datetime'] > ref_dt) & (df['datetime'] <= next_day_dt)]

        if not next_day_data.empty:
            x_start = next_day_data['datetime'].min()
            x_end = next_day_data['datetime'].max()

            # Horizontal line for 3PM High
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[high, high],
                mode='lines',
                line=dict(color='orange', width=1, dash='dot'),
                name='3PM High Ray',
                showlegend=False
            ))

            # Horizontal line for 3PM Low
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[low, low],
                mode='lines',
                line=dict(color='cyan', width=1, dash='dot'),
                name='3PM Low Ray',
                showlegend=False
            ))

    fig.update_layout(
        title=f"NIFTY 15-Min Chart (Last {analysis_days} Trading Days)",
        xaxis_title="DateTime (IST)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),         # hide weekends
                dict(bounds=[16, 9.15], pattern="hour")  # hide non-trading hours
            ],
            showgrid=False
        ),
        yaxis=dict(showgrid=True),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=600
    )
    return fig


def show_trade_metrics(df, label):
    total = len(df)
    wins = df[df['Result'] == '🎯 Target Hit'].shape[0]
    pnl = df['P&L'].sum()
    st.success(f"{label}: {total} trades | 🎯 Wins: {wins} | 💰 Total P&L: ₹{pnl:.2f}")

# ---------------- MAIN ------------------
df = load_nifty_data(period=f"{analysis_days}d")
df = filter_last_n_days(df, analysis_days)

# Dynamically rename OHLC columns if needed
rename_map = {}
for col in df.columns:
    if 'open' in col.lower() and col != 'open':
        rename_map[col] = 'open'
    if 'high' in col.lower() and col != 'high':
        rename_map[col] = 'high'
    if 'low' in col.lower() and col != 'low':
        rename_map[col] = 'low'
    if 'close' in col.lower() and col != 'close':
        rename_map[col] = 'close'
    if 'volume' in col.lower() and col != 'volume':
        rename_map[col] = 'volume'

df = df.rename(columns=rename_map)

required_cols = ['datetime', 'open', 'high', 'low', 'close']
if not all(col in df.columns for col in required_cols):
    st.error("Missing required OHLC columns.")
    st.stop()

#result, exit_time, exit_price = '❌ No Entry', '-', 0
breakout_df, df_3pm = generate_trade_logs(df, offset_points, sl_percent)
fig = plot_candlestick_chart(df, df_3pm)

st.subheader("🕯️ NIFTY Candlestick Chart")
st.plotly_chart(fig, use_container_width=True)

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}; font-weight: bold;'
# Define this function before usage
def color_pnl_text(pnl):
    if pnl > 0:
        return f"🟢 {pnl}"
    elif pnl < 0:
        return f"🔴 {pnl}"
    else:
        return f"{pnl}"

# Filter the breakout dataframe to only show entries
filtered_breakout_df = breakout_df[breakout_df['Result'] != '❌ No Entry']

# Apply colored P&L text
filtered_breakout_df['P&L'] = breakout_df['P&L'].apply(color_pnl_text)

st.dataframe(filtered_breakout_df)




filtered_breakout_df['P&L'] = breakout_df['P&L'].apply(color_pnl_text)
st.dataframe(filtered_breakout_df)

#filtered_breakout_df = filtered_breakout_df[filtered_breakout_df['Result'] != '❌ No Entry']
filtered_breakout_df = breakout_df[breakout_df['Result'] != '❌ No Entry']
st.dataframe(filtered_breakout_df.style.applymap(color_pnl, subset=['P&L']))


st.subheader("📘 Breakout Logs")
st.dataframe(filtered_breakout_df)
show_trade_metrics(filtered_breakout_df, "Breakout Trades")

st.download_button("📥 Download Log", filtered_breakout_df.to_csv(index=False), file_name="breakout_log.csv")
