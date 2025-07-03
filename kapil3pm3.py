import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="NIFTY 15-Min 3PM Breakout Strategy", layout="wide")
st.title("📈 NIFTY 15-Min Chart – 3PM Breakout/Breakdown Strategy")

# Sidebar inputs
st.sidebar.header("🛠️ Strategy Settings")
offset_points = st.sidebar.number_input("🔼 Offset Points (Breakout/Breakdown)", value=100, step=10)
analysis_days = st.sidebar.slider("📅 Days to Analyze", min_value=10, max_value=90, value=60, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛑 Stop Loss & Target Settings")
stoploss_type = st.sidebar.radio("Stop Loss Type", ['Use 3PM Candle', 'Percent Based'])
if stoploss_type == 'Percent Based':
    stoploss_percent = st.sidebar.slider("Stop Loss %", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
else:
    stoploss_percent = None
target_multiplier = st.sidebar.slider("Target Multiplier (Risk:Reward)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.markdown("""
## 📘 Strategy Explanation

- 🔼 **Breakout**: If next day price crosses 3PM High + offset, enter long.
- 🔽 **Breakdown**: If price drops below 3PM Close and further offset, enter short.
- 🛑 **Stop Loss**: Can be % based or 3PM candle based (configurable).
- 🎯 **Target**: Entry ± (Entry - SL) × multiplier.
- ⏰ **Time Exit**: If neither SL nor Target hits.

---
""")

# -------------------- Data Functions --------------------

@st.cache_data(ttl=3600)
def load_nifty_data(ticker="^NSEI", interval="15m", period="60d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty:
        st.error("❌ No data returned from yfinance.")
        st.stop()

    df.reset_index(inplace=True)

    # Ensure all column names are strings
    df.columns = [str(col) for col in df.columns]

    datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    if datetime_col is None:
        st.error("❌ No datetime-like column found.")
        st.stop()

    df.rename(columns={datetime_col: 'datetime'}, inplace=True)

    # Convert all columns to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    df['datetime'] = pd.to_datetime(df['datetime'])

    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

    # Filter market hours
    df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) & (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]

    return df



def filter_last_n_days(df, n_days):
    df['date'] = df['datetime'].dt.date
    unique_days = sorted(df['date'].unique())
    last_days = unique_days[-n_days:]
    return df[df['date'].isin(last_days)].copy()

# -------------------- Trade Logic --------------------

def generate_trade_logs(df, offset, stoploss_type, stoploss_percent, target_multiplier):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
    breakout_logs, breakdown_logs = [], []
    #st.write("Current row keys:", current.index.tolist())
    #print("Current row keys:", current.index.tolist())

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()
        next_day_data = df[df['datetime'].dt.date == next_day_date]

        # Breakout
        entry_breakout = current['high'] + offset
        if stoploss_type == 'Percent Based':
            sl_breakout = entry_breakout * (1 - stoploss_percent / 100)
        else:
            sl_breakout = current['low']
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * target_multiplier

        breakout_row = next_day_data[next_day_data['high'] >= entry_breakout]
        if not breakout_row.empty:
            entry_time = breakout_row.iloc[0]['datetime']
            after_entry = next_day_data[next_day_data['datetime'] >= entry_time]
            target_hit = after_entry[after_entry['high'] >= target_breakout]
            sl_hit = after_entry[after_entry['low'] <= sl_breakout]
            if not target_hit.empty:
                result = '🎯 Target Hit'
                exit_price = target_breakout
                exit_time = target_hit.iloc[0]['datetime']
            elif not sl_hit.empty:
                result = '🛑 Stop Loss Hit'
                exit_price = sl_hit.iloc[0]['low']
                exit_time = sl_hit.iloc[0]['datetime']
            else:
                result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']
            pnl = round(exit_price - entry_breakout, 2)
        else:
            result = '❌ No Entry'
            exit_time = '-'
            pnl = 0.0

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

        # Breakdown
        entry_breakdown = current['close'] - offset
        if stoploss_type == 'Percent Based':
            sl_breakdown = entry_breakdown * (1 + stoploss_percent / 100)
        else:
            sl_breakdown = current['high']
        target_breakdown = entry_breakdown - (sl_breakdown - entry_breakdown) * target_multiplier

        breakdown_row = next_day_data[next_day_data['low'] <= entry_breakdown]
        if not breakdown_row.empty:
            entry_time = breakdown_row.iloc[0]['datetime']
            after_entry = next_day_data[next_day_data['datetime'] >= entry_time]
            target_hit = after_entry[after_entry['low'] <= target_breakdown]
            sl_hit = after_entry[after_entry['high'] >= sl_breakdown]
            if not target_hit.empty:
                result = '🎯 Target Hit'
                exit_price = target_breakdown
                exit_time = target_hit.iloc[0]['datetime']
            elif not sl_hit.empty:
                result = '🛑 Stop Loss Hit'
                exit_price = sl_hit.iloc[0]['high']
                exit_time = sl_hit.iloc[0]['datetime']
            else:
                result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']
            pnl = round(entry_breakdown - exit_price, 2)
        else:
            result = '❌ No Entry'
            exit_time = '-'
            pnl = 0.0

        breakdown_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            'Entry': round(entry_breakdown, 2),
            'SL': round(sl_breakdown, 2),
            'Target': round(target_breakdown, 2),
            'Exit Time': exit_time if isinstance(exit_time, str) else exit_time.time(),
            'Result': result,
            'P&L': pnl
        })

    return pd.DataFrame(breakout_logs), pd.DataFrame(breakdown_logs), df_3pm

# -------------------- Chart and Display --------------------

def plot_candlestick_chart(df, df_3pm):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price")])
    fig.add_trace(go.Scatter(x=df_3pm['datetime'], y=df_3pm['high'],
                             mode='markers', name='3PM High',
                             marker=dict(color='orange', size=8, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=df_3pm['datetime'], y=df_3pm['low'],
                             mode='markers', name='3PM Low',
                             marker=dict(color='cyan', size=8, symbol='triangle-down')))
    fig.update_layout(height=600, plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='white'), xaxis_rangeslider_visible=False)
    return fig

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}; font-weight: bold;'

def show_trade_metrics(df, label):
    total = len(df)
    wins = df[df['Result'] == '🎯 Target Hit'].shape[0]
    pnl = df['P&L'].sum()
    win_rate = (wins / total * 100) if total > 0 else 0
    st.success(f"{label}: {total} trades | 🎯 Wins: {wins} ({win_rate:.1f}%) | 💰 Total P&L: ₹{pnl:.2f}")

def plot_cumulative_pnl(df, title):
    df = df.copy()
    df['cum_pnl'] = df['P&L'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Next Day'], y=df['cum_pnl'],
                             mode='lines+markers', name='PnL', line=dict(color='lime')))
    fig.update_layout(title=title, height=400, plot_bgcolor='black',
                      paper_bgcolor='black', font=dict(color='white'))
    return fig

# -------------------- Main Execution --------------------

df = load_nifty_data()
#st.write("Columns of df after loading:", df.columns.tolist())

df = filter_last_n_days(df, analysis_days)

#st.write("Columns in df:", df.columns.tolist())
st.write("Columns in df:", df.columns.tolist())

df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
st.write("Columns in df_3pm:", df_3pm.columns.tolist())

# Also check first row keys explicitly
if not df_3pm.empty:
    st.write("Keys in first df_3pm row:", df_3pm.iloc[0].index.tolist())

breakout_df, breakdown_df, df_3pm = generate_trade_logs(df, offset_points, stoploss_type, stoploss_percent, target_multiplier)

st.subheader("🕯️ Candlestick Chart")
st.plotly_chart(plot_candlestick_chart(df, df_3pm), use_container_width=True)

st.subheader("📘 Breakout Trades")
st.dataframe(breakout_df.style.applymap(color_pnl, subset=['P&L']))
show_trade_metrics(breakout_df, "Breakout")
st.download_button("📥 Download Breakout Log", breakout_df.to_csv(index=False), file_name="breakout_log.csv")

st.subheader("📕 Breakdown Trades")
st.dataframe(breakdown_df.style.applymap(color_pnl, subset=['P&L']))
show_trade_metrics(breakdown_df, "Breakdown")
st.download_button("📥 Download Breakdown Log", breakdown_df.to_csv(index=False), file_name="breakdown_log.csv")

st.subheader("📊 Cumulative P&L Charts")
st.plotly_chart(plot_cumulative_pnl(breakout_df, "Breakout – Cumulative P&L"))
st.plotly_chart(plot_cumulative_pnl(breakdown_df, "Breakdown – Cumulative P&L"))
