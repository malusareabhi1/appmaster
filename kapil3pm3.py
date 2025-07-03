import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
#import plotly.express as px
st.set_page_config(page_title="NIFTY 15-Min Chart with 3PM Breakout Strategy", layout="wide")

st.title("📈 NIFTY 15-Min Chart – 3PM Breakout/Breakdown Strategy")

st.sidebar.header("🛠️ Strategy Settings")

offset_points = st.sidebar.number_input("🔼 Offset Points (Breakout/Breakdown)", value=100, step=10)
analysis_days = st.sidebar.slider("📅 Days to Analyze", min_value=10, max_value=90, value=60, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛑 Stop Loss / Exit Info")

st.sidebar.markdown("""
- **Breakout SL**: Based on 3PM Low  
- **Breakout Exit**:  
    - 🎯 Target = Entry + 1.5×(Entry - SL)  
    - 🛑 SL Hit = below 3PM Low  
    - ⏰ Time-based: End of session  

- **Breakdown SL**: Based on 3PM High  
- **Breakdown Exit**:  
    - 🎯 Target = Entry - 1.5×(SL - Entry)  
    - 🛑 SL Hit = above 3PM High  
    - ⏰ Time-based: End of session
""")


st.markdown("""
## 📘 Strategy Explanation

This intraday breakout/backtest strategy is based on the NIFTY 15-minute chart.

- 🔼 **Breakout Logic**: At 3:00 PM, capture the high of the 15-minute candle. On the next trading day, if price crosses 3PM High + offset points, mark as a breakout.
- 🔽 **Breakdown Logic**: Track 3PM Close. On the next day, if price crosses below previous close and then drops offset points lower, mark as breakdown.

*Useful for swing and intraday traders planning trades based on end-of-day momentum.*

---
""")

#import plotly.express as px

def plot_cumulative_pnl(df, title="Cumulative P&L"):
    df['cumulative_pnl'] = df['P&L'].cumsum()
    fig = px.line(df, x='3PM Date', y='cumulative_pnl', title=title)
    fig.update_layout(height=400)
    return fig
    
@st.cache_data(ttl=3600)
def load_nifty_data(ticker="^NSEI", interval="15m", period="60d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            st.error("❌ No data returned from yfinance.")
            st.stop()

        df.reset_index(inplace=True)

        # ✅ Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # ✅ Find datetime column automatically
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)

        if not datetime_col:
            st.error("❌ No datetime column found after reset_index().")
            st.write("📋 Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={datetime_col: 'datetime'}, inplace=True)

        # ✅ Convert to datetime and localize
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')

        # ✅ Now lowercase column names
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
    unique_days = sorted(df['date'].unique())
    last_days = unique_days[-n_days:]
    filtered_df = df[df['date'].isin(last_days)].copy()
    filtered_df.drop(columns='date', inplace=True)
    return filtered_df

def generate_trade_logs(df, offset):
    df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
    breakout_logs = []
    breakdown_logs = []

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()

        threepm_high = current['high']
        threepm_close = current['close']
        threepm_low = current['low']

        # Breakout targets and stops
        entry_breakout = threepm_high + offset
        sl_breakout = threepm_low
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * 1.5

        # Breakdown targets and stops
        entry_breakdown = threepm_close
        sl_breakdown = threepm_high
        target_breakdown = entry_breakdown - (sl_breakdown - entry_breakdown) * 1.5

        next_day_data = df[(df['datetime'].dt.date == next_day_date) & 
                           (df['datetime'].dt.time >= pd.to_datetime("09:30").time())].copy()
        next_day_data.sort_values('datetime', inplace=True)

        # --- Breakout Logic ---
        entry_row = next_day_data[next_day_data['high'] >= entry_breakout]
        if not entry_row.empty:
            entry_time = entry_row.iloc[0]['datetime']
            after_entry = next_day_data[next_day_data['datetime'] >= entry_time]

            target_hit = after_entry[after_entry['high'] >= target_breakout]
            sl_hit = after_entry[after_entry['low'] <= sl_breakout]

            if not target_hit.empty:
                breakout_result = '🎯 Target Hit'
                exit_price = target_breakout
                exit_time = target_hit.iloc[0]['datetime']
            elif not sl_hit.empty:
                breakout_result = '🛑 Stop Loss Hit'
                exit_price = sl_breakout
                exit_time = sl_hit.iloc[0]['datetime']
            else:
                breakout_result = '⏰ Time Exit'
                exit_price = after_entry.iloc[-1]['close']
                exit_time = after_entry.iloc[-1]['datetime']

            pnl = round(exit_price - entry_breakout, 2)
        else:
            entry_time = None
            exit_time = None
            breakout_result = '❌ No Entry'
            pnl = 0.0

        breakout_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            '3PM High': round(threepm_high, 2),
            'Entry': round(entry_breakout, 2),
            'SL': round(sl_breakout, 2),
            'Target': round(target_breakout, 2),
            'Entry Time': entry_time.time() if entry_time else '-',
            'Exit Time': exit_time.time() if exit_time else '-',
            'Result': breakout_result,
            'P&L': pnl
        })

        # --- Breakdown Logic ---
        crossed_down = False
        entry_time = None
        exit_time = None
        pnl = 0.0

        for j in range(1, len(next_day_data)):
            prev = next_day_data.iloc[j - 1]
            curr = next_day_data.iloc[j]

            if not crossed_down and prev['high'] > entry_breakdown and curr['low'] < entry_breakdown:
                crossed_down = True
                entry_time = curr['datetime']
                after_entry = next_day_data[next_day_data['datetime'] >= entry_time]

                target_hit = after_entry[after_entry['low'] <= target_breakdown]
                sl_hit = after_entry[after_entry['high'] >= sl_breakdown]

                if not target_hit.empty:
                    breakdown_result = '🎯 Target Hit'
                    exit_price = target_breakdown
                    exit_time = target_hit.iloc[0]['datetime']
                elif not sl_hit.empty:
                    breakdown_result = '🛑 Stop Loss Hit'
                    exit_price = sl_breakdown
                    exit_time = sl_hit.iloc[0]['datetime']
                else:
                    breakdown_result = '⏰ Time Exit'
                    exit_price = after_entry.iloc[-1]['close']
                    exit_time = after_entry.iloc[-1]['datetime']

                pnl = round(entry_breakdown - exit_price, 2)
                break
        else:
            breakdown_result = '❌ No Entry'
            pnl = 0.0

        breakdown_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            '3PM Close': round(threepm_close, 2),
            'Entry': round(entry_breakdown, 2),
            'SL': round(sl_breakdown, 2),
            'Target': round(target_breakdown, 2),
            'Entry Time': entry_time.time() if entry_time else '-',
            'Exit Time': exit_time.time() if exit_time else '-',
            'Result': breakdown_result,
            'P&L': pnl
        })

    breakout_df = pd.DataFrame(breakout_logs)
    breakdown_df = pd.DataFrame(breakdown_logs)
    return breakout_df, breakdown_df

def plot_candlestick_chart(df, df_3pm):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="NIFTY"
    )])

    fig.update_traces(increasing_line_color='green', decreasing_line_color='red')

    # Add 3PM High and Low markers
    fig.add_trace(go.Scatter(
        x=df_3pm['datetime'],
        y=df_3pm['high'],
        mode='markers',
        name='3PM High',
        marker=dict(color='orange', size=8, symbol='triangle-up')
    ))

    fig.add_trace(go.Scatter(
        x=df_3pm['datetime'],
        y=df_3pm['low'],
        mode='markers',
        name='3PM Low',
        marker=dict(color='cyan', size=8, symbol='triangle-down')
    ))

    # Add vertical lines for each 3PM candle
    for dt in df_3pm['datetime']:
        fig.add_vline(x=dt, line_width=1, line_dash="dot", line_color="yellow")

    fig.update_layout(
        title="NIFTY 15-Min Chart (Last {} Trading Days)".format(analysis_days),
        xaxis_title="DateTime (IST)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.15], pattern="hour")
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
    total_trades = len(df)
    wins = df[df['Result'] == '🎯 Target Hit'].shape[0]
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = df['P&L'].mean() if total_trades > 0 else 0
    total_pnl = df['P&L'].sum() if total_trades > 0 else 0

    st.success(f"✅ {label} – Total Trades: {total_trades}, Wins: {wins} ({win_rate:.2f}%), Avg P&L: ₹{avg_pnl:.2f}, Total P&L: ₹{total_pnl:,.2f}")

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}; font-weight: bold;'

# ----------------------- MAIN ------------------------

df = load_nifty_data(period=f"{analysis_days}d")

if df.empty:
    st.stop()

df = filter_last_n_days(df, analysis_days)
df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)
#st.write("Available columns:", df.columns.tolist())
# ✅ Manually set the required columns (works for most tickers)
df = df.rename(columns={
    'datetime': 'datetime',
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})
#st.write("Available columns:", df.columns.tolist())
required_cols = ['datetime', 'open', 'high', 'low', 'close']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()



trade_log_df, breakdown_df = generate_trade_logs(df, offset_points)
#st.write("📋 df_3pm Columns:", df_3pm.columns.tolist())
df_3pm = df_3pm.rename(columns={
    'datetime': 'datetime',
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})
#st.write("📋 df_3pm Columns:", df_3pm.columns.tolist())
# Plot chart
fig = plot_candlestick_chart(df, df_3pm)
st.subheader("🕯️ NIFTY Candlestick Chart (15m)")
st.plotly_chart(fig, use_container_width=True)

# Show breakout trades
st.subheader("📘 Breakout Trades – Next Day Break 3PM High + Offset Points")
st.dataframe(trade_log_df.style.applymap(color_pnl, subset=['P&L']))

show_trade_metrics(trade_log_df, "Breakout Trades")

st.download_button(
    label="📥 Download Breakout Log",
    data=trade_log_df.to_csv(index=False),
    file_name="nifty_3pm_breakout_log.csv",
    mime="text/csv",
    key="breakout_csv"
)

# Show breakdown trades
st.subheader("📉 Breakdown Trades – Next Day Cross Below 3PM Close & Drop Offset Points")
st.dataframe(breakdown_df.style.applymap(color_pnl, subset=['P&L']))

show_trade_metrics(breakdown_df, "Breakdown Trades")

st.download_button(
    label="📥 Download Breakdown Log",
    data=breakdown_df.to_csv(index=False),
    file_name="nifty_3pm_breakdown_log.csv",
    mime="text/csv",
    key="breakdown_csv"
)

# Cumulative P&L plots
st.subheader("📊 Cumulative P&L Over Time")

def plot_cumulative_pnl(df, title):
    df = df.copy()
    df['cum_pnl'] = df['P&L'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Next Day'],
        y=df['cum_pnl'],
        mode='lines+markers',
        name=title,
        line=dict(color='lime')
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative P&L (₹)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=400
    )
    return fig
st.plotly_chart(plot_cumulative_pnl(trade_log_df, "Breakout – Cumulative P&L Over Time"))
st.plotly_chart(plot_cumulative_pnl(breakdown_df, "Breakdown – Cumulative P&L Over Time"))
#st.plotly_chart(plot_cumulative_pnl(trade_log

                                    
