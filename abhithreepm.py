import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY 15-Min Chart", layout="wide")
#st.title("📈 NIFTY 15-Min Chart – Last 60 Days")
st.markdown("## 📘 Strategy Explanation – 3PM 15-Minute Candle")

st.info("""
**Strategy Overview:**

This intraday breakout/backtest strategy is based on the NIFTY 15-minute chart.

### 🔼 Breakout Logic:
- At **3:00 PM**, capture the high of the 15-minute candle.
- On the **next trading day**, if the price crosses **3PM High + 100 points**, it's marked as a successful breakout.
- This can be used as a directional continuation signal.

### 🔽 Breakdown Logic:
- Also track the **3PM Close** price.
- On the next day, if the price first crosses below the previous close and then drops **100 points lower**, it's marked as a successful breakdown.
- This indicates bearish continuation.

### 📈 Use Case:
- Helps identify key market strength or weakness near end-of-day, with potential for trade planning next morning.
- Useful for swing traders and intraday strategists.

Note: This is a **backtest logic** and not a buy/sell recommendation.
""")


with st.spinner("Fetching NIFTY 15-min data..."):
    ticker = "^NSEI"
    df = yf.download(ticker, interval="15m", period="60d", progress=False)
    df.columns = df.columns.get_level_values(-0)

    # Reset index to move Datetime from index to column
    df.reset_index(inplace=True)

    # Get actual datetime column name after reset
    datetime_col = df.columns[0]

    # Convert to datetime and then IST
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[datetime_col] = df[datetime_col].dt.tz_convert('Asia/Kolkata')

    # Rename to 'datetime' for consistency
   # Rename column
    df = df.rename(columns={datetime_col: 'datetime'})
    
    # Remove off-market hours (before 9:15 AM or after 3:30 PM IST)
    df = df[(df['datetime'].dt.time >= pd.to_datetime("09:15").time()) &
            (df['datetime'].dt.time <= pd.to_datetime("15:30").time())]
    # Convert columns to lowercase now
df.columns = df.columns.str.lower()

    


# After loading and processing dataframe df
# Convert all column names to lowercase
df.columns = df.columns.str.lower()

# Filter to last 10 trading days
df['date'] = df['datetime'].dt.date
last_10_trading_days = sorted(df['date'].unique())[-60:]
df = df[df['date'].isin(last_10_trading_days)]
df = df.drop(columns='date')

# ✅ Now safe to filter 3PM candles (after lowercase and filtering)
df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)]
###############################################################################################################################
# Trade log: check if next day breaks 3PM high + 50 points
trade_log = []

for i in range(len(df_3pm) - 1):  # Avoid last day, no "next day" after it
    current_row = df_3pm.iloc[i]
    next_row = df_3pm.iloc[i + 1]

    threepm_date = current_row['datetime'].date()
    next_day_date = next_row['datetime'].date()

    threepm_high = current_row['high']
    target = threepm_high + 100

    # Filter next day's data
    #next_day_data = df[df['datetime'].dt.date == next_day_date]
    next_day_data = df[
        (df['datetime'].dt.date == next_day_date) &
        (df['datetime'].dt.time >= pd.to_datetime("09:30").time())
    ]


    if next_day_data.empty:
        hit = False
        hit_time = None
    else:
        breakout = next_day_data[next_day_data['high'] >= target]
        hit = not breakout.empty
        hit_time = breakout['datetime'].iloc[0] if hit else None

    trade_log.append({
        '3PM Date': threepm_date,
        'Next Day': next_day_date,
        '3PM High': round(threepm_high, 2),
        'Target (High + 100)': round(target, 2),
        'Hit?': '✅ Yes' if hit else '❌ No',
        'Hit Time': hit_time.time() if hit else '-'
    })

# Convert to DataFrame
#trade_log_df = pd.DataFrame(trade_log)


####################################################################################################################################


# Additional trade log for close-breakdown and 100-point fall
close_breakdown_log = []

for i in range(len(df_3pm) - 1):
    current_row = df_3pm.iloc[i]
    next_day_date = df_3pm.iloc[i + 1]['datetime'].date()

    threepm_close = current_row['close']
    target_down = threepm_close - 100

    # Get next day's data
    #next_day_data = df[df['datetime'].dt.date == next_day_date].copy()
    next_day_data = df[
        (df['datetime'].dt.date == next_day_date) &
        (df['datetime'].dt.time >= pd.to_datetime("09:30").time())
    ].copy()
    if next_day_data.empty:
        continue

    # Sort by time to simulate price movement
    next_day_data.sort_values(by='datetime', inplace=True)

    # Detect first time price crosses below the previous close from above
    crossed_down = False
    target_hit = False
    hit_time = None

    for j in range(1, len(next_day_data)):
        prev_candle = next_day_data.iloc[j - 1]
        this_candle = next_day_data.iloc[j]

        # Check if we crossed the close from above to below
        if not crossed_down and prev_candle['high'] > threepm_close and this_candle['low'] < threepm_close:
            crossed_down = True
            cross_time = this_candle['datetime']

        # After crossing, check if it dropped 100 points
        if crossed_down and this_candle['low'] <= target_down:
            target_hit = True
            hit_time = this_candle['datetime']
            break

    close_breakdown_log.append({
        '3PM Date': current_row['datetime'].date(),
        'Next Day': next_day_date,
        '3PM Close': round(threepm_close, 2),
        'Target (Close - 100)': round(target_down, 2),
        'Crossed & Dropped 100?': '✅ Yes' if target_hit else '❌ No',
        'Drop Hit Time': hit_time.time() if hit_time else '-'
    })

# Convert to DataFrame
# breakdown_df = pd.DataFrame(close_breakdown_log)
############################################################################################################################

def generate_trade_log(df_3pm, df):
    breakout_logs = []
    breakdown_logs = []

    for i in range(len(df_3pm) - 1):
        current = df_3pm.iloc[i]
        next_day_date = df_3pm.iloc[i + 1]['datetime'].date()

        threepm_high = current['high']
        threepm_close = current['close']
        threepm_low = current['low']

        # Entry/SL/Target for breakout
        entry_breakout = threepm_high + 100
        sl_breakout = threepm_low
        target_breakout = entry_breakout + (entry_breakout - sl_breakout) * 1.5

        # Entry/SL/Target for breakdown
        entry_breakdown = threepm_close
        sl_breakdown = threepm_high
        target_breakdown = entry_breakdown - (sl_breakdown - entry_breakdown) * 1.5

        next_day_data = df[(df['datetime'].dt.date == next_day_date) &
                           (df['datetime'].dt.time >= pd.to_datetime("09:30").time())].copy()

        # Sort by datetime
        next_day_data.sort_values('datetime', inplace=True)

        ### 📈 Breakout Logic
        entry_row = next_day_data[next_day_data['high'] >= entry_breakout]
        if not entry_row.empty:
            entry_time = entry_row.iloc[0]['datetime']
            after_entry = next_day_data[next_day_data['datetime'] >= entry_time]

            target_hit = after_entry[after_entry['high'] >= target_breakout]
            sl_hit = after_entry[after_entry['low'] <= sl_breakout]

            if not target_hit.empty:
                breakout_result = '🎯 Target Hit'
                exit_time = target_hit.iloc[0]['datetime']
            elif not sl_hit.empty:
                breakout_result = '🛑 Stop Loss Hit'
                exit_time = sl_hit.iloc[0]['datetime']
            else:
                breakout_result = '⏰ Time Exit'
                exit_time = after_entry.iloc[-1]['datetime']
        else:
            entry_time = None
            exit_time = None
            breakout_result = '❌ No Entry'

        breakout_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            '3PM High': round(threepm_high, 2),
            'Entry': round(entry_breakout, 2),
            'SL': round(sl_breakout, 2),
            'Target': round(target_breakout, 2),
            'Entry Time': entry_time.time() if entry_time else '-',
            'Exit Time': exit_time.time() if exit_time else '-',
            'Result': breakout_result
        })

        ### 📉 Breakdown Logic
        crossed_down = False
        target_hit = False
        entry_time = None
        exit_time = None

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
                    exit_time = target_hit.iloc[0]['datetime']
                elif not sl_hit.empty:
                    breakdown_result = '🛑 Stop Loss Hit'
                    exit_time = sl_hit.iloc[0]['datetime']
                else:
                    breakdown_result = '⏰ Time Exit'
                    exit_time = after_entry.iloc[-1]['datetime']
                break
        else:
            breakdown_result = '❌ No Entry'

        breakdown_logs.append({
            '3PM Date': current['datetime'].date(),
            'Next Day': next_day_date,
            '3PM Close': round(threepm_close, 2),
            'Entry': round(entry_breakdown, 2),
            'SL': round(sl_breakdown, 2),
            'Target': round(target_breakdown, 2),
            'Entry Time': entry_time.time() if entry_time else '-',
            'Exit Time': exit_time.time() if exit_time else '-',
            'Result': breakdown_result
        })

    return pd.DataFrame(breakout_logs), pd.DataFrame(breakdown_logs)

    
trade_log_df, breakdown_df = generate_trade_log(df_3pm, df)




#####################################################################################################################################################################
# Keep only the last 10 **trading days**
df['date'] = df['datetime'].dt.date
last_10_trading_days = sorted(df['date'].unique())[-60:]
df = df[df['date'].isin(last_10_trading_days)]
df = df.drop(columns='date')  # Optional cleanup
# Debug columns
#st.write("Columns available:", df.columns.tolist())

# Check columns exist
required_cols = ['datetime', 'open', 'high', 'low', 'close']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    st.error(f"Missing columns: {missing}")
    st.stop()


# Show dataframe sample
#st.dataframe(df.head())

# Then proceed to plot
fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='NIFTY'
)])
#st.plotly_chart(fig)


# Preview data
#st.subheader("📋 Data Preview")
#st.dataframe(df.tail(50))

# Candlestick chart
st.subheader("🕯️ NIFTY Candlestick Chart (15m)")

fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="NIFTY"
)])

fig.update_traces(increasing_line_color='green', decreasing_line_color='red')

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

fig.update_layout(
    title="NIFTY 15-Min Chart (Last 10 Trading Days)",
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

###########################################################OUTPUT###########################OUTPUT######################################################
st.plotly_chart(fig, use_container_width=True)
####################################################################################OUTPUT########################################################
st.subheader("📘 Trade Log – Did Next Day Break 3PM High + 100 Points?")
st.dataframe(trade_log_df)
# Filter successful breakouts
#success_count = trade_log_df[trade_log_df['Hit?'] == '✅ Yes'].shape[0]
success_count = trade_log_df[trade_log_df['Result'] == '🎯 Target Hit'].shape[0]
total_checked = trade_log_df.shape[0]

st.markdown(f"### 📊 Summary: 3PM High + 100 Point Breakout")
st.success(f"✅ This scenario happened **{success_count} times** out of **{total_checked}** trading days.")


st.download_button(
    label="📥 Download Trade Log as CSV",
    data=trade_log_df.to_csv(index=False),
    file_name="nifty_3pm_breakout_tradelog.csv",
    mime="text/csv",
    key="breakout_log"
)
###############################################################################OUTPUT#############################################################
st.subheader("📉 Breakdown Log – Did Price Cross Below 3PM Close and Drop 100 Points?")
st.dataframe(breakdown_df)

# Show total count
count = breakdown_df[breakdown_df['Crossed & Dropped 100?'] == '✅ Yes'].shape[0]
st.success(f"✅ This scenario happened **{count} times** in the last {len(df_3pm)-1} trading days.")

st.download_button(
    label="📥 Download Trade Log as CSV",
    data=breakdown_df.to_csv(index=False),
    file_name="nifty_3pm_breakout_tradelog.csv",
    mime="text/csv",
     key="breakdown_log"
    
)
###########################################################################OUTPUT#################################################################
st.subheader("📘 Trade Log – Did Next Day Break 3PM High + 100 Points?")
st.dataframe(trade_log_df)

st.success(f"✅ Target hit {trade_log_df[trade_log_df['Result'] == '🎯 Target Hit'].shape[0]} times out of {len(trade_log_df)} trades.")

st.download_button(
    label="📥 Download Breakout Log",
    data=trade_log_df.to_csv(index=False),
    file_name="nifty_3pm_breakout_log.csv",
    mime="text/csv",
    key="breakout_csv"
)

st.subheader("📉 Breakdown Log – Did Price Cross 3PM Close & Drop?")
st.dataframe(breakdown_df)

st.success(f"✅ Target hit {breakdown_df[breakdown_df['Result'] == '🎯 Target Hit'].shape[0]} times out of {len(breakdown_df)} trades.")

st.download_button(
    label="📥 Download Breakdown Log",
    data=breakdown_df.to_csv(index=False),
    file_name="nifty_3pm_breakdown_log.csv",
    mime="text/csv",
    key="breakdown_csv"
)
