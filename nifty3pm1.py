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


    # Add horizontal dotted lines from 3PM to next day 3PM for Open and Close prices
    for i in range(len(df_3pm) - 1):
        row = df_3pm.iloc[i]
        next_row = df_3pm.iloc[i + 1]
    
        dt_start = row['datetime'].to_pydatetime()
        dt_end = next_row['datetime'].to_pydatetime()
    
        # Horizontal line for Open
        fig.add_shape(
            type="line",
            x0=dt_start, x1=dt_end,
            y0=row['open'], y1=row['open'],
            line=dict(color="blue", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=dt_start,
            y=row['open'],
            text=f"{dt_start.strftime('%b %d')} ",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="blue"),
            bgcolor="black"
        )
    
        # Horizontal line for Close
        fig.add_shape(
            type="line",
            x0=dt_start, x1=dt_end,
            y0=row['close'], y1=row['close'],
            line=dict(color="orange", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=dt_start,
            y=row['close'],
            text=f"{dt_start.strftime('%b %d')} ",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="orange"),
            bgcolor="black"
        )
    # Add vertical lines for each 3PM candle
    # Handle last 3PM candle (no next day available)


   # for dt in df_3pm['datetime']:
        #fig.add_vline(x=dt, line_width=1, line_dash="dot", line_color="yellow")

    if len(df_3pm) > 0:
        last_row = df_3pm.iloc[-1]
        dt_start = last_row['datetime'].to_pydatetime()
        dt_end = df['datetime'].max().to_pydatetime()  # use last available candle
    
        # Open line
        fig.add_shape(
            type="line",
            x0=dt_start, x1=dt_end,
            y0=last_row['open'], y1=last_row['open'],
            line=dict(color="blue", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=dt_start,
            y=last_row['open'],
            text=f"{dt_start.strftime('%b %d')}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="blue"),
            bgcolor="black"
        )
    
        # Close line
        fig.add_shape(
            type="line",
            x0=dt_start, x1=dt_end,
            y0=last_row['close'], y1=last_row['close'],
            line=dict(color="orange", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=dt_start,
            y=last_row['close'],
            text=f"{dt_start.strftime('%b %d')} ",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="orange"),
            bgcolor="black"
        )

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



# ✅ Load price data
df = load_nifty_data(period=f"{analysis_days}d")
if df.empty:
    st.stop()

# ✅ Rename columns if needed
df = df.rename(columns={
    'open_^nsei': 'open',
    'high_^nsei': 'high',
    'low_^nsei': 'low',
    'close_^nsei': 'close',
    'volume_^nsei': 'volume'
})

# ✅ Filter last N days
df = filter_last_n_days(df, analysis_days)

# Now df is fully clean
df_3pm = df[(df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute == 0)].reset_index(drop=True)

#st.write("Available columns:", df.columns.tolist())
required_cols = ['datetime', 'open', 'high', 'low', 'close']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ✅ Load option chain (must come before trade log generation!)
option_chain_df = get_nifty_option_chain_simple()
if option_chain_df.empty:
    st.warning("⚠️ Option chain data is empty.")
    st.stop()

# ✅ Now it's safe to call this
trade_log_df, breakdown_df = generate_trade_logs(df, offset_points, option_chain_df)

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


# ✅ Display results
st.subheader("\U0001F4C4 Breakout Trade Log (CALLS)")
st.dataframe(trade_log_df)

st.subheader("\U0001F4C4 Breakdown Trade Log (PUTS)")
st.dataframe(breakdown_df)

#if not option_chain_df.empty:
    #st.subheader("\U0001F4CA NSE Option Chain Snapshot")
    #st.dataframe(option_chain_df.head(20))

# Parameters for paper trading (add near your sidebar or constants)
stop_loss_points = st.sidebar.number_input("Stop Loss Points", value=50, step=5)
target_points = st.sidebar.number_input("Target Points", value=100, step=5)

# Paper Trading Logic
def run_paper_trading(price_df, breakout_df, breakdown_df):
    """
    Simulate paper trades for breakout (CALL) and breakdown (PUT) options.

    Parameters:
    - price_df: DataFrame with 'datetime', 'open', 'high', 'low', 'close' prices
    - breakout_df: DataFrame with breakout trades info (CALLS)
    - breakdown_df: DataFrame with breakdown trades info (PUTS)

    Returns:
    - DataFrame with simulated trades and their PnL
    """

    trades = []

    # Helper function to get exit price for next trading day close
    def get_exit_price(trade_date):
        next_day = trade_date + pd.Timedelta(days=1)
        next_day_data = price_df[price_df['datetime'].dt.date == next_day.date()]
        if not next_day_data.empty:
            # Exit at close price of next day
            return next_day_data.iloc[-1]['close'], next_day_data.iloc[-1]['datetime']
        else:
            # If no next day data, exit at last available price
            last_row = price_df[price_df['datetime'].dt.date == trade_date].iloc[-1]
            return last_row['close'], last_row['datetime']

    # Process breakout (CALL) trades
    for _, row in breakout_df.iterrows():
        trade_date = row['3PM Date']
        entry_price = row['Breakout Entry']

        exit_price, exit_time = get_exit_price(trade_date)

        profit = exit_price - entry_price  # Profit for CALL

        trades.append({
            '3PM Date': trade_date,
            'Type': 'CALL',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Exit Time': exit_time,
            'Status': 'Closed',
            'Profit': profit
        })

    # Process breakdown (PUT) trades
    for _, row in breakdown_df.iterrows():
        trade_date = row['3PM Date']
        entry_price = row['Breakdown Entry']

        exit_price, exit_time = get_exit_price(trade_date)

        profit = entry_price - exit_price  # Profit for PUT

        trades.append({
            '3PM Date': trade_date,
            'Type': 'PUT',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Exit Time': exit_time,
            'Status': 'Closed',
            'Profit': profit
        })

    if not trades:
        return pd.DataFrame(columns=['3PM Date', 'Type', 'Entry Price', 'Exit Price', 'Exit Time', 'Status', 'Profit'])

    return pd.DataFrame(trades)


# After generating trade_log_df and breakdown_df, combine for paper trading:
all_trades_df = pd.concat([trade_log_df, breakdown_df], ignore_index=True)

paper_trades_df = run_paper_trading(df, all_trades_df)

st.subheader("📋 Paper Trading Results")
st.dataframe(paper_trades_df)

# Summary statistics
total_trades = len(paper_trades_df)
wins = len(paper_trades_df[paper_trades_df['Profit'] > 0])
losses = len(paper_trades_df[paper_trades_df['Profit'] <= 0])
win_rate = (wins / total_trades * 100) if total_trades else 0
total_profit = paper_trades_df['Profit'].sum()

st.markdown(f"""
**Total Trades:** {total_trades}  
**Wins:** {wins}  
**Losses:** {losses}  
**Win Rate:** {win_rate:.2f}%  
**Total PnL:** ₹{total_profit:.2f}  
""")

st.write("Paper Trading Data Columns:", paper_trades_df.columns.tolist())
st.write("Sample Paper Trading Data:", paper_trades_df.head())

