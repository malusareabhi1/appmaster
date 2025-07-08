import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import plotly.graph_objects as go
import numpy as np
import requests
#from datetime import datetime
import datetime


# ---------------- Sample Strategy Functions ----------------

# Paper Trading Logic
def run_paper_trading0(price_df, breakout_df, breakdown_df):
    trades = []

    def get_exit_price(trade_date):
        # trade_date is likely a datetime.date or Timestamp
        if isinstance(trade_date, pd.Timestamp):
            next_day = trade_date + pd.Timedelta(days=1)
            next_day_date = next_day.date()
        elif isinstance(trade_date, (datetime.date, datetime.datetime)):
            # If it's a date, add one day directly
            next_day_date = trade_date + pd.Timedelta(days=1)
            # If next_day_date is datetime.date or datetime.datetime, keep it
        else:
            # fallback
            next_day_date = trade_date
        
        next_day_data = price_df[price_df['datetime'].dt.date == next_day_date]
        if not next_day_data.empty:
            last_row = next_day_data.iloc[-1]
            return last_row['close'], last_row['datetime']
        else:
            # fallback: get last available data on trade_date
            day_data = price_df[price_df['datetime'].dt.date == (trade_date.date() if hasattr(trade_date, 'date') else trade_date)]
            if not day_data.empty:
                last_row = day_data.iloc[-1]
                return last_row['close'], last_row['datetime']
            else:
                # no data found, return NaN
                return np.nan, None

    for _, row in breakout_df.iterrows():
        trade_date = row['3PM Date']
        entry_price = row['Breakout Entry']
        exit_price, exit_time = get_exit_price(trade_date)
        if pd.isna(exit_price):
            continue  # skip if no exit price

        profit = exit_price - entry_price

        trades.append({
            '3PM Date': trade_date,
            'Type': 'CALL',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Exit Time': exit_time,
            'Status': 'Closed',
            'Profit': profit
        })

    for _, row in breakdown_df.iterrows():
        trade_date = row['3PM Date']
        entry_price = row['Breakdown Entry']
        exit_price, exit_time = get_exit_price(trade_date)
        if pd.isna(exit_price):
            continue  # skip if no exit price

        profit = entry_price - exit_price

        trades.append({
            '3PM Date': trade_date,
            'Type': 'PUT',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Exit Time': exit_time,
            'Status': 'Closed',
            'Profit': profit
        })

    return pd.DataFrame(trades)

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
        
def run_930_ce_pe_strategy(price_df, option_chain_df, target_points=50, sl_pct=5):
    trades = []

    price_df = price_df.sort_values('datetime')
    unique_dates = sorted(price_df['datetime'].dt.date.unique())

    for i in range(1, len(unique_dates)):
        today = unique_dates[i]
        prev_day = unique_dates[i - 1]

        # Get 3PM candle of previous day
        prev_day_3pm = price_df[
            (price_df['datetime'].dt.date == prev_day) &
            (price_df['datetime'].dt.hour == 15) &
            (price_df['datetime'].dt.minute == 0)
        ]
        if prev_day_3pm.empty:
            continue

        threepm_candle = prev_day_3pm.iloc[0]
        threepm_open = threepm_candle['open']
        threepm_close = threepm_candle['close']

        # Get 9:30 AM candle of today
        candle_930 = price_df[
            (price_df['datetime'].dt.date == today) &
            (price_df['datetime'].dt.hour == 9) &
            (price_df['datetime'].dt.minute == 30)
        ]
        if candle_930.empty:
            continue

        candle_930 = candle_930.iloc[0]
        spot_930 = candle_930['close']

        # Find ATM strike
        strikes = sorted(option_chain_df['strikePrice'].dropna())
        atm_strike = min(strikes, key=lambda x: abs(x - spot_930))

        # Check both CE and PE conditions
        if spot_930 > threepm_open and spot_930 > threepm_close:
            option_type = "CE"
            symbol = f"NIFTY_CE_{atm_strike}"
            entry_price = option_chain_df.loc[option_chain_df['strikePrice'] == atm_strike, 'CE_LTP'].values[0]
            stop_loss_price = entry_price - (entry_price * sl_pct / 100)
            target_spot = spot_930 + target_points
        elif spot_930 < threepm_open and spot_930 < threepm_close:
            option_type = "PE"
            symbol = f"NIFTY_PE_{atm_strike}"
            entry_price = option_chain_df.loc[option_chain_df['strikePrice'] == atm_strike, 'PE_LTP'].values[0]
            stop_loss_price = entry_price - (entry_price * sl_pct / 100)
            target_spot = spot_930 - target_points
        else:
            continue

        if pd.isna(entry_price) or entry_price == 0:
            continue

        exit_price = None
        exit_time = None
        exit_reason = "Time Exit"

        intraday_data = price_df[
            (price_df['datetime'].dt.date == today) &
            (price_df['datetime'].dt.time >= pd.to_datetime("09:45").time()) &
            (price_df['datetime'].dt.time <= pd.to_datetime("11:00").time())
        ]

        for _, row in intraday_data.iterrows():
            nifty_spot = row['close']
            time = row['datetime']

            # Simulated option price movement
            multiplier = 0.7
            delta = nifty_spot - spot_930 if option_type == "CE" else spot_930 - nifty_spot
            simulated_option_price = entry_price + (delta * multiplier)

            if simulated_option_price <= stop_loss_price:
                exit_price = stop_loss_price
                exit_time = time
                exit_reason = "Stop Loss Hit"
                break
            elif (option_type == "CE" and nifty_spot >= target_spot) or (option_type == "PE" and nifty_spot <= target_spot):
                exit_price = simulated_option_price
                exit_time = time
                exit_reason = "Target Hit"
                break

        if exit_price is None:
            last_candle = intraday_data[intraday_data['datetime'].dt.time == pd.to_datetime("11:00").time()]
            if not last_candle.empty:
                nifty_end = last_candle.iloc[0]['close']
                delta = nifty_end - spot_930 if option_type == "CE" else spot_930 - nifty_end
                exit_price = entry_price + (delta * multiplier)
                exit_time = last_candle.iloc[0]['datetime']

        #pnl = round(exit_price - entry_price, 2)
        if exit_price is None or entry_price is None:
            continue  # skip trade with missing data
        
        pnl = round(exit_price - entry_price, 2)


        trades.append({
            'Date': today,
            'Entry Spot': spot_930,
            'Target Spot': target_spot,
            'Option': symbol,
            'Trade Type': option_type,
            'Entry Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Exit Time': exit_time,
            'Exit Reason': exit_reason,
            'P&L': pnl
        })

    return pd.DataFrame(trades)


def run_sma_crossover_strategy(sma_fast, sma_slow):
    # Dummy logic (replace with actual)
    data = {
        "Date": ["2025-07-08"],
        "Strategy": ["SMA Crossover"],
        "Fast SMA": [sma_fast],
        "Slow SMA": [sma_slow],
        "P&L": [800]
    }
    return pd.DataFrame(data)

# ---------------- Streamlit Layout ----------------

st.set_page_config(page_title="NIFTY Strategy Backtester", layout="wide")

st.title("📈 NIFTY Option Strategy Dashboard")

# Sidebar: Strategy selection
strategy = st.sidebar.selectbox("Select Strategy", [
    "930 CE/PE Strategy",
    "SMA Crossover Strategy"
])

# Strategy-specific inputs and processing
if strategy == "930 CE/PE Strategy":
    analysis_days=3
    st.sidebar.subheader("📊 Parameters")
    target = st.sidebar.number_input("🎯 Target (Points)", value=50)
    stoploss = st.sidebar.number_input("🛑 Stop Loss (%)", value=5)

    # ✅ Load option chain (must come before trade log generation!)
    option_chain_df = get_nifty_option_chain_simple()
    if option_chain_df.empty:
        st.warning("⚠️ Option chain data is empty.")
        st.stop()

    

    st.subheader("🔍 Strategy: 930 CE/PE Breakout")
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
    # Plot chart
    fig = plot_candlestick_chart(df, df_3pm)
    st.subheader("🕯️ NIFTY Candlestick Chart (15m)")
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Now it's safe to call this
    trade_log_df, breakdown_df = generate_trade_logs(df, target, option_chain_df)
    # ✅ Display results
    st.subheader("\U0001F4C4 Breakout Trade Log (CALLS)")
    st.dataframe(trade_log_df)
    
    st.subheader("\U0001F4C4 Breakdown Trade Log (PUTS)")
    st.dataframe(breakdown_df)
    #st.write("Available columns:", df.columns.tolist())
    required_cols = ['datetime', 'open', 'high', 'low', 'close']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()
        #df = run_930_ce_pe_strategy(target, stoploss)
    paper_trades_df = run_paper_trading(df, trade_log_df, breakdown_df)


    st.subheader("📋 Paper Trading Results")
    st.dataframe(paper_trades_df)
    st.dataframe(df)

elif strategy == "SMA Crossover Strategy":
    st.sidebar.subheader("📊 Parameters")
    sma_fast = st.sidebar.number_input("Fast SMA Period", value=5)
    sma_slow = st.sidebar.number_input("Slow SMA Period", value=20)

    st.subheader("🔍 Strategy: SMA Crossover")
    df = run_sma_crossover_strategy(sma_fast, sma_slow)
    st.dataframe(df)

# Add optional summary section
if 'df' in locals() and not df.empty:
    st.markdown("---")
    #st.metric("📈 Total P&L", f"₹{df['P&L'].sum():,.2f}")
