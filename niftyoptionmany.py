import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---------------- Sample Strategy Functions ----------------
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

    # Plot chart
    fig = plot_candlestick_chart(df, df_3pm)
    st.subheader("🕯️ NIFTY Candlestick Chart (15m)")
    st.plotly_chart(fig, use_container_width=True)

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
