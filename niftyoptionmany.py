import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# Your existing strategy functions (run_930_ce_pe_strategy, run_sma_crossover_option_strategy) here
# (Make sure these functions are defined above this Streamlit app code)

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


# Example minimal option chain fetcher (you can replace with your own function)
@st.cache_data(ttl=300)
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

@st.cache_data(ttl=3600)
def load_nifty_data():
    df = yf.download("^NSEI", interval="15m", period="10d", progress=False)
    df.reset_index(inplace=True)
    df['datetime'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
    return df

# Streamlit UI starts here
st.set_page_config(page_title="NIFTY Option Strategy Runner", layout="wide")
st.title("📊 NIFTY Option Strategy Backtester")

# Sidebar for strategy selection
strategy = st.sidebar.selectbox(
    "Choose Trading Strategy",
    ("930 CE/PE Strategy", "SMA Crossover Option Strategy")
)

# Load data once
with st.spinner("Loading NIFTY price data..."):
    price_df = load_nifty_data()
with st.spinner("Fetching Option Chain..."):
    option_chain_df = get_nifty_option_chain_simple()

if price_df.empty or option_chain_df.empty:
    st.error("Data loading failed. Check your internet or NSE site availability.")
    st.stop()

# Execute selected strategy
if strategy == "930 CE/PE Strategy":
    st.sidebar.markdown("**Parameters:**")
    target_points = st.sidebar.number_input("Target Points", value=50)
    stop_loss_pct = st.sidebar.number_input("Stop Loss %", value=5)
    
    with st.spinner("Running 930 CE/PE Strategy..."):
        trades_df = run_930_ce_pe_strategy(price_df, option_chain_df, target_points=target_points, sl_pct=stop_loss_pct)
elif strategy == "SMA Crossover Option Strategy":
    st.sidebar.markdown("**Parameters:**")
    target_points = st.sidebar.number_input("Target Points", value=40)
    stop_loss_pct = st.sidebar.number_input("Stop Loss %", value=5)

    with st.spinner("Running SMA Crossover Option Strategy..."):
        trades_df = run_sma_crossover_option_strategy(price_df, option_chain_df, target_points=target_points, sl_pct=stop_loss_pct)

# Show trades on right side
st.subheader(f"Strategy: {strategy}")
if trades_df.empty:
    st.warning("No trades generated by this strategy for selected data.")
else:
    st.dataframe(trades_df)

    # Summary statistics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['P&L'] > 0])
    losses = len(trades_df[trades_df['P&L'] <= 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = trades_df['P&L'].sum()
    avg_pnl = trades_df['P&L'].mean()

    st.markdown(f"""
    **Total Trades:** {total_trades}  
    **Winning Trades:** {wins}  
    **Losing Trades:** {losses}  
    **Win Rate:** {win_rate:.2f}%  
    **Total P&L:** ₹{total_pnl:.2f}  
    **Average P&L per Trade:** ₹{avg_pnl:.2f}
    """)

    # Day-wise performance table
    day_perf = trades_df.groupby('Date').agg(
        Trades=('P&L', 'count'),
        Wins=('P&L', lambda x: (x > 0).sum()),
        Losses=('P&L', lambda x: (x <= 0).sum()),
        Daily_PnL=('P&L', 'sum')
    ).reset_index()

    st.subheader("Day-wise Performance 📅")
    st.dataframe(day_perf)
