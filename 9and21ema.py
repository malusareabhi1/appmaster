import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

# 📌 NIFTY 100 stock list (shortened for speed – replace with full list as needed)
nifty_100_stocks = [
    'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'ITC.NS', 'LT.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS'
]

# 🎛️ Sidebar
st.sidebar.title("🔍 EMA 9/21 Crossover Strategy")
selected_stock = st.sidebar.selectbox("Select NIFTY100 Stock", nifty_100_stocks)
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '1d'])
days = st.sidebar.slider("Lookback Period (days)", 5, 60, 30)

# 📈 Page Title
st.title("📊 9 & 21 EMA Crossover Strategy – NIFTY 100 Stocks")
st.markdown(f"Analyzing **{selected_stock}** on **{timeframe}** timeframe for last **{days} days**")

# 📥 Fetch data
@st.cache_data
def load_data(symbol, interval, days):
    df = yf.download(symbol, period=f'{days}d', interval=interval)
    df.dropna(inplace=True)
    return df

df = load_data(selected_stock, timeframe, days)

# 📉 EMA Calculation
df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

# 🎯 Signal Logic
df['Signal'] = 0
df['Signal'][df['EMA9'] > df['EMA21']] = 1
df['Signal'][df['EMA9'] < df['EMA21']] = -1
df['Crossover'] = df['Signal'].diff()

# 📍 Trade Logic
trades = []
position = None

for i in range(1, len(df)):
    row = df.iloc[i]

    # ✔️ Make sure this is scalar!
    crossover_value = int(row['Crossover'])

    if crossover_value == 2:
        entry_price = row['Close']
        entry_time = row.name
        position = {'entry_price': entry_price, 'entry_time': entry_time}

    elif crossover_value == -2 and position is not None:
        exit_price = row['Close']
        exit_time = row.name
        pnl = round(exit_price - position['entry_price'], 2)
        duration = exit_time - position['entry_time']
        trades.append({
            'Stock': selected_stock,
            'Entry Time': position['entry_time'],
            'Entry Price': position['entry_price'],
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'PnL': pnl,
            'Duration': duration
        })
        position = None


# 📋 Show trade summary
trade_df = pd.DataFrame(trades)
st.subheader("📑 Trade Log")
if not trade_df.empty:
    st.dataframe(trade_df)
    st.success(f"Total Trades: {len(trade_df)} | Net PnL: ₹{trade_df['PnL'].sum():.2f}")
else:
    st.warning("No trades found for selected stock & timeframe.")

# 📊 Candlestick Chart
st.subheader("📉 Price Chart with EMA")
fig = go.Figure()

# Candle
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Candles'
))

# EMA lines
fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA 9', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA 21', line=dict(color='orange')))

# Buy/Sell Markers
for trade in trades:
    fig.add_trace(go.Scatter(
        x=[trade['Entry Time']], y=[trade['Entry Price']],
        mode='markers', marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy'
    ))
    fig.add_trace(go.Scatter(
        x=[trade['Exit Time']], y=[trade['Exit Price']],
        mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell'
    ))

fig.update_layout(xaxis_rangeslider_visible=False, height=600)
st.plotly_chart(fig, use_container_width=True)

