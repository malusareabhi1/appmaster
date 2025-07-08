import streamlit as st
import pandas as pd

# ---------------- Sample Strategy Functions ----------------

def run_930_ce_pe_strategy(target, stoploss):
    # Dummy logic (replace with actual)
    data = {
        "Date": ["2025-07-08"],
        "Strategy": ["930 CE/PE"],
        "Target": [target],
        "Stoploss": [stoploss],
        "P&L": [1200]
    }
    return pd.DataFrame(data)

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
    st.sidebar.subheader("📊 Parameters")
    target = st.sidebar.number_input("🎯 Target (Points)", value=50)
    stoploss = st.sidebar.number_input("🛑 Stop Loss (%)", value=5)

    st.subheader("🔍 Strategy: 930 CE/PE Breakout")
    df = run_930_ce_pe_strategy(target, stoploss)
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
    st.metric("📈 Total P&L", f"₹{df['P&L'].sum():,.2f}")
