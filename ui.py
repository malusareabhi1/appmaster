import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("📈 Scanner Dashboard")

# Sidebar Scanners
with st.sidebar:
    st.header("🧠 Discover Scanners")
    st.button("Short Term Bearish")
    st.button("Possible Uptrend")
    st.button("5 EMA")
    st.button("15 min Opening")
    st.button("Volume Burst")
    st.button("Short Term Bullish")

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    index = st.selectbox("Scan on", ["Nifty 500", "Nifty 50", "Bank Nifty"])
with col2:
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
with col3:
    timeframe = st.selectbox("Time Frame", ["1 Day", "15 min", "5 min"])
with col4:
    st.button("Run Scan", use_container_width=True)

st.markdown("---")

# Table Data
data = [
    ["ABBOTINDIA", "₹34320.00", "-0.29%", 782],
    ["ABFRL", "₹76.89", "+0.05%", 1569294],
    ["ABREL", "₹2194.80", "-0.28%", 26988],
    ["ABSLAMC", "₹846.95", "+0.31%", 137508]
]
df = pd.DataFrame(data, columns=["Symbol", "LTP", "Change", "Volume"])

# Search
search = st.text_input("Search from results")

# Filtered table
if search:
    df = df[df["Symbol"].str.contains(search.upper())]

# Display table with Buy/Sell
for idx, row in df.iterrows():
    cols = st.columns([2, 2, 1, 2, 1, 1])
    cols[0].markdown(f"**{row['Symbol']}**")
    cols[1].markdown(row["LTP"])
    cols[2].markdown(row["Change"])
    cols[3].markdown(f"📊 {row['Volume']}")
    cols[4].button("🟢 B", key=f"buy_{idx}")
    cols[5].button("🔴 S", key=f"sell_{idx}")

