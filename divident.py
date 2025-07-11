import streamlit as st
import pandas as pd
from datetime import datetime, date
from nsemine import NSE

st.set_page_config("Live Dividend Calendar", layout="wide")
st.title("📅 Live NSE Dividend Calendar (using nsemine)")

nse = NSE()
events = nse.corporate_actions().as_dict()
dividends = [e for e in events if e.get("actionType") == "Dividend"]

df = pd.DataFrame(dividends)
df['exDate'] = pd.to_datetime(df['exDate'], dayfirst=True, errors='coerce')

# Sidebar filter
st.sidebar.header("Filter by Ex‑Date")
start = st.sidebar.date_input("From", date.today())
end = st.sidebar.date_input("To", date.today())

mask = (df['exDate'].dt.date >= start) & (df['exDate'].dt.date <= end)
filtered = df.loc[mask, ['symbol', 'companyName', 'exDate', 'amount']]

filtered = filtered.rename(columns={
    'symbol': 'Symbol',
    'companyName': 'Company',
    'exDate': 'Ex‑Date',
    'amount': 'Dividend (₹)'
})

st.dataframe(filtered.sort_values('Ex‑Date'), use_container_width=True)
st.caption("Source: nsemine – real-time NSE/BSE corporate actions.")
