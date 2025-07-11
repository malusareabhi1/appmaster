import streamlit as st
import pandas as pd
from datetime import datetime
from nsepython import dividend_calendar  # from nsepython library

# 🛠️ App setup
st.set_page_config(page_title="Dividend Calendar", layout="wide")
st.title("📅 NSE Dividend Calendar (Live Data)")

# Sidebar date filter
st.sidebar.header("Filter by Ex-Date")
today = datetime.today().date()
start_date = st.sidebar.date_input("From", today)
end_date = st.sidebar.date_input("To", today)

# Fetch and process data
data = dividend_calendar()  # fetches upcoming/ex-date dividend info
df = pd.DataFrame(data)
df['exDate'] = pd.to_datetime(df['exDate'], format="%d-%b-%Y")
df['amount'] = df['amount'].astype(float)

# Filter DataFrame
mask = (df['exDate'].dt.date >= start_date) & (df['exDate'].dt.date <= end_date)
filtered = df.loc[mask, ['symbol', 'companyName', 'exDate', 'amount']]

# Display
st.dataframe(filtered.rename(columns={
    'symbol': 'Symbol',
    'companyName': 'Company',
    'exDate': 'Ex‑Date',
    'amount': 'Dividend (₹)'
}), use_container_width=True)

st.markdown("💡 *Powered by [`nsepython`](https://pypi.org/project/nsepython)*")
