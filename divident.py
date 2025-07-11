import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(page_title="Live Dividend Calendar", layout="wide")
st.title("💰 Live NSE Dividend Calendar (Moneycontrol)")

# Fetch data from Moneycontrol
url = "https://www.moneycontrol.com/stocks/marketinfo/dividends_declared/index.php"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

if response.status_code != 200:
    st.error("❌ Failed to fetch data from Moneycontrol.")
else:
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", class_="tbldata14")

    if table is None:
        st.error("⚠️ Dividend data table not found. Moneycontrol may have changed their website layout.")
    else:
        # Parse table rows
        rows = table.find_all("tr")[1:]  # Skip header

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                data.append({
                    "Company": cols[0].text.strip(),
                    "Symbol": cols[1].text.strip(),
                    "Dividend": cols[2].text.strip(),
                    "Ex-Date": cols[3].text.strip(),
                    "Record Date": cols[4].text.strip(),
                })

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['Ex-Date'] = pd.to_datetime(df['Ex-Date'], errors='coerce')

        # Sidebar filters
        st.sidebar.header("📅 Filter by Ex-Date")
        today = datetime.today().date()
        start_date = st.sidebar.date_input("From", today)
        end_date = st.sidebar.date_input("To", today)

        # Filter by date
        mask = (df['Ex-Date'].dt.date >= start_date) & (df['Ex-Date'].dt.date <= end_date)
        filtered_df = df[mask]

        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
        st.caption("📌 Source: Moneycontrol.com | Data updated live.")
