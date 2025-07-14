import streamlit as st
import requests
import pandas as pd

st.title("📋 Fetch & Display NIFTY 200 Stocks List")

url = "https://www1.nseindia.com/content/indices/ind_nifty200list.csv"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

if st.button("Fetch NIFTY 200 List from NSE"):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise error if bad response

        df = pd.read_csv(pd.compat.StringIO(response.text))

        # Extract symbols
        symbols = df['Symbol'].tolist()
        nifty200_stocks = [s + ".NS" for s in symbols]

        st.success(f"Fetched {len(nifty200_stocks)} stocks!")

        st.write("Sample symbols:")
        st.write(nifty200_stocks[:20])

        st.download_button(
            label="Download full list as CSV",
            data="\n".join(nifty200_stocks),
            file_name="nifty200_stocks.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
