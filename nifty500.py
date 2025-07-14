import requests
import pandas as pd

# NSE URL for NIFTY 200 CSV (may change; this is an example)
url = "https://www1.nseindia.com/content/indices/ind_nifty200list.csv"

# NSE blocks simple requests, so set headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

# Fetch CSV
response = requests.get(url, headers=headers)

# Save CSV temporarily
csv_path = "nifty200.csv"
with open(csv_path, "wb") as f:
    f.write(response.content)

# Read CSV using pandas
df = pd.read_csv(csv_path)

# Assuming the column with symbols is named 'Symbol' (confirm by inspecting CSV)
symbols = df['Symbol'].tolist()

# Add '.NS' suffix for yfinance
nifty200_stocks = [symbol + ".NS" for symbol in symbols]

# Print Python list format
print("nifty200_stocks = [")
for sym in nifty200_stocks:
    print(f'    "{sym}",')
print("]")
