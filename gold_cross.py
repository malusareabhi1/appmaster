import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

stocks = ["RELIANCE.NS"]  # test one first

end_date = datetime.today()
start_date = end_date - timedelta(days=200)

signals = []

for stock in stocks:
    try:
        df = yf.download(stock, start=start_date, end=end_date)

        print(stock, df.shape)
        print("Index unique?", df.index.is_unique)

        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        if len(df) < 2:
            continue

        prev = df.iloc[-2]
        latest = df.iloc[-1]

        # Extract scalar values safely
        prev_ema20 = prev["EMA20"].item() if hasattr(prev["EMA20"], 'item') else prev["EMA20"]
        prev_ema50 = prev["EMA50"].item() if hasattr(prev["EMA50"], 'item') else prev["EMA50"]
        latest_ema20 = latest["EMA20"].item() if hasattr(latest["EMA20"], 'item') else latest["EMA20"]
        latest_ema50 = latest["EMA50"].item() if hasattr(latest["EMA50"], 'item') else latest["EMA50"]

        print(f"{stock} prev EMA20: {prev_ema20}, EMA50: {prev_ema50}")
        print(f"{stock} latest EMA20: {latest_ema20}, EMA50: {latest_ema50}")

        if prev_ema20 < prev_ema50 and latest_ema20 > latest_ema50:
            signals.append({"Stock": stock, "Signal": "📈 Golden Cross"})
        elif prev_ema20 > prev_ema50 and latest_ema20 < latest_ema50:
            signals.append({"Stock": stock, "Signal": "📉 Death Cross"})

    except Exception as e:
        print(f"Error for {stock}: {e}")

print("Signals:", signals)
