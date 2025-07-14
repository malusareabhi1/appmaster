import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ------------------------------------
# Page Title and Strategy Explanation
# ------------------------------------
st.set_page_config(page_title="EMA Crossover + Pullback Scanner", layout="wide")
st.title("📈 EMA Crossover & Pullback to 20 EMA Scanner")

with st.expander("ℹ️ Strategy Explanation"):
    st.markdown("""
### 🔶 Golden Cross 📈
- 20 EMA crosses above 50 EMA → **Bullish** (entry signal)

### 🔷 Death Cross 📉
- 20 EMA crosses below 50 EMA → **Bearish** (exit signal)

### 🟢 Pullback to EMA20 (Buy the Dip)
- Price above EMA20 and EMA50 (Uptrend)
- Pullback to near EMA20 (within 1%)
- **Reversal candlestick** (Bullish Engulfing or Hammer)
- RSI > 40 for confirmation
""")

# ------------------------------------
# Stock selection and date range
# ------------------------------------
#nifty50_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'SBIN.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'AXISBANK.NS', 'ITC.NS', 'LT.NS', 'MARUTI.NS']
nifty50_stocks = [
    '360ONE.NS',
    '3MINDIA.NS',
    'ABB.NS',
    'ACC.NS',
    'ACMESOLAR.NS',
    'AIAENG.NS',
    'APLAPOLLO.NS',
    'AUBANK.NS',
    'AWL.NS',
    'AADHARHFC.NS',
    'AARTIIND.NS',
    'AAVAS.NS',
    'ABBOTINDIA.NS',
    'ACE.NS',
    'ADANIENSOL.NS',
    'ADANIENT.NS',
    'ADANIGREEN.NS',
    'ADANIPORTS.NS',
    'ADANIPOWER.NS',
    'ATGL.NS',
    'ABCAPITAL.NS',
    'ABFRL.NS',
    'ABREL.NS',
    'ABSLAMC.NS',
    'AEGISLOG.NS',
    'AFCONS.NS',
    'AFFLE.NS',
    'AJANTPHARM.NS',
    'AKUMS.NS',
    'APLLTD.NS',
    'ALIVUS.NS',
    'ALKEM.NS',
    'ALKYLAMINE.NS',
    'ALOKINDS.NS',
    'ARE&M.NS',
    'AMBER.NS',
    'AMBUJACEM.NS',
    'ANANDRATHI.NS',
    'ANANTRAJ.NS',
    'ANGELONE.NS',
    'APARINDS.NS',
    'APOLLOHOSP.NS',
    'APOLLOTYRE.NS',
    'APTUS.NS',
    'ASAHIINDIA.NS',
    'ASHOKLEY.NS',
    'ASIANPAINT.NS',
    'ASTERDM.NS',
    'ASTRAZEN.NS',
    'ASTRAL.NS',
    'ATUL.NS',
    'AUROPHARMA.NS',
    'AIIL.NS',
    'DMART.NS',
    'AXISBANK.NS',
    'BASF.NS',
    'BEML.NS',
    'BLS.NS',
    'BSE.NS',
    'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS',
    'BAJAJFINSV.NS',
    'BAJAJHLDNG.NS',
    'BAJAJHFL.NS',
    'BALKRISIND.NS',
    'BALRAMCHIN.NS',
    'BANDHANBNK.NS',
    'BANKBARODA.NS',
    'BANKINDIA.NS',
    'MAHABANK.NS',
    'BATAINDIA.NS',
    'BAYERCROP.NS',
    'BERGEPAINT.NS',
    'BDL.NS',
    'BEL.NS',
    'BHARATFORG.NS',
    'BHEL.NS',
    'BPCL.NS',
    'BHARTIARTL.NS',
    'BHARTIHEXA.NS',
    'BIKAJI.NS',
    'BIOCON.NS',
    'BSOFT.NS',
    'BLUEDART.NS',
    'BLUESTARCO.NS',
    'BBTC.NS',
    'BOSCHLTD.NS',
    'FIRSTCRY.NS',
    'BRIGADE.NS',
    'BRITANNIA.NS',
    'MAPMYINDIA.NS',
    'CCL.NS',
    'CESC.NS',
    'CGPOWER.NS',
    'CRISIL.NS',
    'CAMPUS.NS',
    'CANFINHOME.NS',
    'CANBK.NS',
    'CAPLIPOINT.NS',
    'CGCL.NS',
    'CARBORUNIV.NS',
    'CASTROLIND.NS',
    'CEATLTD.NS',
    'CENTRALBK.NS',
    'CDSL.NS',
    'CENTURYPLY.NS',
    'CERA.NS',
    'CHALET.NS',
    'CHAMBLFERT.NS',
    'CHENNPETRO.NS',
    'CHOLAHLDNG.NS',
    'CHOLAFIN.NS',
    'CIPLA.NS',
    'CUB.NS',
    'CLEAN.NS',
    'COALINDIA.NS',
    'COCHINSHIP.NS',
    'COFORGE.NS',
    'COHANCE.NS',
    'COLPAL.NS',
    'CAMS.NS',
    'CONCORDBIO.NS',
    'CONCOR.NS',
    'COROMANDEL.NS',
    'CRAFTSMAN.NS',
    'CREDITACC.NS',
    'CROMPTON.NS',
    'CUMMINSIND.NS',
    'CYIENT.NS',
    'DCMSHRIRAM.NS',
    'DLF.NS',
    'DOMS.NS',
    'DABUR.NS',
    'DALBHARAT.NS',
    'DATAPATTNS.NS',
    'DEEPAKFERT.NS',
    'DEEPAKNTR.NS',
    'DELHIVERY.NS',
    'DEVYANI.NS',
    'DIVISLAB.NS',
    'DIXON.NS',
    'LALPATHLAB.NS',
    'DRREDDY.NS',    
    'EIDPARRY.NS',
    'EIHOTEL.NS',
    'EICHERMOT.NS',
    'ELECON.NS',
    'ELGIEQUIP.NS',
    'EMAMILTD.NS',
    'EMCURE.NS',
    'ENDURANCE.NS',
    'ENGINERSIN.NS',
    'ERIS.NS',
    'ESCORTS.NS',
    'ETERNAL.NS',
    'EXIDEIND.NS',
    'NYKAA.NS',
    'FEDERALBNK.NS',
    'FACT.NS',
    'FINCABLES.NS',
    'FINPIPE.NS',
    'FSL.NS',
    'FIVESTAR.NS',
    'FORTIS.NS',
    'GAIL.NS',
    'GVT&D.NS',
    'GMRAIRPORT.NS',
    'GRSE.NS',
    'GICRE.NS',
    'GILLETTE.NS',
    'GLAND.NS',
    'GLAXO.NS',
    'GLENMARK.NS',
    'MEDANTA.NS',
    'GODIGIT.NS',
    'GPIL.NS',
    'GODFRYPHLP.NS',
    'GODREJAGRO.NS',
    'GODREJCP.NS',
    'GODREJIND.NS',
    'GODREJPROP.NS',
    'GRANULES.NS',
    'GRAPHITE.NS',
    'GRASIM.NS',
    'GRAVITA.NS',
    'GESHIP.NS',
    'FLUOROCHEM.NS',
    'GUJGASLTD.NS',
    'GMDCLTD.NS',
    'GNFC.NS',
    'GPPL.NS',
    'GSPL.NS',
    'HEG.NS',
    'HBLENGINE.NS',
    'HCLTECH.NS',
    'HDFCAMC.NS',
    'HDFCBANK.NS',
    'HDFCLIFE.NS',
    'HFCL.NS',
    'HAPPSTMNDS.NS',
    'HAVELLS.NS',
    'HEROMOTOCO.NS',
    'HSCL.NS',
    'HINDALCO.NS',
    'HAL.NS',
    'HINDCOPPER.NS',
    'HINDPETRO.NS',
    'HINDUNILVR.NS',
    'HINDZINC.NS',
    'POWERINDIA.NS',
    'HOMEFIRST.NS',
    'HONASA.NS',
    'HONAUT.NS',
    'HUDCO.NS',
    'HYUNDAI.NS',
    'ICICIBANK.NS',
    'ICICIGI.NS',
    'ICICIPRULI.NS',
    'IDBI.NS',
    'IDFCFIRSTB.NS',
    'IFCI.NS',
    'IIFL.NS',
    'INOXINDIA.NS',
    'IRB.NS',
    'IRCON.NS',
    'ITC.NS',
    'ITI.NS',
    'INDGN.NS',
    'INDIACEM.NS',
    'INDIAMART.NS',
    'INDIANB.NS',
    'IEX.NS',
    'INDHOTEL.NS',
    'IOC.NS',
    'IOB.NS',
    'IRCTC.NS',
    'IRFC.NS',
    'IREDA.NS',
    'IGL.NS',
    'INDUSTOWER.NS',
    'INDUSINDBK.NS',
    'NAUKRI.NS',
    'INFY.NS',
    'INOXWIND.NS',
    'INTELLECT.NS',
    'INDIGO.NS',
    'IGIL.NS',
    'IKS.NS',
    'IPCALAB.NS',
    'JBCHEPHARM.NS',
    'JKCEMENT.NS',
    'JBMA.NS',
    'JKTYRE.NS',
    'JMFINANCIL.NS',
    'JSWENERGY.NS',
    'JSWHL.NS',
    'JSWINFRA.NS',
    'JSWSTEEL.NS',
    'JPPOWER.NS',
    'J&KBANK.NS',
    'JINDALSAW.NS',
    'JSL.NS',
    'JINDALSTEL.NS',
    'JIOFIN.NS',
    'JUBLFOOD.NS',
    'JUBLINGREA.NS',
    'JUBLPHARMA.NS',
    'JWL.NS',
    'JUSTDIAL.NS',
    'JYOTHYLAB.NS',
    'JYOTICNC.NS',
    'KPRMILL.NS',
    'KEI.NS',
    'KNRCON.NS',
    'KPITTECH.NS',
    'KAJARIACER.NS',
    'KPIL.NS',
    'KALYANKJIL.NS',
    'KANSAINER.NS',
    'KARURVYSYA.NS',
    'KAYNES.NS',
    'KEC.NS',
    'KFINTECH.NS',
    'KIRLOSBROS.NS',
    'KIRLOSENG.NS',
    'KOTAKBANK.NS',
    'KIMS.NS',
    'LTF.NS',
    'LTTS.NS',
    'LICHSGFIN.NS',
    'LTFOODS.NS',
    'LTIM.NS',
    'LT.NS',
    'LATENTVIEW.NS',
    'LAURUSLABS.NS',
    'LEMONTREE.NS',
    'LICI.NS',
    'LINDEINDIA.NS',
    'LLOYDSME.NS',
    'LUPIN.NS',
    'MMTC.NS',
    'MRF.NS',
    'LODHA.NS',
    'MGL.NS',
    'MAHSEAMLES.NS',
    'M&MFIN.NS',
    'M&M.NS',
    'MANAPPURAM.NS',
    'MRPL.NS',
    'MANKIND.NS',
    'MARICO.NS',
    'MARUTI.NS',
    'MASTEK.NS',
    'MFSL.NS',
    'MAXHEALTH.NS',
    'MAZDOCK.NS',
    'METROPOLIS.NS',
    'MINDACORP.NS',
    'MSUMI.NS',
    'MOTILALOFS.NS',
    'MPHASIS.NS',
    'MCX.NS',
    'MUTHOOTFIN.NS',
    'NATCOPHARM.NS',
    'NBCC.NS',
    'NCC.NS',
    'NHPC.NS',
    'NLCINDIA.NS',
    'NMDC.NS',
    'NSLNISP.NS',
    'NTPCGREEN.NS',
    'NTPC.NS',
    'NH.NS',
    'NATIONALUM.NS',
    'NAVA.NS',
    'NAVINFLUOR.NS',
    'NESTLEIND.NS',
    'NETWEB.NS',
    'NETWORK18.NS',
    'NEULANDLAB.NS',
    'NEWGEN.NS',
    'NAM-INDIA.NS',
    'NIVABUPA.NS',
    'NUVAMA.NS',
    'OBEROIRLTY.NS',
    'ONGC.NS',
    'OIL.NS',
    'OLAELEC.NS',
    'OLECTRA.NS',
    'PAYTM.NS',
    'OFSS.NS',
    'POLICYBZR.NS',
    'PCBL.NS',
    'PGEL.NS',
    'PIIND.NS',
    'PNBHOUSING.NS',
    'PNCINFRA.NS',
    'PTCIL.NS',
    'PVRINOX.NS',
    'PAGEIND.NS',
    'PATANJALI.NS',
    'PERSISTENT.NS',
    'PETRONET.NS',
    'PFIZER.NS',
    'PHOENIXLTD.NS',
    'PIDILITIND.NS',
    'PEL.NS',
    'PPLPHARMA.NS',
    'POLYMED.NS',
    'POLYCAB.NS',
    'POONAWALLA.NS',
    'PFC.NS',
    'POWERGRID.NS',
    'PRAJIND.NS',
    'PREMIERENE.NS',
    'PRESTIGE.NS',
    'PNB.NS',
    'RRKABEL.NS',
    'RBLBANK.NS',
    'RECLTD.NS',
    'RHIM.NS',
    'RITES.NS',
    'RADICO.NS',
    'RVNL.NS',
    'RAILTEL.NS',
    'RAINBOW.NS',
    'RKFORGE.NS',
    'RCF.NS',
    'RTNINDIA.NS',
    'RAYMONDLSL.NS',
    'RAYMOND.NS',
    'REDINGTON.NS',
    'RELIANCE.NS',
    'RPOWER.NS',
    'ROUTE.NS',
    'SBFC.NS',
    'SBICARD.NS',
    'SBILIFE.NS',
    'SJVN.NS',
    'SKFINDIA.NS',
    'SRF.NS',
    'SAGILITY.NS',
    'SAILIFE.NS',
    'SAMMAANCAP.NS',
    'MOTHERSON.NS',
    'SAPPHIRE.NS',
    'SARDAEN.NS',
    'SAREGAMA.NS',
    'SCHAEFFLER.NS',
    'SCHNEIDER.NS',
    'SCI.NS',
    'SHREECEM.NS',
    'RENUKA.NS',
    'SHRIRAMFIN.NS',
    'SHYAMMETL.NS',
    'SIEMENS.NS',
    'SIGNATURE.NS',
    'SOBHA.NS',
    'SOLARINDS.NS',
    'SONACOMS.NS',
    'SONATSOFTW.NS',
    'STARHEALTH.NS',
    'SBIN.NS',
    'SAIL.NS',
    'SWSOLAR.NS',
    'SUMICHEM.NS',
    'SUNPHARMA.NS',
    'SUNTV.NS',
    'SUNDARMFIN.NS',
    'SUNDRMFAST.NS',
    'SUPREMEIND.NS',
    'SUZLON.NS',
    'SWANENERGY.NS',
    'SWIGGY.NS',
    'SYNGENE.NS',
    'SYRMA.NS',
    'TBOTEK.NS',
    'TVSMOTOR.NS',
    'TANLA.NS',
    'TATACHEM.NS',
    'TATACOMM.NS',
    'TCS.NS',
    'TATACONSUM.NS',
    'TATAELXSI.NS',
    'TATAINVEST.NS',
    'TATAMOTORS.NS',
    'TATAPOWER.NS',
    'TATASTEEL.NS',
    'TATATECH.NS',
    'TTML.NS',
    'TECHM.NS',
    'TECHNOE.NS',
    'TEJASNET.NS',
    'NIACL.NS',
    'RAMCOCEM.NS',
    'THERMAX.NS',
    'TIMKEN.NS',
    'TITAGARH.NS',
    'TITAN.NS',
    'TORNTPHARM.NS',
    'TORNTPOWER.NS',
    'TARIL.NS',
    'TRENT.NS',
    'TRIDENT.NS',
    'TRIVENI.NS',
    'TRITURBINE.NS',
    'TIINDIA.NS',
    'UCOBANK.NS',
    'UNOMINDA.NS',
    'UPL.NS',
    'UTIAMC.NS',
    'ULTRACEMCO.NS',
    'UNIONBANK.NS',
    'UBL.NS',
    'UNITDSPR.NS',
    'USHAMART.NS',
    'VGUARD.NS',
    'DBREALTY.NS',
    'VTL.NS',
    'VBL.NS',
    'MANYAVAR.NS',
    'VEDL.NS',
    'VIJAYA.NS',
    'VMM.NS',
    'IDEA.NS',
    'VOLTAS.NS',
    'WAAREEENER.NS',
    'WELCORP.NS',
    'WELSPUNLIV.NS',
    'WESTLIFE.NS',
    'WHIRLPOOL.NS',
    'WIPRO.NS',
    'WOCKPHARMA.NS',
    'YESBANK.NS',
    'ZFCVINDIA.NS',
    'ZEEL.NS',
    'ZENTEC.NS',
    'ZENSARTECH.NS',
    'ZYDUSLIFE.NS',
    'ECLERX.NS',
]
selected_stocks = st.multiselect("Select Stocks to Scan", nifty50_stocks, default=nifty50_stocks)

end_date = datetime.today()
start_date = end_date - timedelta(days=200)

scan_crossover = st.checkbox("Scan for Golden/Death Cross", value=True)
scan_pullback = st.checkbox("Scan for Pullback to EMA20 (Buy the Dip)", value=True)
scan_macd_divergence = st.checkbox("Scan for MACD Divergence (Bullish/Bearish)", value=True)


# ------------------------------------
# Main Scan Logic
# ------------------------------------
signals = []
pullback_signals = []

if st.button("🔍 Run Scan"):
    progress_bar = st.progress(0, text="Starting scan...")

    for i, stock in enumerate(selected_stocks):
        try:
            df = yf.download(stock, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 60:
                st.warning(f"{stock}: Not enough data")
                continue

            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            #df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
            df["RSI"] = RSIIndicator(close=df["Close"].squeeze(), window=14).rsi()


            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # ------------- EMA Crossover -------------
            if scan_crossover:
                prev_ema20 = prev["EMA20"]
                prev_ema50 = prev["EMA50"]
                latest_ema20 = latest["EMA20"]
                latest_ema50 = latest["EMA50"]

                if prev_ema20 < prev_ema50 and latest_ema20 > latest_ema50:
                    signals.append({"Stock": stock, "Signal": "📈 Golden Cross"})

                elif prev_ema20 > prev_ema50 and latest_ema20 < latest_ema50:
                    signals.append({"Stock": stock, "Signal": "📉 Death Cross"})

            # ------------- Buy the Dip -------------
            if scan_pullback:
                # Ensure you get individual rows
                prev = df.iloc[-2]
                latest = df.iloc[-1]
                
                # Convert to scalar values (float)
                prev_close = float(prev["Close"])
                prev_open = float(prev["Open"])
                latest_close = float(latest["Close"])
                latest_open = float(latest["Open"])
                latest_low = float(latest["Low"])
                latest_ema20 = float(latest["EMA20"])
                latest_ema50 = float(latest["EMA50"])
                latest_rsi = float(latest["RSI"])
                
                # --- Conditions ---
                
                # Condition 1: Uptrend
                in_uptrend = latest_close > latest_ema20 and latest_ema20 > latest_ema50
                
                # Condition 2: Close near EMA20
                near_ema20 = abs(latest_close - latest_ema20) / latest_close < 0.01
                
                # Condition 3: Reversal candle pattern
                is_bullish_engulfing = (
                    prev_close < prev_open and
                    latest_close > latest_open and
                    latest_close > prev_open and
                    latest_open < prev_close
                )
                
                is_hammer = (
                    latest_close > latest_open and
                    (latest_open - latest_low) > 2 * (latest_close - latest_open)
                )
                
                # Condition 4: RSI > 40
                rsi_ok = latest_rsi > 40
                
                # Final condition
                if in_uptrend and near_ema20 and rsi_ok and (is_bullish_engulfing or is_hammer):
                    pullback_signals.append({"Stock": stock, "Signal": "🟢 Pullback Buy"})

        # ------------- MACD Divergence Detection -------------
            if scan_macd_divergence:
                recent = df.dropna(subset=["MACD"]).iloc[-30:]

                if len(recent) < 10:
                    raise ValueError("Not enough MACD data")
                
                closes = recent["Close"].values
                macds = recent["MACD"].values
                #closes = recent["Close"].values
                #macds = recent["MACD"].values
                #df["MACD"] = MACD(close=df["Close"]).macd()
                macd_calc = MACD(close=df["Close"], window_fast=12, window_slow=26, window_sign=9)
                df["MACD"] = macd_calc.macd()
                df = df.dropna(subset=["EMA20", "EMA50", "RSI", "MACD"])



                def find_extreme(values, mode="min"):
                    idx = None
                    val = None
                    for i in range(1, len(values)-1):
                        if mode == "min" and values[i] < values[i-1] and values[i] < values[i+1]:
                            if idx is None or values[i] < val:
                                idx = i
                                val = values[i]
                        elif mode == "max" and values[i] > values[i-1] and values[i] > values[i+1]:
                            if idx is None or values[i] > val:
                                idx = i
                                val = values[i]
                    return idx, val

                # Bullish divergence
                low1_idx, low1_price = find_extreme(closes, "min")
                low2_idx, low2_price = find_extreme(closes[::-1], "min")
                low2_idx = len(closes) - 1 - low2_idx if low2_idx is not None else None

                if low1_idx is not None and low2_idx is not None and low2_idx > low1_idx:
                    price_diff = closes[low2_idx] - closes[low1_idx]
                    macd_diff = macds[low2_idx] - macds[low1_idx]
                    if price_diff < 0 and macd_diff > 0:
                        signals.append({"Stock": stock, "Signal": "📉 MACD Bullish Divergence"})

                # Bearish divergence
                high1_idx, high1_price = find_extreme(closes, "max")
                high2_idx, high2_price = find_extreme(closes[::-1], "max")
                high2_idx = len(closes) - 1 - high2_idx if high2_idx is not None else None

                if high1_idx is not None and high2_idx is not None and high2_idx > high1_idx:
                    price_diff = closes[high2_idx] - closes[high1_idx]
                    macd_diff = macds[high2_idx] - macds[high1_idx]
                    if price_diff > 0 and macd_diff < 0:
                        signals.append({"Stock": stock, "Signal": "📈 MACD Bearish Divergence"})





        

        



        except Exception as e:
            st.error(f"Error with {stock}: {e}")

        progress_bar.progress((i + 1) / len(selected_stocks), text=f"Scanning {stock}...")

    progress_bar.empty()

    # ----------------- Results -----------------
    if scan_crossover:
        st.subheader("📊 EMA Crossover Results")
        if signals:
            st.dataframe(pd.DataFrame(signals))
        else:
            st.info("No EMA crossover signals found.")

    if scan_crossover or scan_macd_divergence:
        st.subheader("📊 EMA & MACD Signal Results")
        if signals:
            st.dataframe(pd.DataFrame(signals))
        else:
            st.info("No EMA or MACD signals found.")

    if scan_pullback:
        st.subheader("📉 Pullback to 20 EMA (Buy the Dip)")
        if pullback_signals:
            st.dataframe(pd.DataFrame(pullback_signals))
        else:
            st.info("No pullback buy signals found.")
