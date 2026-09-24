from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from common import ask_llama, read_table, ollama_available

st.set_page_config(page_title="AI Stock Forecasting & Market Analytics", page_icon="📈", layout="wide")
st.title("📈 AI Stock Forecasting & Market Analytics")
st.caption("Ticker-based forecasting, technical indicators, model comparison, upload support, and Llama-assisted interpretation.")

@st.cache_data(ttl=900)
def fetch_market_data(ticker: str, period: str):
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No market data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def rsi(series, window=14):
    delta = series.diff(); gain = delta.clip(lower=0).rolling(window).mean(); loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100/(1+rs))

def prepare_features(df: pd.DataFrame, horizon: int):
    d=df.copy()
    if "date" not in d.columns:
        for c in ["datetime","timestamp"]:
            if c in d.columns: d=d.rename(columns={c:"date"})
    if "close" not in d.columns: raise ValueError("Dataset requires a 'close' column.")
    if "volume" not in d.columns: d["volume"]=0.0
    d["date"]=pd.to_datetime(d["date"], errors="coerce") if "date" in d.columns else pd.RangeIndex(len(d))
    d=d.sort_values("date").reset_index(drop=True)
    d["return_1d"]=d.close.pct_change(); d["sma_5"]=d.close.rolling(5).mean(); d["sma_20"]=d.close.rolling(20).mean()
    d["ema_12"]=d.close.ewm(span=12,adjust=False).mean(); d["ema_26"]=d.close.ewm(span=26,adjust=False).mean(); d["macd"]=d.ema_12-d.ema_26
    d["volatility_20"]=d.return_1d.rolling(20).std(); d["rsi_14"]=rsi(d.close,14)
    d["volume_change"]=d.volume.pct_change().replace([np.inf,-np.inf],np.nan)
    for lag in [1,2,5,10]: d[f"close_lag_{lag}"]=d.close.shift(lag)
    d["target"]=d.close.shift(-horizon)
    d["future_date"]=d.date.shift(-horizon)
    features=["close","volume","return_1d","sma_5","sma_20","macd","volatility_20","rsi_14","volume_change","close_lag_1","close_lag_2","close_lag_5","close_lag_10"]
    return d,features

def evaluate_models(df, horizon):
    feat,features=prepare_features(df,horizon)
    trainable=feat.dropna(subset=features+["target"]).copy()
    if len(trainable)<80: raise ValueError("Need at least ~100 observations for a useful temporal evaluation.")
    cut=int(len(trainable)*.8); tr=trainable.iloc[:cut]; te=trainable.iloc[cut:]
    Xtr,ytr=tr[features],tr.target; Xte,yte=te[features],te.target
    models={
        "Ridge":Pipeline([("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]),
        "Random Forest":RandomForestRegressor(n_estimators=220,max_depth=10,min_samples_leaf=3,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=180,max_depth=3,learning_rate=.04,random_state=42),
    }
    rows=[]; fitted={}; preds={}
    naive=te["close"].to_numpy()
    rows.append({"Model":"Naive current-price baseline","MAE":mean_absolute_error(yte,naive),"RMSE":mean_squared_error(yte,naive)**.5,
                 "Directional Accuracy":np.mean((naive>te.close.values)==(yte.values>te.close.values))})
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); fitted[name]=m; preds[name]=p
        rows.append({"Model":name,"MAE":mean_absolute_error(yte,p),"RMSE":mean_squared_error(yte,p)**.5,
                     "Directional Accuracy":np.mean((p>te.close.values)==(yte.values>te.close.values))})
    metrics=pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_ml=min(models.keys(), key=lambda n: mean_absolute_error(yte,preds[n]))
    # refit best ML on all labeled data
    fitted[best_ml].fit(trainable[features],trainable.target)
    latest=feat.dropna(subset=features).iloc[[-1]]
    forecast=float(fitted[best_ml].predict(latest[features])[0])
    return feat,metrics,best_ml,forecast,te,yte,preds

source=st.sidebar.radio("Data source",["Live ticker (Yahoo Finance)","Upload CSV / Excel"])
period=st.sidebar.selectbox("History",["1y","2y","5y","10y"],index=2)
horizon=st.sidebar.selectbox("Forecast horizon (trading days)",[1,5,10,30],index=1)

if source.startswith("Live"):
    ticker=st.sidebar.text_input("Ticker",value="AAPL").upper().strip()
    st.sidebar.caption("Examples: AAPL, MSFT, NVDA, GOOGL, RELIANCE.NS, TCS.NS, INFY.NS")
    try:
        raw=fetch_market_data(ticker,period)
        data_label=f"Yahoo Finance: {ticker}"
    except Exception as e:
        st.error(str(e)); st.stop()
else:
    up=st.sidebar.file_uploader("Upload historical market data",type=["csv","xlsx","xls"])
    if not up:
        st.info("Upload a file containing at least `date`, `close`, and optionally `volume`."); st.stop()
    raw=read_table(up); raw.columns=[str(c).lower().strip() for c in raw.columns]; ticker="Uploaded asset"; data_label=up.name

if len(raw)<100:
    st.warning("This dataset is small. Forecast metrics may be unstable.")

try:
    feat,metrics,best_ml,forecast,te,yte,preds=evaluate_models(raw,horizon)
except Exception as e:
    st.exception(e); st.stop()

latest=feat.dropna(subset=["close"]).iloc[-1]; current=float(latest.close); delta=(forecast/current-1)*100

with st.sidebar:
    st.divider(); st.write(f"**Dataset:** {data_label}"); st.write(f"**Rows:** {len(raw):,}")
    if "date" in raw.columns: st.write(f"**Range:** {pd.to_datetime(raw.date).min().date()} → {pd.to_datetime(raw.date).max().date()}")
    st.write(f"**Llama:** {'Connected' if ollama_available() else 'Ollama not detected'}")

tabs=st.tabs(["Forecast", "Technical Analytics", "Model Comparison", "Watchlist Comparison", "Upload Guidance", "About"])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric("Asset",ticker); b.metric("Latest close",f"{current:,.2f}"); c.metric(f"{horizon}-day forecast",f"{forecast:,.2f}",f"{delta:+.2f}%")
    d.metric("Best ML model",best_ml)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=raw["date"] if "date" in raw.columns else np.arange(len(raw)),y=raw["close"],name="Close"))
    fig.update_layout(title=f"{ticker} price history",xaxis_title="Date",yaxis_title="Close")
    st.plotly_chart(fig,use_container_width=True)
    facts={k:(None if pd.isna(latest.get(k,np.nan)) else float(latest.get(k))) for k in ["rsi_14","macd","sma_20","volatility_20","return_1d"]}
    prompt=f"""You are a market analytics assistant. Use only these computed facts. Asset={ticker}; current close={current:.4f}; horizon={horizon} trading days; forecast={forecast:.4f}; forecast change={delta:.2f}%; best ML model={best_ml}; indicators={facts}. Explain the result in 5 concise bullets, include uncertainty, and state that this is experimental decision support rather than investment advice."""
    if st.button("Generate Llama forecast explanation"):
        st.info(ask_llama(prompt))

with tabs[1]:
    cols=st.columns(4)
    cols[0].metric("RSI(14)","n/a" if pd.isna(latest.rsi_14) else f"{latest.rsi_14:.1f}")
    cols[1].metric("MACD","n/a" if pd.isna(latest.macd) else f"{latest.macd:.2f}")
    cols[2].metric("SMA20","n/a" if pd.isna(latest.sma_20) else f"{latest.sma_20:.2f}")
    cols[3].metric("20-day volatility","n/a" if pd.isna(latest.volatility_20) else f"{latest.volatility_20:.2%}")
    chart=go.Figure(); chart.add_trace(go.Scatter(x=feat.date,y=feat.close,name="Close")); chart.add_trace(go.Scatter(x=feat.date,y=feat.sma_5,name="SMA5")); chart.add_trace(go.Scatter(x=feat.date,y=feat.sma_20,name="SMA20")); chart.update_layout(title="Price and moving averages")
    st.plotly_chart(chart,use_container_width=True)

with tabs[2]:
    st.dataframe(metrics.style.format({"MAE":"{:.4f}","RMSE":"{:.4f}","Directional Accuracy":"{:.1%}"}),use_container_width=True,hide_index=True)
    best_row=metrics.iloc[0]
    if best_row.Model=="Naive current-price baseline":
        st.warning("The naïve baseline currently beats the trained ML models on MAE. Treat the forecast as experimental; do not present it as superior predictive performance.")
    else:
        st.success(f"{best_row.Model} currently has the lowest holdout MAE in this run.")
    st.caption("Temporal holdout is used: the newest 20% of labeled observations are kept for evaluation.")

with tabs[3]:
    st.subheader("Compare several stocks")
    watch_text=st.text_input("Tickers (comma-separated, max 5)",value="AAPL,MSFT,NVDA")
    st.caption("NSE examples can be included, e.g. RELIANCE.NS,TCS.NS,INFY.NS")
    if st.button("Run watchlist forecasts"):
        rows=[]
        for tk in [x.strip().upper() for x in watch_text.split(',') if x.strip()][:5]:
            try:
                d=fetch_market_data(tk,period)
                f,m,bm,fc,*_=evaluate_models(d,horizon)
                lc=float(f.dropna(subset=["close"]).iloc[-1].close)
                br=m.iloc[0]
                rows.append({"Ticker":tk,"Latest":lc,"Forecast":fc,"Forecast %":(fc/lc-1)*100,"Best ML":bm,"Lowest holdout MAE model":br.Model,"MAE":br.MAE})
            except Exception as exc:
                rows.append({"Ticker":tk,"Latest":np.nan,"Forecast":np.nan,"Forecast %":np.nan,"Best ML":"Error","Lowest holdout MAE model":str(exc),"MAE":np.nan})
        comp=pd.DataFrame(rows)
        st.dataframe(comp.style.format({"Latest":"{:.2f}","Forecast":"{:.2f}","Forecast %":"{:+.2f}%","MAE":"{:.4f}"}),use_container_width=True,hide_index=True)
        st.caption("This comparison is for research/demo use. Forecasts across different price scales should not be compared using raw MAE alone.")

with tabs[4]:
    st.subheader("Custom dataset schema")
    st.code("date,open,high,low,close,volume\n2026-01-02,100,103,99,102,1200000")
    st.write("Only `date` and `close` are mandatory. `volume` is recommended. The app engineers returns, moving averages, RSI, MACD, volatility, and lag features automatically.")

with tabs[5]:
    st.write("This PoC compares a naïve baseline with Ridge, Random Forest, and Gradient Boosting. The numerical forecast comes from ML models; Llama only explains computed evidence.")
    st.warning("Stock prices are noisy and regime-dependent. This application is a technical PoC, not a promise of future performance or investment advice.")
