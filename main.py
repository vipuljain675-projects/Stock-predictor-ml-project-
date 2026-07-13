import os
import pickle
import json
import urllib.request
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai

# Load env variables
load_dotenv()

app = FastAPI(title="Aegis Markets API")

# Enable CORS for Next.js development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DATA PIPELINE FUNCTIONS -----------------

def fetch_live_stock_data(ticker: str) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    try:
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) > 0:
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.dropna(inplace=True)
            return data
    except Exception:
        pass
    return pd.DataFrame()

def fetch_live_rss_news() -> tuple:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    sources = [
        ("Al Jazeera",   "https://www.aljazeera.com/xml/rss/all.xml"),
        ("BBC",          "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("GNews Energy", "https://news.google.com/rss/search?q=crude+oil+OR+Brent+OR+OPEC+OR+Hormuz+OR+Iran+sanctions&hl=en-US&gl=US&ceid=US:en"),
        ("GNews Geo",    "https://news.google.com/rss/search?q=geopolitics+OR+military+conflict+OR+sanctions+OR+war&hl=en-US&gl=US&ceid=US:en"),
        ("GNews Market", "https://news.google.com/rss/search?q=oil+market+OR+energy+prices+OR+Fed+rates+OR+inflation&hl=en-US&gl=US&ceid=US:en"),
        ("GNews India",  "https://news.google.com/rss/search?q=Nifty+OR+RBI+OR+India+economy+OR+rupee&hl=en-IN&gl=IN&ceid=IN:en"),
    ]

    headlines = []
    analyzer = SentimentIntensityAnalyzer()

    for source_name, url in sources:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml, */*'
                }
            )
            with urllib.request.urlopen(req, timeout=8, context=ssl_ctx) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:15]:
                title_el = item.find('title')
                if title_el is None or not title_el.text:
                    continue
                title = title_el.text.strip()
                if len(title) < 15:
                    continue
                pub_el = item.find('pubDate')
                try:
                    pub_date = pub_el.text if (pub_el is not None and pub_el.text is not None) else ''
                    date_parsed = datetime.strptime(pub_date[:25].strip(), "%a, %d %b %Y %H:%M:%S")
                except:
                    date_parsed = datetime.now()
                sentiment = analyzer.polarity_scores(title)['compound']
                headlines.append({
                    "date": date_parsed.strftime('%Y-%m-%d'),
                    "datetime": date_parsed,
                    "title": title,
                    "sentiment": sentiment,
                    "source": source_name
                })
        except Exception:
            continue

    if not headlines:
        return pd.DataFrame(columns=['date', 'datetime', 'title', 'sentiment', 'source']), False

    df = pd.DataFrame(headlines)
    df = df.sort_values('datetime', ascending=False)
    df = df.drop_duplicates(subset='title')
    df = df.head(60)
    return df, True

def fetch_news_via_gemini(api_key: str) -> tuple:
    if not api_key:
        return pd.DataFrame(columns=['date', 'datetime', 'title', 'sentiment', 'source']), False
    try:
        from google.genai import types
        client = genai.Client(api_key=api_key)
        prompt = """Use Google Search to find the 25 most recent news headlines from the last 48 hours about:
- Brent crude oil prices, OPEC+ decisions, oil supply disruptions
- Strait of Hormuz, Iran, IRGC, tanker seizures, Middle East conflict
- Russia energy exports, Ukraine war impact on commodities
- US Federal Reserve, dollar index DXY, inflation data
- Global stock markets, China economy, trade wars
- India economy, RBI policy, Nifty market

Return ONLY the raw headlines, one per line. No numbering, no bullets, no commentary, no sources. Just the headlines."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        analyzer = SentimentIntensityAnalyzer()
        headlines = []
        now = datetime.now()
        for line in response.text.strip().split('\n'):
            line = line.strip().lstrip('-').lstrip('*').lstrip('•').strip()
            import re
            line = re.sub(r'^\d+[.)\s]+', '', line).strip()
            if len(line) > 20 and not line.startswith('http'):
                sentiment = analyzer.polarity_scores(line)['compound']
                headlines.append({
                    'date': now.strftime('%Y-%m-%d'),
                    'datetime': now,
                    'title': line,
                    'sentiment': sentiment,
                    'source': 'Gemini+Search'
                })
        if headlines:
            return pd.DataFrame(headlines), True
    except Exception:
        pass
    return pd.DataFrame(columns=['date', 'datetime', 'title', 'sentiment', 'source']), False

def fetch_macro_indicators(ticker_symbol: str) -> dict:
    if ticker_symbol == "^NSEI":
        signals = {
            "USD_INR":   {"ticker": "USDINR=X",  "label": "USD/INR Rate",  "unit": "₹", "inverse_good": True},
            "India_VIX": {"ticker": "^INDIAVIX", "label": "India VIX",     "unit": "",  "inverse_good": True},
            "Gold":      {"ticker": "GC=F",       "label": "Gold ($/oz)",  "unit": "$", "inverse_good": False},
            "Crude_Oil": {"ticker": "BZ=F",       "label": "Brent Crude",  "unit": "$", "inverse_good": True},
            "SP500":     {"ticker": "^GSPC",      "label": "S&P 500",     "unit": "$", "inverse_good": False},
        }
    else:
        signals = {
            "DXY":    {"ticker": "DX-Y.NYB", "label": "US Dollar Index", "unit": "",  "inverse_good": True},
            "Gold":   {"ticker": "GC=F",     "label": "Gold ($/oz)",     "unit": "$", "inverse_good": False},
            "SP500":  {"ticker": "^GSPC",    "label": "S&P 500",         "unit": "$", "inverse_good": False},
            "NatGas": {"ticker": "NG=F",     "label": "Natural Gas",     "unit": "$", "inverse_good": False},
            "VIX":    {"ticker": "^VIX",     "label": "VIX (Fear Index)","unit": "",  "inverse_good": True},
        }

    results = {}
    end = datetime.now()
    start = end - timedelta(days=60)
    for key, sig in signals.items():
        try:
            df = yf.download(sig["ticker"], start=start.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and 'Close' in df.columns:
                close = df['Close'].dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                latest = float(close.iloc[-1])
                prev_30 = float(close.iloc[0]) if len(close) > 1 else latest
                change_pct = ((latest - prev_30) / prev_30) * 100
                results[key] = {
                    "label": sig["label"],
                    "unit": sig["unit"],
                    "value": latest,
                    "change_pct": change_pct,
                    "trend": "up" if change_pct > 0 else "down",
                    "inverse_good": sig["inverse_good"],
                    "series": close
                }
        except Exception:
            continue
    return results

def compute_aegis_macro_score(macro_data: dict, ticker_symbol: str) -> float:
    if ticker_symbol == "^NSEI":
        weights = {
            "USD_INR":   (0.20, False),
            "India_VIX": (0.20, False),
            "Gold":      (0.10, False),
            "Crude_Oil": (0.25, False),
            "SP500":     (0.25, True),
        }
    else:
        weights = {
            "DXY":    (0.25, False),
            "Gold":   (0.15, True),
            "SP500":  (0.20, True),
            "NatGas": (0.15, True),
            "VIX":    (0.25, False),
        }
    score = 0.0
    weight_total = 0.0
    for key, (weight, good_when_rising) in weights.items():
        if key in macro_data:
            chg = macro_data[key]["change_pct"]
            normalized = np.clip(chg / 5.0, -1, 1)
            signal = normalized if good_when_rising else -normalized
            score += signal * weight
            weight_total += weight
    if weight_total > 0:
        score = score / weight_total
    return float(np.clip(score, -1, 1))

def detect_market_shocks(news_df: pd.DataFrame, ticker_symbol: str) -> Optional[dict]:
    SUPPLY_SHOCK_KEYWORDS = [
        'hormuz', 'strait of hormuz', 'blockade', 'tanker attack', 'tanker seized',
        'oil field attack', 'pipeline attack', 'refinery strike', 'shipping disruption',
        'naval blockade', 'production cut', 'supply disruption', 'chokepoint',
        'opec cuts output', 'strait closure'
    ]
    DE_ESCALATION_KEYWORDS = [
        'ceasefire', 'peace talks', 'diplomatic breakthrough', 'de-escalate',
        'de-escalation', 'truce', 'sanctions lifted', 'sanctions eased',
        'nuclear deal', 'agreement reached', 'tensions ease', 'stability restored',
        'opec increases output', 'production increase'
    ]
    DEMAND_SHOCK_KEYWORDS = [
        'recession', 'economic slowdown', 'demand destruction', 'gdp contracts',
        'layoffs surge', 'manufacturing slumps', 'growth stalls', 'global slowdown'
    ]

    recent = news_df.sort_values('date', ascending=False).head(15)

    for _, row in recent.iterrows():
        title_lower = row['title'].lower()
        if any(kw in title_lower for kw in SUPPLY_SHOCK_KEYWORDS):
            return {
                'type': 'Supply shock', 'headline': row['title'],
                'crude_dir': 'up', 'nifty_dir': 'down',
                'reasoning': 'Supply-side disruption → oil supply risk rises → crude up, costlier imports drag Nifty down'
            }
    for _, row in recent.iterrows():
        title_lower = row['title'].lower()
        if any(kw in title_lower for kw in DE_ESCALATION_KEYWORDS):
            return {
                'type': 'De-escalation', 'headline': row['title'],
                'crude_dir': 'down', 'nifty_dir': 'up',
                'reasoning': 'Tension easing → supply risk removed → crude down, cheaper imports lift Nifty up'
            }
    for _, row in recent.iterrows():
        title_lower = row['title'].lower()
        if any(kw in title_lower for kw in DEMAND_SHOCK_KEYWORDS):
            return {
                'type': 'Demand shock', 'headline': row['title'],
                'crude_dir': 'down', 'nifty_dir': 'down',
                'reasoning': 'Recession/slowdown fears → weaker demand outlook → both crude and Nifty pressured down'
            }
    return None

def get_asset_impact(title: str, ticker_symbol: str):
    t = title.lower()
    if ticker_symbol == "^NSEI":
        bullish = [
            ("hormuz open",        "Hormuz open → crude stable → ✓ Nifty"),
            ("strait open",        "Strait open → no supply shock → ✓ Nifty"),
            ("ceasefire",          "Ceasefire → risk-on globally → ✓ Nifty"),
            ("peace deal",         "Peace → risk premium falls → ✓ Nifty"),
            ("rate cut",           "Rate cut → liquidity surge → FII inflow → ✓ Nifty"),
            ("fed pivot",          "Fed dovish → EM inflows → ✓ Nifty"),
            ("fed rate",           "Fed dovish signal → ✓ Nifty"),
            ("oil falls",          "Oil cheaper → India import costs ↓ → ✓ Nifty"),
            ("oil drops",          "Oil dropping → India CAD improves → ✓ Nifty"),
            ("crude falls",        "Crude down → India benefits → ✓ Nifty"),
            ("crude drops",        "Crude dropping → ✓ Nifty"),
            ("supply restored",    "Supply restored → crude stable → ✓ Nifty"),
        ]
        bearish = [
            ("hormuz block",       "Hormuz blocked → crude spike → India import bill ↑ → ✗ Nifty"),
            ("hormuz shut",        "Hormuz shut → crude spike → ✗ Nifty"),
            ("hormuz close",       "Hormuz closing → crude ↑ → India pays more → ✗ Nifty"),
            ("tanker seiz",        "Tanker seizure → supply risk → crude ↑ → ✗ Nifty"),
            ("tanker hit",         "Tanker attack → supply risk → crude ↑ → ✗ Nifty"),
            ("irgc",               "IRGC action → Hormuz risk → crude ↑ → ✗ Nifty"),
            ("oil spike",          "Oil spike → India CAD widens → ✗ Nifty"),
            ("crude surges",       "Crude surge → India import costs ↑ → ✗ Nifty"),
            ("crude rises",        "Crude rising → India macro pressure → ✗ Nifty"),
            ("oil rises",          "Oil rising → India import bill ↑ → ✗ Nifty"),
            ("rate hike",          "Rate hike → liquidity tightens → ✗ Nifty"),
            ("recession",          "Recession fear → FII flee EM → ✗ Nifty"),
        ]
    else:
        bullish = [
            ("hormuz block",       "Hormuz blocked → supply crisis → ↑ Crude"),
            ("hormuz shut",        "Hormuz shut → supply crunch → ↑ Crude"),
            ("hormuz close",       "Hormuz closing → supply risk → ↑ Crude"),
            ("tanker seiz",        "Tanker seized → supply risk premium → ↑ Crude"),
            ("tanker hit",         "Tanker attack → supply disruption → ↑ Crude"),
            ("irgc",               "IRGC activity → Hormuz risk → ↑ Crude"),
            ("opec cut",           "OPEC production cut → supply tightens → ↑ Crude"),
            ("production cut",     "Output cut → supply deficit → ↑ Crude"),
            ("strikes with iran",  "Active Iran strikes → supply risk premium → ↑ Crude"),
        ]
        bearish = [
            ("hormuz open",        "Hormuz confirmed open → supply risk falls → ↓ Crude"),
            ("strait open",        "Strait open → no disruption premium → ↓ Crude"),
            ("ceasefire",          "Ceasefire → risk premium falls → ↓ Crude"),
            ("peace deal",         "Peace → supply risk lower → ↓ Crude"),
            ("opec increase",      "OPEC output increase → oversupply → ↓ Crude"),
            ("supply restored",    "Supply restored → market normalizes → ↓ Crude"),
            ("recession",          "Recession → demand destruction → ↓ Crude"),
        ]

    import re
    for pattern, reason in bullish:
        if re.search(pattern, t):
            return 'bullish', reason
    for pattern, reason in bearish:
        if re.search(pattern, t):
            return 'bearish', reason
    return 'neutral', ''

def load_ml_assets(ticker: str):
    clean_ticker = ticker.replace("^", "").replace("=", "")
    model_path  = f"lstm_model_{clean_ticker}.h5"
    scaler_path = f"scaler_{clean_ticker}.pkl"
    target_scaler_path = f"target_scaler_{clean_ticker}.pkl"
    config_path = f"feature_config_{clean_ticker}.json"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        model_path  = "lstm_model.h5"
        scaler_path = "scaler.pkl"
        target_scaler_path = None
        config_path = None

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            import tensorflow as tf
            @tf.keras.utils.register_keras_serializable()
            class SelfAttention(tf.keras.layers.Layer):
                def build(self, input_shape):
                    self.W = self.add_weight(name='attn_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform')
                    self.b = self.add_weight(name='attn_bias', shape=(input_shape[1], 1), initializer='zeros')
                    super().build(input_shape)
                def call(self, x):
                    e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
                    a = tf.nn.softmax(e, axis=1)
                    return tf.reduce_sum(x * a, axis=1)
            from tensorflow.keras.models import load_model
            model = load_model(model_path, custom_objects={'SelfAttention': SelfAttention})
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            target_scaler = None
            if target_scaler_path and os.path.exists(target_scaler_path):
                with open(target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)
            feature_config = None
            if config_path and os.path.exists(config_path):
                with open(config_path) as f:
                    feature_config = json.load(f)
            return model, scaler, target_scaler, False, feature_config
        except Exception:
            return None, None, None, True, None
    return None, None, None, True, None

def engineer_features(ohlcv_df, sentiment_series, macro_series_dict):
    df = ohlcv_df.copy()
    df['SMA_10']    = df['Close'].rolling(10).mean()
    df['SMA_20']    = df['Close'].rolling(20).mean()
    df['EMA_20']    = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50']    = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_Ratio'] = df['SMA_10'] / (df['SMA_20'] + 1e-9)

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    df['Momentum_5']  = df['Close'].pct_change(5)

    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + 2 * std20
    df['BB_Lower'] = df['SMA_20'] - 2 * std20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['SMA_20'] + 1e-9)

    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low']  - df['Close'].shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['Rolling_Vol'] = df['Close'].pct_change().rolling(10).std()

    obv = [0]
    close_vals = df['Close'].values
    vol_vals = df['Volume'].values
    for i in range(1, len(df)):
        if close_vals[i] > close_vals[i-1]:
            obv.append(obv[-1] + vol_vals[i])
        elif close_vals[i] < close_vals[i-1]:
            obv.append(obv[-1] - vol_vals[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = np.array(obv) / 1e7

    df = df.merge(sentiment_series.rename('Sentiment'), left_index=True, right_index=True, how='left')
    df['Sentiment'] = df['Sentiment'].fillna(0.0)

    for name, series in macro_series_dict.items():
        col_name = f'Macro_{name}_5d'
        pct = series.pct_change(5).rename(col_name)
        df = df.merge(pct, left_index=True, right_index=True, how='left')
        df[col_name] = df[col_name].fillna(0.0)

    return df

def run_7day_prediction(model, scaler, target_scaler, is_fallback, stock_df, news_df, macro_data, ticker_symbol="BZ=F", feature_config=None):
    if news_df.empty:
        daily_news = pd.Series(dtype=float)
    else:
        daily_news = news_df.groupby('date')['sentiment'].mean()
        daily_news.index = pd.to_datetime(daily_news.index)

    macro_series_dict = {name: sig['series'] for name, sig in macro_data.items()}

    feat_df = engineer_features(stock_df, daily_news, macro_series_dict)
    feat_df.dropna(inplace=True)

    last_price = float(stock_df['Close'].iloc[-1])
    last_date  = feat_df.index[-1] if len(feat_df) else stock_df.index[-1]
    current_sentiment = float(news_df['sentiment'].mean()) if not news_df.empty else 0.0

    future_dates = []
    current_date = last_date
    while len(future_dates) < 7:
        current_date += timedelta(days=1)
        future_dates.append(current_date)

    max_daily_change = 0.015 if ticker_symbol == "^NSEI" else 0.020

    FEATURES = feature_config.get('features') if feature_config else None
    LOOKBACK = feature_config.get('lookback', 30) if feature_config else 30

    model_usable = (
        not is_fallback and scaler is not None and target_scaler is not None and FEATURES is not None
        and len(feat_df) >= LOOKBACK + 1
        and all(f in feat_df.columns for f in FEATURES)
    )
    if model_usable:
        try:
            scaler_min_close = scaler.data_min_[0]
            scaler_max_close = scaler.data_max_[0]
            if last_price > scaler_max_close * 5 or last_price < scaler_min_close / 5:
                model_usable = False
        except Exception:
            model_usable = False

    if not model_usable:
        return last_price, None, None, None, None, None, False

    try:
        # Calculate recent daily volatility (historical returns standard deviation)
        returns = stock_df['Close'].pct_change().dropna()
        volatility = float(returns.std())
        if np.isnan(volatility) or volatility == 0:
            volatility = 0.008 if ticker_symbol == "^NSEI" else 0.015

        # Fetch shock info
        shock = detect_market_shocks(news_df, ticker_symbol)

        # Volatility-scaled Geopolitical Adjustment
        geopolitical_adjustment = 0.0
        if ticker_symbol == "BZ=F":
            # Brent Crude rises on geopolitical tension (negative sentiment)
            geopolitical_adjustment = -current_sentiment * volatility
            if shock:
                if shock['type'] == 'Supply shock':
                    geopolitical_adjustment += 1.5 * volatility
                elif shock['type'] == 'De-escalation':
                    geopolitical_adjustment -= 1.5 * volatility
        else:  # ^NSEI
            # Nifty falls on geopolitical tension (negative sentiment)
            geopolitical_adjustment = current_sentiment * 0.8 * volatility
            if shock:
                if shock['type'] == 'Supply shock':
                    geopolitical_adjustment -= 1.2 * volatility
                elif shock['type'] == 'De-escalation':
                    geopolitical_adjustment += 1.2 * volatility

        # 1. RAW ML MODEL FORECAST LOOP
        raw_predictions = []
        working_ohlcv_raw = stock_df.copy()
        working_macro_raw = {k: v.copy() for k, v in macro_series_dict.items()}
        working_sent_raw = daily_news.copy()

        for i in range(7):
            feats_now = engineer_features(working_ohlcv_raw, working_sent_raw, working_macro_raw).dropna()
            seq = feats_now[FEATURES].values[-LOOKBACK:]
            scaled_seq = scaler.transform(seq)
            X = np.expand_dims(scaled_seq, axis=0)

            pred_ret_scaled = model.predict(X, verbose=0)
            pred_return = float(target_scaler.inverse_transform(pred_ret_scaled)[0, 0])

            prev_price = float(working_ohlcv_raw['Close'].iloc[-1])
            capped_return = np.clip(pred_return, -max_daily_change, max_daily_change)
            pred_price = prev_price * (1.0 + capped_return)
            raw_predictions.append(pred_price)

            new_date = working_ohlcv_raw.index[-1] + timedelta(days=1)
            new_row = pd.DataFrame({
                'Open': [pred_price], 'High': [pred_price], 'Low': [pred_price],
                'Close': [pred_price], 'Volume': [float(working_ohlcv_raw['Volume'].iloc[-1])]
            }, index=[new_date])
            working_ohlcv_raw = pd.concat([working_ohlcv_raw, new_row])

            future_sentiment = current_sentiment * (0.70 ** (i + 1))
            working_sent_raw = pd.concat([working_sent_raw, pd.Series([future_sentiment], index=[new_date])])
            for k in working_macro_raw:
                last_val = working_macro_raw[k].iloc[-1]
                working_macro_raw[k] = pd.concat([working_macro_raw[k], pd.Series([last_val], index=[new_date])])

        # 2. GEOPOLITICAL-ADJUSTED FORECAST LOOP
        adj_predictions = []
        working_ohlcv_adj = stock_df.copy()
        working_macro_adj = {k: v.copy() for k, v in macro_series_dict.items()}
        working_sent_adj = daily_news.copy()

        for i in range(7):
            feats_now = engineer_features(working_ohlcv_adj, working_sent_adj, working_macro_adj).dropna()
            seq = feats_now[FEATURES].values[-LOOKBACK:]
            scaled_seq = scaler.transform(seq)
            X = np.expand_dims(scaled_seq, axis=0)

            pred_ret_scaled = model.predict(X, verbose=0)
            pred_return = float(target_scaler.inverse_transform(pred_ret_scaled)[0, 0])

            # Apply decayed geopolitical adjustment
            decayed_adj = geopolitical_adjustment * (0.75 ** i)
            adjusted_return = pred_return + decayed_adj

            prev_price = float(working_ohlcv_adj['Close'].iloc[-1])
            capped_return = np.clip(adjusted_return, -max_daily_change, max_daily_change)
            pred_price = prev_price * (1.0 + capped_return)
            adj_predictions.append(pred_price)

            new_date = working_ohlcv_adj.index[-1] + timedelta(days=1)
            new_row = pd.DataFrame({
                'Open': [pred_price], 'High': [pred_price], 'Low': [pred_price],
                'Close': [pred_price], 'Volume': [float(working_ohlcv_adj['Volume'].iloc[-1])]
            }, index=[new_date])
            working_ohlcv_adj = pd.concat([working_ohlcv_adj, new_row])

            future_sentiment = current_sentiment * (0.70 ** (i + 1))
            working_sent_adj = pd.concat([working_sent_adj, pd.Series([future_sentiment], index=[new_date])])
            for k in working_macro_adj:
                last_val = working_macro_adj[k].iloc[-1]
                working_macro_adj[k] = pd.concat([working_macro_adj[k], pd.Series([last_val], index=[new_date])])

        pred_df = pd.DataFrame({'Close': adj_predictions}, index=future_dates)
        raw_df = pd.DataFrame({'Close': raw_predictions}, index=future_dates)

        # Derive a real confidence score from the model's saved directional accuracy.
        # directional_accuracy is stored in feature_config as a percentage (e.g. 52.24).
        # We normalise it to [0, 1]. A model that calls direction correctly 50% of the time
        # is no better than a coin-flip, so we floor at 0.50 and scale to [0, 1].
        raw_dir_acc = feature_config.get("directional_accuracy", 50.0) if feature_config else 50.0
        confidence_score = float(np.clip(raw_dir_acc / 100.0, 0.0, 1.0))

        return last_price, adj_predictions[0], pred_df, raw_predictions[0], raw_df, confidence_score, True
    except Exception:
        return last_price, None, None, None, None, None, False

# ----------------- FASTAPI ROUTES -----------------

class ChatRequest(BaseModel):
    query: str
    ticker_symbol: str
    messages: List[Dict[str, str]]

@app.get("/api/market-data")
def get_market_data(symbol: str = Query(..., description="BZ=F or ^NSEI")):
    stock_df = fetch_live_stock_data(symbol)
    if stock_df.empty:
        raise HTTPException(status_code=500, detail="Failed to fetch stock history")

    # Fetch news
    api_key = os.environ.get("GEMINI_API_KEY", "")
    news_feed, news_is_live = fetch_news_via_gemini(api_key)
    news_source = "Gemini+Search"
    if not news_is_live:
        news_feed, news_is_live = fetch_live_rss_news()
        news_source = "RSS"

    # Fetch macro
    macro_data = fetch_macro_indicators(symbol)
    macro_score = compute_aegis_macro_score(macro_data, symbol)
    avg_sentiment = float(news_feed['sentiment'].mean()) if not news_feed.empty else 0.0

    # Load ML assets
    model, scaler, target_scaler, is_fallback, feature_config = load_ml_assets(symbol)

    # Predictions
    # Predictions
    last_close, next_predict, future_forecast_df, next_predict_raw, future_forecast_raw_df, confidence_score, model_usable = run_7day_prediction(
        model, scaler, target_scaler, is_fallback, stock_df, news_feed, macro_data, symbol, feature_config
    )

    # Shock detection
    shock = detect_market_shocks(news_feed, symbol)

    # Build response data structure
    historical_chart = []
    for date, row in stock_df.iterrows():
        historical_chart.append({
            "date": pd.to_datetime(date).strftime('%Y-%m-%d'),
            "close": float(row["Close"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
        })

    forecast_chart = []
    if model_usable and future_forecast_df is not None:
        for date, row in future_forecast_df.iterrows():
            forecast_chart.append({
                "date": pd.to_datetime(date).strftime('%Y-%m-%d'),
                "close": float(row["Close"])
            })

    forecast_chart_raw = []
    if model_usable and future_forecast_raw_df is not None:
        for date, row in future_forecast_raw_df.iterrows():
            forecast_chart_raw.append({
                "date": pd.to_datetime(date).strftime('%Y-%m-%d'),
                "close": float(row["Close"])
            })

    news_list = []
    if not news_feed.empty:
        for _, row in news_feed.iterrows():
            impact, impact_reason = get_asset_impact(row["title"], symbol)
            news_list.append({
                "date": row["date"],
                "title": row["title"],
                "sentiment": float(row["sentiment"]),
                "source": row["source"],
                "impact": impact,
                "impact_reason": impact_reason
            })

    macro_list = []
    for key, sig in macro_data.items():
        macro_list.append({
            "key": key,
            "label": sig["label"],
            "unit": sig["unit"],
            "value": float(sig["value"]),
            "change_pct": float(sig["change_pct"]),
            "trend": sig["trend"]
        })

    model_metadata = None
    if model_usable and feature_config:
        model_metadata = {
            "mape": feature_config.get("mape"),
            "directional_accuracy": feature_config.get("directional_accuracy"),
            "n_features": feature_config.get("n_features"),
            "lookback": feature_config.get("lookback")
        }

    return {
        "ticker_symbol": symbol,
        "asset_name": "Brent Crude Oil" if symbol == "BZ=F" else "Nifty 50",
        "last_close": last_close,
        "next_predict": next_predict,
        "next_predict_raw": next_predict_raw,
        "change_pct": ((next_predict - last_close) / last_close * 100) if next_predict else None,
        "change_pct_raw": ((next_predict_raw - last_close) / last_close * 100) if next_predict_raw else None,
        "avg_sentiment": avg_sentiment,
        "macro_score": macro_score,
        "confidence_score": confidence_score,
        "model_usable": model_usable,
        "model_metadata": model_metadata,
        "news_source": news_source,
        "news_is_live": news_is_live,
        "shock": shock,
        "historical_chart": historical_chart,
        "forecast_chart": forecast_chart,
        "forecast_chart_raw": forecast_chart_raw,
        "news": news_list,
        "macro": macro_list
    }

@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is not configured in backend .env")

    # Fetch context to construct prompt
    stock_df = fetch_live_stock_data(req.ticker_symbol)
    if stock_df.empty:
        raise HTTPException(status_code=500, detail="Failed to fetch stock history for context")

    news_feed, news_is_live = fetch_news_via_gemini(api_key)
    if not news_is_live:
        news_feed, _ = fetch_live_rss_news()

    macro_data = fetch_macro_indicators(req.ticker_symbol)
    macro_score = compute_aegis_macro_score(macro_data, req.ticker_symbol)
    avg_sentiment = float(news_feed['sentiment'].mean()) if not news_feed.empty else 0.0

    model, scaler, target_scaler, is_fallback, feature_config = load_ml_assets(req.ticker_symbol)
    last_close, next_predict, _, _, _, _, model_usable = run_7day_prediction(
        model, scaler, target_scaler, is_fallback, stock_df, news_feed, macro_data, req.ticker_symbol, feature_config
    )

    headline_list = "\n".join([f"- {row['title']} (Sentiment: {row['sentiment']:.2f})" for _, row in news_feed.head(5).iterrows()]) if not news_feed.empty else "No recent headlines available"
    macro_context = "\n".join([f"- {sig['label']}: {sig['value']:.2f} (30d change: {sig['change_pct']:+.2f}%)" for sig in macro_data.values()]) if macro_data else "Macro indicators unavailable"

    change_pct = ((next_predict - last_close) / last_close * 100) if next_predict else 0.0

    context_prompt = f"""
    You are 'Aegis Intel', a world-class geopolitical analyst and quantitative financial advisor specializing in Indian and global markets.
    Your job is to provide specific, data-driven answers to user queries regarding {"Brent Crude Oil" if req.ticker_symbol == "BZ=F" else "Nifty 50"} ({req.ticker_symbol}).
    
    Current Market Context:
    - Asset: {"Brent Crude Oil" if req.ticker_symbol == "BZ=F" else "Nifty 50"} ({req.ticker_symbol})
    - Current Price: {last_close:,.2f}
    - LSTM Predicted Change in 24h: {change_pct:+.2f}%
    - Geopolitical Sentiment: {avg_sentiment:.2f} (range -1 tension to +1 peace)
    - Aegis Composite Macro Score: {macro_score*100:+.1f}/100
    
    Live Macro Indicators (30-day trend):
    {macro_context}
    
    Recent News Headlines:
    {headline_list}
    
    Instructions:
    - Use macro context (USD/INR, VIX, crude, yields) in your reasoning, especially for India/Nifty questions.
    - Be specific about numbers, correlations, and causality chains.
    - Explain the difference between supply-driven and demand-driven crude moves for India context.
    - Respond in a professional, analytical tone. Be concise but insightful.
    
    User Query: {req.query}
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=context_prompt
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")
