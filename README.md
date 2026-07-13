# 🛡️ AEGIS MARKETS
### *Geopolitical Intelligence × Quantitative Finance × Deep Learning*

> **Real-time market intelligence for Brent Crude Oil & Nifty 50 — powered by a custom-trained Bidirectional LSTM + Self-Attention neural network, live macroeconomic signals, geopolitical shock detection, and AI-driven news analysis.**

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     BRENT CRUDE OIL  ×  NIFTY 50  ×  GEOPOLITICS           ║
║                                                              ║
║     BiLSTM → Self-Attention → 7-Day Forecast                ║
║     Live News → VADER NLP → Geopolitical Overlay            ║
║     Macro Signals → Weighted Composite Score                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

---

## 📑 Table of Contents

1. [What is Aegis Markets?](#-what-is-aegis-markets)
2. [The Big Picture — AI vs ML vs DL](#-the-big-picture--ai-vs-ml-vs-dl)
3. [Solo Examples — Can They Exist Alone?](#-solo-examples--can-they-exist-alone)
4. [Why Stocks Need Sequences — The Core Insight](#-why-stocks-need-sequences--the-core-insight)
5. [Why LSTM and Not Random Forest?](#-why-lstm-and-not-random-forest)
6. [Project Architecture](#️-project-architecture)
7. [The LSTM Model — Deep Dive](#-the-lstm-model--deep-dive)
8. [Training Data & Feature Engineering](#-training-data--feature-engineering)
9. [Why Google Colab?](#️-why-google-colab)
10. [The Geopolitical Intelligence Layer](#-the-geopolitical-intelligence-layer)
11. [The Big Design Decision — Why News Is NOT in Training](#-the-big-design-decision--why-news-is-not-in-training)
12. [Model Performance — Honest Numbers](#-model-performance--honest-numbers)
13. [What Paid News APIs Would Unlock](#-what-paid-news-apis-would-unlock)
14. [Tech Stack](#-tech-stack)
15. [Local Setup](#-local-setup)
16. [Project Structure](#-project-structure)
17. [FAQ — Everything Explained](#-faq--everything-explained)

---

## 🎯 What is Aegis Markets?

Aegis Markets is a full-stack financial intelligence platform that predicts the next 7 days of **Brent Crude Oil (BZ=F)** and **Nifty 50 (^NSEI)** price movement.

It doesn't just run a model — it fuses **three separate intelligence layers** into a single forecast:

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE INTELLIGENCE LAYERS                 │
│                                                             │
│  🧠 Layer 1: LSTM Neural Network                            │
│     └─ Predicts next-day return from 30 days of             │
│        technical + macro features                           │
│                                                             │
│  📡 Layer 2: Geopolitical NLP Engine                        │
│     └─ Scores live news for supply shocks,                  │
│        de-escalation events, demand collapses               │
│                                                             │
│  📊 Layer 3: Macro Signal Composite Score                   │
│     └─ Weighted score from USD/INR, VIX,                    │
│        Crude, Gold, S&P 500                                 │
└─────────────────────────────────────────────────────────────┘
```

The three layers are blended into a **dual forecast**:
- **Raw ML Prediction** — pure model output, no adjustments
- **Geopolitically-Adjusted Prediction** — model + live news overlay

---

## 🧩 The Big Picture — AI vs ML vs DL

> **This is the most important concept to understand before anything else.**

People use AI, ML, and DL as if they're three different things you need to "collect." They're not. They're **nested categories** — like Russian dolls.

```
╔══════════════════════════════════════════════════════╗
║                🤖 AI (Artificial Intelligence)        ║
║         "Any machine doing intelligent work"          ║
║                                                      ║
║    ╔══════════════════════════════════════════╗      ║
║    ║         📈 ML (Machine Learning)          ║      ║
║    ║    "System that learns from data,         ║      ║
║    ║     no manual rules needed"               ║      ║
║    ║                                           ║      ║
║    ║    ╔════════════════════════════════╗     ║      ║
║    ║    ║    🧠 DL (Deep Learning)       ║     ║      ║
║    ║    ║  "ML using neural networks     ║     ║      ║
║    ║    ║   with multiple layers"        ║     ║      ║
║    ║    ║                                ║     ║      ║
║    ║    ║     👉 OUR LSTM IS HERE        ║     ║      ║
║    ║    ╚════════════════════════════════╝     ║      ║
║    ╚══════════════════════════════════════════╝      ║
╚══════════════════════════════════════════════════════╝
```

**Key Rule:** DL ⊂ ML ⊂ AI

Whatever is DL is automatically also ML and also AI. You don't need to "combine" them separately — they're descriptions of the same thing from different angles.

### The Three Angles of Our LSTM Model

```
Same model. Three descriptions.

"TensorFlow se neural network banaya"       → DL ka angle
"7 saal ke data se seekha"                  → ML ka angle
"Market predictions karta hai intelligently" → AI ka angle
```

Think of it like one person:
- *"IIT se padha"* → Engineer
- *"Python likhta hai"* → Programmer
- *"Google mein kaam karta hai"* → Googler

Ek hi banda — teen alag descriptions. Waise hi ek hi model — teen alag labels.

---

## 🔍 Solo Examples — Can They Exist Alone?

### 🔵 Sirf AI (Without ML or DL) — Possible ✅

These are **rule-based systems**. A human programmer manually wrote every decision rule. No data learning, no neural network.

| Example | What It Does | Why AI but not ML/DL |
|---|---|---|
| **Chess engine (1990s)** | Evaluates millions of positions | Rules manually programmed — never learned from data |
| **Old spam filters** | `IF "WIN LOTTERY" in email → spam` | Hard-coded keyword rules |
| **Google Maps routing** | Finds shortest path via Dijkstra | Pure algorithm, no learning |
| **Thermostat** | `IF temp > 25° → turn AC on` | Simple if-else logic |
| **Expert Systems (1980s)** | Medical diagnosis via decision trees | Rules written by human experts |

```
Rule-Based AI Example:
IF oil_price > 100 AND VIX > 30:
    signal = "SELL"
ELIF oil_price < 70 AND VIX < 15:
    signal = "BUY"
ELSE:
    signal = "HOLD"

→ AI ✅ (intelligent decision)
→ ML ❌ (never learned from data — YOU wrote these numbers)
→ DL ❌ (no neural network)
```

---

### 🟡 Sirf ML (Without DL) — Possible ✅

These systems **learn from data** but use mathematical/statistical models — no neural networks.

| Algorithm | Real Use Case | Why ML but not DL |
|---|---|---|
| **Linear Regression** | House price prediction | Simple line fit — no neural network |
| **Random Forest** | Loan approval / fraud detection | 100s of decision trees — no neural network |
| **XGBoost** | Kaggle competitions, tabular data | Gradient boosting — no neural network |
| **SVM** | Medical diagnosis, spam | Mathematical boundary — no neural network |
| **K-Means** | Customer segmentation | Clustering algorithm — no neural network |
| **Naive Bayes** | Email spam filter | Probability math — no neural network |

```
Random Forest Example (Loan Approval):

Input: Age=28, Salary=50k, Credit=720, Loans=1

Tree 1: Age > 25? YES → Salary > 40k? YES → APPROVE
Tree 2: Credit > 700? YES → Loans < 3? YES → APPROVE
Tree 3: Salary > 30k? YES → Age > 22? YES → APPROVE
...100 trees vote...

Result: ✅ LOAN APPROVED

→ ML ✅ (learned from past loan data)
→ DL ❌ (no neural network layers)
→ AI ✅ (intelligent decision making)
```

**Random Forest aaj bhi heavily used hai** in banks, insurance, medical — because it's fast, explainable ("loan reject hua because credit score < 600"), and works well with small data.

---

### 🔴 Sirf DL (Without ML) — IMPOSSIBLE ❌

This **cannot exist**. Deep Learning, by definition, involves a neural network that *learns from data*. "Learning from data" is exactly what Machine Learning means.

```
DL = Neural Network + Learning from Data
                           ↑
               This IS the definition of ML

∴ DL without ML = Impossible
∴ DL ⊂ ML — always, no exceptions
```

---

### Summary Table

```
Type        │ Data Learning │ Neural Network │ Can Exist Solo?
────────────┼───────────────┼────────────────┼────────────────
AI only     │      ❌       │       ❌       │      ✅ Yes
ML only     │      ✅       │       ❌       │      ✅ Yes
DL          │      ✅       │       ✅       │ Always ML too
```

---

## 📈 Why Stocks Need Sequences — The Core Insight

> **This is why we used DL (LSTM) and not just ML (Random Forest) — and it's the most important architectural decision in this project.**

### The Problem with Solo ML for Stocks

A Random Forest sees each data point as **independent**. Feed it today's RSI, today's MACD, today's VIX — it predicts. But it has no memory of yesterday, or the day before.

```
Day 1:  RSI=70, Close=100
Day 2:  RSI=65, Close=98
Day 3:  RSI=55, Close=95    ← teen din se gir raha hai
Day 4:  RSI=42, Close=91
Day 5:  ????

Random Forest ko puchho:
"Day 5 predict karo"

Random Forest dekhta hai:
RSI=42, Close=91 → [predicts something]

What it MISSED:
RSI was 70 → 65 → 55 → 42 (continuously falling)
This MOMENTUM matters — but RF sees only a snapshot
```

### The Power of Sequence Memory

LSTM remembers the entire 30-day journey:

```
LSTM ko puchho: "Day 5 predict karo"

LSTM dekhta hai:
[Day1: RSI=70][Day2: RSI=65][Day3: RSI=55][Day4: RSI=42]
      ↑              ↑              ↑              ↑
 Pattern start   Accelerating   Still falling   Momentum down

LSTM ki memory: "Maine yeh pattern 47 baar dekha hai
                 2019 mein, 2021 mein, 2023 mein...
                 Iske baad 2-3 din aur price neeche gayi
                 toh kal bhi giregi"

Prediction: -1.2% ← informed by the full sequence
```

### Real World Proof — COVID Crash 2020

```
Feb 15: Nifty 12,000  (↓ 0.5%)
Feb 20: Nifty 11,500  (↓ 4.2%)  ← pace badh rahi hai
Feb 25: Nifty 10,800  (↓ 6.1%)  ← momentum strong
Mar 01: Nifty 10,000  (↓ 7.4%)  ← panic setting in
Mar 05: ???

Random Forest: "Mar 5 ka data daalo, predict karunga"
               → Sequence ka koi context nahi

LSTM: "5 din mein 17% gira, pace accelerate ho rahi hai,
       yeh pattern historically 8-12 din tak chalta hai,
       selling abhi khatam nahi hui"
       → Sequence ka full context hai ✅
```

### Same Number, Different Meaning

```
RSI = 30

Without sequence: "Oversold! Buy karo!"

With sequence:
RSI was: 80 → 70 → 55 → 42 → 30
         ↑ Falling for 5 days straight
         → "Oversold lekin momentum still bearish, wait karo"

vs.

RSI was: 20 → 22 → 25 → 28 → 30
         ↑ Rising for 5 days from extreme low
         → "Recovering from oversold, momentum turning bullish, buy karo"

SAME NUMBER. COMPLETELY DIFFERENT SIGNAL.
Only LSTM can see this difference. Random Forest cannot.
```

> **Conclusion: Solo ML (Random Forest, XGBoost) stock prediction ke liye "fucked up" hai — data important hai, but us data mein sequence pata hona chahiye. Isliye DL (LSTM) chahiye tha.**

---

## 🔀 Why LSTM and Not Random Forest?

Complete comparison for our specific use case:

```
╔═══════════════════════════╦═══════════════════════════╗
║    RANDOM FOREST (ML)     ║      BiLSTM (DL + ML)     ║
╠═══════════════════════════╬═══════════════════════════╣
║ ✅ Fast training          ║ ✅ Sequence memory         ║
║ ✅ Explainable results    ║ ✅ Temporal patterns       ║
║ ✅ Small data works       ║ ✅ Complex non-linear      ║
║ ✅ No GPU needed          ║    patterns                ║
║ ❌ Sequence BLIND         ║ ✅ Self-Attention focus    ║
║ ❌ No temporal context    ║ ❌ Needs GPU for training  ║
║ ❌ Can't learn: RSI fall  ║ ❌ Black box (less         ║
║    over 5 days = sell     ║    explainable)            ║
║ ❌ Misses momentum        ║ ❌ Needs more data         ║
╚═══════════════════════════╩═══════════════════════════╝

For stock price time-series: BiLSTM wins ✅
For tabular/structured data: Random Forest wins ✅
```

---

## 🏗️ Project Architecture

```
                 ┌──────────────────────────────────────┐
                 │         Next.js Frontend              │
                 │  Price Charts · News Feed             │
                 │  Macro Panel · Chat Interface         │
                 └─────────────────┬────────────────────┘
                                   │ HTTP REST API
                 ┌─────────────────▼────────────────────┐
                 │       FastAPI Backend (main.py)        │
                 │                                       │
                 │  ┌─────────────────────────────────┐  │
                 │  │  1. DATA PIPELINE               │  │
                 │  │  yfinance → 180 days OHLCV      │  │
                 │  │  Gemini Search → Live headlines  │  │
                 │  │  RSS fallback → Al Jazeera, BBC  │  │
                 │  │  yfinance → 5 Macro indicators   │  │
                 │  └───────────────┬─────────────────┘  │
                 │                  ▼                     │
                 │  ┌─────────────────────────────────┐  │
                 │  │  2. FEATURE ENGINEERING         │  │
                 │  │  SMA/EMA · RSI · MACD · BB      │  │
                 │  │  ATR · OBV · Macro 5d returns   │  │
                 │  │  26 total features per day       │  │
                 │  └───────────────┬─────────────────┘  │
                 │                  ▼                     │
                 │  ┌─────────────────────────────────┐  │
                 │  │  3. BiLSTM + SELF-ATTENTION     │  │
                 │  │  Pre-trained on 7 years data    │  │
                 │  │  30-day lookback window         │  │
                 │  │  → Raw 7-day price forecast     │  │
                 │  └───────────────┬─────────────────┘  │
                 │                  ▼                     │
                 │  ┌─────────────────────────────────┐  │
                 │  │  4. GEOPOLITICAL OVERLAY        │  │
                 │  │  VADER NLP sentiment scoring    │  │
                 │  │  Shock detection (supply/demand) │  │
                 │  │  Volatility-scaled adjustment   │  │
                 │  │  Exponential decay over 7 days  │  │
                 │  └───────────────┬─────────────────┘  │
                 │                  ▼                     │
                 │  ┌─────────────────────────────────┐  │
                 │  │  5. MACRO COMPOSITE SCORE       │  │
                 │  │  Weighted signals (VIX, DXY...) │  │
                 │  │  Asset-specific weight configs  │  │
                 │  └─────────────────────────────────┘  │
                 └──────────────────────────────────────┘
                                   │
                 ┌─────────────────▼────────────────────┐
                 │         Gemini 2.5 Flash              │
                 │  Context-aware chat (separate system) │
                 │  Uses LSTM output as context          │
                 └──────────────────────────────────────┘
```

> **Note:** The LSTM model and Gemini chat are **two completely separate systems**. The LSTM predicts numbers. Gemini handles natural language conversation using the LSTM's output as context. This is a deliberate architectural choice — because our LSTM cannot "talk" (it was only trained on price numbers, not text), and Gemini cannot make precise quantitative predictions (it generalizes, doesn't specialize).

---

## 🧠 The LSTM Model — Deep Dive

### What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of **Recurrent Neural Network (RNN)** — designed specifically for *sequential data*. Unlike regular neural networks that process each input independently, LSTMs maintain a **memory cell** that selectively retains relevant information from the past.

Think of it like reading a sentence: you don't forget the beginning by the time you reach the end. LSTM works the same way with time-series data.

### Is It AI, ML, or DL?

```
AI  ✅  Yes — it performs intelligent market prediction
ML  ✅  Yes — it learned from 7 years of historical data
DL  ✅  Yes — it uses a multi-layer neural network

All three simultaneously. One model. Three labels.
```

### Our Architecture: BiLSTM + Self-Attention

```
Input Shape: (batch_size, 30 timesteps, 26 features)
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│        Bidirectional LSTM  (128 units)              │
│        reads sequence → FORWARD + BACKWARD ←       │
│        BatchNormalization + Dropout(0.30)           │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│        Bidirectional LSTM  (64 units)               │
│        deeper temporal pattern extraction           │
│        BatchNormalization                           │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│        Self-Attention Layer  (custom)               │
│        learns WHICH of the 30 days matter most      │
│        e = tanh(X @ W + b)                          │
│        a = softmax(e)  ← attention weights          │
│        out = sum(X * a) ← weighted context vector   │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│        Dense(64, ReLU) + Dropout(0.15)              │
│        Dense(32, ReLU)                              │
│        Dense(1) → next-day % return prediction      │
└─────────────────────────────────────────────────────┘
```

### Why Bidirectional?

A standard LSTM reads left-to-right (past → present). A **Bidirectional LSTM** runs two LSTMs in parallel — one forward, one backward — and concatenates their outputs.

```
Forward LSTM:   [Day1] → [Day2] → [Day3] → ... → [Day30]
Backward LSTM:  [Day30] → [Day29] → ... → [Day1]
                              ↓
                    Concatenate both
                              ↓
              Richer representation of each timestep
              Context from both before AND after
```

This captures momentum reversals and pattern completions that a one-directional LSTM would miss.

### Why Self-Attention?

The **Self-Attention layer** is the same concept that powers GPT, BERT, Gemini, and Claude (Transformers). It learns a *weight* for each of the 30 lookback days — telling the model which moments in the past are most predictive.

```python
# What Self-Attention actually computes:
e = tanh(X @ W + b)      # score each of the 30 days
a = softmax(e, axis=1)   # normalize to probabilities (sum = 1.0)
out = sum(X * a, axis=1) # weighted sum — important days get more weight

# Example output:
# Day 3  → weight 0.18 (important — big RSI divergence)
# Day 15 → weight 0.22 (important — VIX spike happened)
# Day 27 → weight 0.04 (not important — quiet day)
# Day 30 → weight 0.31 (most recent, usually highest weight)
```

Instead of treating all 30 days equally, the model dynamically focuses on the most predictive moments.

### What Does It Predict?

The model predicts the **next-day percentage return**, not the raw price.

```
Wrong approach (price prediction):
Model outputs: 24,532 (Nifty tomorrow)
Problem: Prices are non-stationary — they trend indefinitely
         A model can "cheat" by just predicting yesterday's price + small noise

Correct approach (return prediction):
Model outputs: +0.34% (next-day % change)
Then: predicted_price = current_price × (1 + 0.0034)
Why: Returns oscillate around zero — statistically stable
     Standard in quantitative finance
```

---

## 📚 Training Data & Feature Engineering

### Data Source

- **Provider:** `yfinance` (Yahoo Finance — free, reliable)
- **Assets:** `^NSEI` (Nifty 50) and `BZ=F` (Brent Crude Oil)
- **Training window:** 7 years of daily OHLCV data (≈ 1,750 trading days)
- **Macro signals:** Gold, VIX, USD/INR, S&P 500, Natural Gas, DXY (depending on asset)

### Why Two Separate Models?

```
Nifty 50 is driven by:              Brent Crude is driven by:
├── FII foreign inflows              ├── OPEC production decisions
├── RBI interest rate policy         ├── US Dollar Index (DXY)
├── USD/INR exchange rate            ├── Geopolitical supply risks
├── India VIX (fear index)           ├── Global demand (VIX, SP500)
├── Global risk sentiment            ├── Natural Gas correlation
└── Domestic earnings season         └── Iran/Hormuz tensions

Training one model on both = compromise = suboptimal
Training separate models = each specialized = better accuracy ✅
```

### The 26 Features

```
Category        │ Features
────────────────┼──────────────────────────────────────────────
Price (OHLCV)   │ Close, Open, High, Low, Volume
Trend           │ SMA(10), SMA(20), EMA(20), EMA(50), SMA_Ratio
Momentum        │ RSI(14), MACD, MACD_Signal, MACD_Hist, Momentum_5d
Volatility      │ BB_Upper, BB_Lower, BB_Width, ATR(14), Rolling_Vol(10d)
Volume          │ OBV (On-Balance Volume)
Macro (Nifty)   │ USD/INR 5d%, India_VIX 5d%, Gold 5d%, Crude 5d%, SP500 5d%
Macro (Crude)   │ DXY 5d%, Gold 5d%, SP500 5d%, NatGas 5d%, VIX 5d%
```

**Why 5-day returns for macro, not daily?**

```
Daily macro returns = too noisy
"SP500 moved 0.2% today" → not meaningful signal

5-day macro returns = meaningful trend
"SP500 moved -4.3% this week" → real risk-off signal that affects our asset

Markets reprice macro information over a week, not a day.
5-day window captures the signal without noise.
```

### Scaling Strategy

Two separate scalers are used — and this matters a lot:

```
Input Scaler (MinMaxScaler):
├── Scales all 26 features to [0, 1]
├── Fit ONLY on training set (prevents data leakage)
├── Test set is just transformed (not fit)
└── Saved as scaler_{TICKER}.pkl

Target Scaler (StandardScaler):
├── Scales the target returns to zero-mean, unit-variance
├── Helps model output reasonable gradients
├── "Return of +1.5%" becomes a normalized value
└── Saved as target_scaler_{TICKER}.pkl

Why two scalers?
Input features = different scales (Close is 24000, RSI is 0-100, OBV is millions)
→ MinMaxScaler normalizes all to same range

Target returns = already small numbers (-0.05 to +0.05)
→ StandardScaler standardizes for gradient stability
```

### Train / Test Split

```
Full dataset: 7 years (~1750 days)
         ↓
Split: 80% train │ 20% test
       ~1400 days  ~350 days

IMPORTANT: Chronological split — NOT random shuffle
If you shuffle: Future data leaks into training = fake accuracy
If chronological: Model trained on past, tested on actual future = honest
```

### Training Callbacks

```python
EarlyStopping(patience=15, restore_best_weights=True)
# → Stops training when val_loss stops improving for 15 epochs
# → Restores weights from best epoch (not last epoch)
# → Prevents overfitting

ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
# → Halves learning rate when val_loss plateaus for 5 epochs
# → Allows model to fine-tune without overshooting
# → Squeezes out extra performance
```

---

## 🖥️ Why Google Colab?

### The Problem: Local Hardware

Training a Bidirectional LSTM on 1,750 days × 26 features × 30 lookback × 100 epochs requires serious compute. A typical laptop:
- Has a weak GPU or no discrete GPU
- Would take 2-4 hours on CPU
- Fan would run at full speed throughout

### The Solution: Google Colab

**Google Colab** is Google's free cloud Jupyter environment. It provides:

```
Free Resources on Colab:
├── Tesla T4 GPU (16 GB VRAM) — 10-20x faster than CPU
├── Python 3.10 pre-installed
├── TensorFlow, NumPy, Pandas pre-installed
├── 12 GB RAM
├── One-click file download after training
└── No setup needed — open browser, run, done
```

### How Training Works — Step by Step

```
Step 1: Open Aegis_Colab_Training.ipynb in Google Colab
Step 2: Set TICKER = "^NSEI" or "BZ=F"
Step 3: Click "Run All"
        │
        ├── Downloads 7 years stock + 5 macro assets (yfinance)
        ├── Engineers all 26 features
        ├── Splits 80/20 chronologically
        ├── Scales inputs + targets
        ├── Builds BiLSTM + Self-Attention architecture
        ├── Trains on T4 GPU (~10-15 minutes)
        ├── Evaluates directional accuracy + MAPE on test set
        └── Auto-downloads 4 files to your machine:
            ├── lstm_model_{TICKER}.h5    (trained model weights)
            ├── scaler_{TICKER}.pkl       (input MinMaxScaler)
            ├── target_scaler_{TICKER}.pkl (target StandardScaler)
            └── feature_config_{TICKER}.json (features + accuracy metrics)

Step 4: Drop the 4 files into project root
Step 5: FastAPI backend auto-loads them → live predictions start
```

### Why Not Train Locally?

```
Local CPU training:
├── Time: 2-4 hours
├── Fan: 100% speed the entire time
├── Laptop heats up significantly
└── Can't use laptop while training

Google Colab (free T4 GPU):
├── Time: 10-15 minutes
├── Your laptop stays cool
├── You can do other work
└── Zero cost
```

---

## 🌍 The Geopolitical Intelligence Layer

This is what makes Aegis unique beyond just being an LSTM wrapper.

### News Sourcing — Dual Path

```
Primary Path: Gemini 2.5 Flash + Google Search
└── 25 most recent headlines (last 48 hours)
└── Queries: crude oil, OPEC, Hormuz, Iran, Fed, Nifty, RBI
└── Advantage: Real-time, high quality, specific

Fallback Path: RSS Feed Parsing
├── Al Jazeera (global news)
├── BBC World (international)
├── Google News: Energy (crude, OPEC, Hormuz)
├── Google News: Geopolitics (conflict, sanctions, war)
├── Google News: Markets (oil, Fed, inflation)
└── Google News: India (Nifty, RBI, rupee)
```

### Sentiment Scoring — VADER NLP

Every headline is scored using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:

```
"OPEC agrees to emergency production cut"
→ VADER compound score: -0.72 (negative for oil consumers, bearish signal)

"Iran agrees to nuclear deal, sanctions to be lifted"
→ VADER compound score: +0.81 (de-escalation, supply risk falls)

"US recession fears deepen as manufacturing contracts"
→ VADER compound score: -0.64 (demand destruction risk)

Output range: -1.0 (maximum bearish) to +1.0 (maximum bullish)
```

VADER is rule-based NLP — specifically tuned for short news-style text. It's not as powerful as a neural language model, but it's fast, free, and accurate enough for headline-level sentiment.

### Shock Detection System

The system scans recent 15 headlines for three event categories:

```
SUPPLY SHOCK Keywords:
'hormuz', 'strait of hormuz', 'blockade', 'tanker attack',
'tanker seized', 'oil field attack', 'pipeline attack',
'refinery strike', 'production cut', 'supply disruption',
'naval blockade', 'opec cuts output', 'strait closure'

DE-ESCALATION Keywords:
'ceasefire', 'peace talks', 'diplomatic breakthrough',
'de-escalate', 'truce', 'sanctions lifted', 'sanctions eased',
'nuclear deal', 'agreement reached', 'tensions ease',
'stability restored', 'opec increases output'

DEMAND SHOCK Keywords:
'recession', 'economic slowdown', 'demand destruction',
'gdp contracts', 'layoffs surge', 'manufacturing slumps',
'growth stalls', 'global slowdown'
```

### Shock Impact on Assets

```
╔═══════════════════╦═══════════════════╦══════════════════╗
║   Shock Type      ║   Crude (BZ=F)    ║   Nifty (^NSEI)  ║
╠═══════════════════╬═══════════════════╬══════════════════╣
║ Supply Shock      ║  ↑ UP             ║  ↓ DOWN          ║
║ (Hormuz blocked)  ║  Supply risk →    ║  Oil costlier →  ║
║                   ║  crude spikes     ║  India pays more ║
╠═══════════════════╬═══════════════════╬══════════════════╣
║ De-escalation     ║  ↓ DOWN           ║  ↑ UP            ║
║ (ceasefire)       ║  Risk premium     ║  Cheaper oil →   ║
║                   ║  removed          ║  India benefits  ║
╠═══════════════════╬═══════════════════╬══════════════════╣
║ Demand Shock      ║  ↓ DOWN           ║  ↓ DOWN          ║
║ (recession)       ║  Less demand →    ║  Global selloff  ║
║                   ║  crude falls      ║  hits EM markets ║
╚═══════════════════╩═══════════════════╩══════════════════╝
```

### The Adjustment Math

```python
# Volatility-scaled geopolitical adjustment
historical_volatility = stock_df['Close'].pct_change().std()

# For Brent Crude: negative sentiment → crude goes UP
geopolitical_adjustment = -current_sentiment × historical_volatility

# For Nifty: negative sentiment → Nifty goes DOWN
geopolitical_adjustment = current_sentiment × 0.8 × historical_volatility

# Supply shock amplifier
if shock == 'Supply shock':
    geopolitical_adjustment += 1.5 × historical_volatility  # (Crude)
    geopolitical_adjustment -= 1.2 × historical_volatility  # (Nifty)

# Applied with exponential decay over 7 days
# (geopolitical shocks have diminishing impact — markets digest news fast)
for day i in range(7):
    adjusted_return = ml_return + geopolitical_adjustment × (0.75 ** i)
    #                                                          ↑
    #                                           Day 0: 100% adjustment
    #                                           Day 1: 75% adjustment
    #                                           Day 2: 56% adjustment
    #                                           Day 7: ~13% adjustment
```

---

## 🔬 The Big Design Decision — Why News Is NOT in Training

> **This is one of the most important engineering decisions in this project — and understanding it reveals a lot about how real ML systems are built.**

### The Original (Wrong) Approach

When Aegis was first being built, the training notebook included news sentiment as a feature:

```
Training data structure (original):
Day 1: [RSI, MACD, Close...] + [sentiment: -0.23] → predict return
Day 2: [RSI, MACD, Close...] + [sentiment: +0.15] → predict return
...
```

**The problem:** News APIs only had 30 days of history available. But we needed 7 years of training data.

```
Price data:  7 years available → ~1,750 rows ✅
News data:   30 days available → ~30 rows ❌

Result:
Row 1 to 1720:  sentiment = 0.0  (no news data — artificially filled)
Row 1721-1750:  sentiment = real values

Model learned: "When sentiment = 0.0 (which is almost always),
               price behaves like X"
= Garbage learning ❌
= Accuracy dropped significantly
```

### The Solution — Separate the Systems

```
Training:  
Pure technical + macro features only (7 years of clean data)
No news — because historical news data wasn't available for 7 years
→ Model learned clean, reliable patterns ✅

Inference (prediction time):
Step 1: LSTM gives raw prediction (no news influence)
Step 2: Geopolitical adjustment layer adds news overlay
Step 3: Final prediction = raw + geo-adjustment

News affects the FINAL prediction but NOT the training
= Clean separation ✅
= Honest training ✅
= Better accuracy ✅
```

### Why This Is Actually Professional-Grade

Hedge funds and quant firms use this exact pattern:

```
Professional Quant Architecture:
┌─────────────────────────────────┐
│   Quantitative Model (pure)     │  ← trained on clean historical data
│   Price + Macro only            │     no noisy/incomplete features
└──────────────────┬──────────────┘
                   │
                   ▼
┌─────────────────────────────────┐
│   Overlay / Adjustment Layer    │  ← applied at prediction time
│   News, Events, Sentiment       │     not baked into training
└──────────────────┬──────────────┘
                   │
                   ▼
┌─────────────────────────────────┐
│   Final Signal                  │
└─────────────────────────────────┘
```

Aegis follows this exact architecture — not by accident, but because the problem revealed itself during development and the right solution was found.

---

## 📊 Model Performance — Honest Numbers

| Asset | Directional Accuracy | MAPE | What This Means |
|---|---|---|---|
| **Nifty 50** (`^NSEI`) | **52.24%** | 0.64% | Gets direction right 52% of the time |
| **Brent Crude** (`BZ=F`) | **50.16%** | 2.08% | Barely better than random for direction |

### What These Numbers Actually Mean

```
Coin flip:              50% directional accuracy
Random guessing:        50% directional accuracy
Our Nifty model:        52.24% directional accuracy

"That's barely better!" → True, but misleading.

In financial markets:
- If you're right 55%+ consistently → you beat most hedge funds
- If you're right 52% consistently with good position sizing → profitable
- The edge compounds over hundreds of trades

MAPE (Mean Absolute Percentage Error):
- Nifty MAPE = 0.64% → On average, price prediction is off by 0.64%
- Crude MAPE = 2.08% → Crude is more volatile, harder to predict

These are HONEST numbers from the held-out test set.
Not cherry-picked. Not training set accuracy. Real out-of-sample performance.
```

### The Confidence Score

The confidence score shown in the UI is derived directly from the model's actual directional accuracy:

```python
# In main.py — real confidence, not hardcoded
raw_dir_acc = feature_config.get("directional_accuracy", 50.0)
confidence_score = float(np.clip(raw_dir_acc / 100.0, 0.0, 1.0))

# Nifty:  52.24% → confidence = 0.5224 → UI shows ~52%
# Crude:  50.16% → confidence = 0.5016 → UI shows ~50%
```

> **We removed the hardcoded 84% confidence that was in the original code. That number was fake. These numbers are real.**

---

## 💰 What Paid News APIs Would Unlock

Our current setup is impressive for a free-tier system. But here's what paid APIs would change:

### Current Free Setup

```
Gemini Search → ~25 headlines (48h window)
RSS Feeds     → Al Jazeera, BBC, Google News
VADER NLP     → Basic sentiment scoring (-1 to +1)
```

### What Paid APIs Provide

| API | Cost | Unlock |
|---|---|---|
| **Alpha Vantage** | $50/mo | Historical news + pre-built finance sentiment |
| **Benzinga Pro** | $99/mo | Finance-specific news, pre-market alerts |
| **EODHD** | $79/mo | News + sentiment + fundamentals |
| **Refinitiv (Reuters)** | $1,500/mo | Professional real-time everything |
| **Bloomberg Terminal** | $2,000/mo | What hedge funds use |

### The Key Unlock: Historical News in Training

```
With paid API (7 years historical news):

Training data becomes:
Day 1:    [RSI, MACD, Close...] + [real sentiment: -0.23] → return
Day 2:    [RSI, MACD, Close...] + [real sentiment: +0.15] → return
...
Day 1750: [RSI, MACD, Close...] + [real sentiment: -0.61] → return

Model can now learn:
"When RSI falling + negative sentiment + VIX rising
 → crude supply shock pattern → price up 2-4%"

This is what the current model CANNOT learn (because no historical news)
Accuracy improvement: potentially significant (55-58% directional accuracy)
```

### Upgrade Path

```
Level 0: RSS + Gemini Search        → Free ✅  (current)
Level 1: Alpha Vantage free tier    → Free ✅  (add now — easy)
Level 2: Alpha Vantage premium      → $50/mo   (student budget)
Level 3: Benzinga Pro               → $99/mo   (serious trader)
Level 4: Refinitiv Elektron         → $500+/mo (startup level)
Level 5: Bloomberg Terminal         → $2,000/mo (hedge fund level)
```

---

## ⚡ Tech Stack

```
╔══════════════════════════════════════════════════════════╗
║                      TECH STACK                          ║
╠══════════════════════╦═══════════════════════════════════╣
║ Backend API          ║ Python + FastAPI                  ║
║ ML/DL Framework      ║ TensorFlow 2.x + Keras            ║
║ Market Data          ║ yfinance (Yahoo Finance)           ║
║ NLP Sentiment        ║ VADER (vaderSentiment library)     ║
║ AI Chat              ║ Google Gemini 2.5 Flash (API)      ║
║ Data Processing      ║ pandas + numpy                     ║
║ Scalers              ║ scikit-learn (MinMaxScaler, SS)    ║
║ Frontend             ║ Next.js (React)                    ║
║ Training Env         ║ Google Colab (free T4 GPU)         ║
║ Model Format         ║ Keras .h5 (HDF5 weights file)      ║
║ Scaler Format        ║ Python pickle (.pkl)               ║
║ Config Format        ║ JSON                               ║
╚══════════════════════╩═══════════════════════════════════╝
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Backend Setup

```bash
# 1. Clone repo
git clone <your-repo-url>
cd "Militiary ai"

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install fastapi uvicorn tensorflow yfinance vaderSentiment \
            google-genai python-dotenv pandas numpy scikit-learn

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add:
# GEMINI_API_KEY=your_key_here

# 5. Start backend
uvicorn main:app --reload --port 8000
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Frontend at http://localhost:3000
```

### Training New Models

```
1. Go to https://colab.research.google.com
2. Upload Aegis_Colab_Training.ipynb
3. Set TICKER = "^NSEI" or "BZ=F" in Step 3 cell
4. Runtime > Run All
5. Wait ~10-15 minutes
6. Download the 4 files that auto-download
7. Place them in project root directory
8. Restart backend — new model loads automatically
```

---

## 📁 Project Structure

```
Militiary ai/
│
├── main.py                         ← FastAPI backend
│   ├── Data pipeline (yfinance + RSS + Gemini)
│   ├── Feature engineering (26 features)
│   ├── ML asset loading (model + scalers)
│   ├── 7-day prediction (raw + geo-adjusted)
│   ├── Geopolitical shock detection
│   ├── Macro composite score
│   └── Chat endpoint (Gemini integration)
│
├── Aegis_Colab_Training.ipynb      ← Training notebook (run on Colab)
│
├── lstm_model_NSEI.h5              ← Trained BiLSTM weights (Nifty 50)
├── lstm_model_BZF.h5               ← Trained BiLSTM weights (Brent Crude)
│
├── scaler_NSEI.pkl                 ← MinMaxScaler for Nifty inputs
├── scaler_BZF.pkl                  ← MinMaxScaler for Crude inputs
│
├── target_scaler_NSEI.pkl          ← StandardScaler for Nifty returns
├── target_scaler_BZF.pkl           ← StandardScaler for Crude returns
│
├── feature_config_NSEI.json        ← Features list + accuracy metrics
├── feature_config_BZF.json         ← Features list + accuracy metrics
│
├── frontend/                       ← Next.js frontend
├── .env                            ← API keys (never commit this)
├── .gitignore
└── README.md                       ← You are here
```

---

## ❓ FAQ — Everything Explained

**Q: Is this AI, ML, or DL?**
> All three simultaneously. They're nested categories, not separate things. Our LSTM is a Deep Learning model (neural network) → which means it's also Machine Learning (learns from data) → which means it's also AI (performs intelligent tasks). Three labels, one model.

**Q: Can the LSTM model "talk" to users by itself?**
> No — and this is a fundamental limitation, not a hardware one. Our LSTM was trained exclusively on numerical stock data (RSI, MACD, prices). It has never seen a single word of text. It cannot understand language. To talk, you need a Language Model (LLM) — which requires training on terabytes of text with billions of parameters. That's what Gemini does, and why we use the Gemini API for chat. Both are DL+ML+AI, but different architectures for completely different purposes.

**Q: Why is Gemini used separately? Can't the LSTM also chat?**
> No. Our LSTM outputs one number: a percentage return. That's all it can ever output — because that's all it was trained to produce. Gemini is an LLM trained on human language — it outputs text. The LSTM's prediction is fed to Gemini as *context*, and then Gemini uses natural language to explain what it means. Two separate systems, each doing what it was designed for.

**Q: Why wasn't news included in training?**
> Smart question. We tried it — and accuracy dropped. The problem: we needed 7 years of training data, but only had 30 days of news history. So 1,720 of 1,750 training rows had `sentiment = 0.0` (artificially filled). The model learned: "When sentiment is 0 (which is almost always), price does X" — meaning it learned from the absence of news data, not from actual news. Removing news from training and applying it as a post-processing overlay at inference time fixed the problem significantly.

**Q: Why Random Forest would be "fucked up" for stocks?**
> Because Random Forest is sequence-blind. It sees each day as independent. Give it today's RSI = 42 — it predicts something. But it doesn't know that RSI was 70 five days ago and has been falling every day. That momentum — the sequence — is everything in financial markets. Same number means completely different things depending on where you're coming from. LSTM's memory handles this; Random Forest cannot.

**Q: What's the confidence score in the UI?**
> It's the model's **actual directional accuracy from the held-out test set** — stored in `feature_config_{TICKER}.json`. Nifty: 52.24%, Crude: 50.16%. The original code had a hardcoded `0.84` (84%) which was completely fabricated. That was removed and replaced with the real numbers from actual model evaluation.

**Q: Why are directional accuracies so low (52%)?**
> Because financial markets are genuinely hard to predict. They're near-random-walk processes. A model that gets direction right 52% of the time consistently, on out-of-sample data, is actually meaningful — it's not just flipping coins. Most retail traders and many institutional strategies don't beat this benchmark consistently. The value is in the combined system: LSTM trend + geopolitical overlay + macro scoring — not just the raw directional accuracy.

**Q: What did paid news APIs unlock?**
> Currently: only live headlines for real-time overlay. With paid APIs having historical news: we could train the LSTM with 7 years of sentiment data as a feature, allowing it to learn patterns like "when Iran tensions spike + RSI falling + DXY rising → crude historically jumps 3-5%." That's currently not learnable because we don't have 7 years of news training data.

**Q: Is this project "complete" AI — having all three (AI/ML/DL)?**
> Yes. Our LSTM is simultaneously DL (neural network) + ML (trained on data) + AI (intelligent prediction). You don't need to "add" them separately — DL is always automatically ML and AI. The confusion comes from thinking they're three separate things to combine, when they're actually just three different ways to describe the same thing.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Do not make investment decisions based solely on this system's output. Financial markets are inherently uncertain — even the best quantitative models carry significant prediction error. Always consult a qualified financial advisor.

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** — neural network framework
- **yfinance** — free market data access
- **VADER Sentiment** — NLP sentiment analysis
- **Google Gemini** — AI chat and search grounding
- **Google Colab** — free GPU training environment
- **FastAPI** — high-performance Python API framework

---

<div align="center">

*Built with ❤️, Python, TensorFlow, FastAPI, and Google Gemini*

```
"Data ne sikha. Neural network ne seekha. AI ban gaya." 🚀
```

</div>
