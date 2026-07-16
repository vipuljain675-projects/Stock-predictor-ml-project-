### From APIs to Training Neural Networks: Building Aegis Markets

After working with basic APIs in earlier projects and competing in a hackathon, I wanted to go a level deeper: training a custom deep learning model from scratch.

I’ve built **Aegis Markets (GeoQuant AI)**—a quantitative dashboard that forecasts Nifty 50 and Brent Crude by blending a custom-trained BiLSTM + Attention neural network with real-time geopolitical sentiment analysis.

Here is how it fits into the quantitative landscape, the challenges I faced, and what I learned:

---

#### 🗺️ The Landscape: Where does this stand?
* **Tier 1 (Retail)**: Relying on basic technical indicators (RSI, MACD) or simple regressions. Blind to macro shifts.
* **Tier 2 (Hybrid Quant - My Project)**: Blending deep learning models with alternative unstructured data (VIX, DXY, Gold, and LLM-parsed news sentiment).
* **Tier 3 (Wall Street HFTs)**: Running colocated servers with microsecond execution times, trading order books, and leveraging alternative feeds.

---

#### 🛠️ The Tech Stack & Pipeline:
* **Data Processing**: NumPy & Pandas to clean and shape historical data into tensors.
* **Deep Learning**: A **Bidirectional LSTM + Self-Attention** model in TensorFlow/Keras. The BiLSTM captures temporal momentum; Attention weights the critical lookback days.
* **News Analysis Pipeline (Backend)**: Scraping live headlines via Google Search/News APIs, running **VADER Sentiment** for polarity, and using **Gemini** to classify shock types (supply shocks vs de-escalation) and their severity.

---

#### 🧠 The Debugging Battles:
* **Trivial Predictor Convergence**: The LSTM initially converged to a flat line, constantly predicting `+0.05%` (historical average). The optimizer minimized loss by just predicting the historical average return.
* **Price Scale Saturation**: The first iteration used absolute prices. When Nifty rose to ₹24,135 in production (out-of-training range), the scaler saturated and froze the outputs.
* **The Stationary Solution**: I refactored the pipeline to feed only relative returns and moving average ratios (Close/SMA_10). The model became dynamic, responsive, and immune to price drift. In extreme crash simulations, it correctly predicted a technical relief bounce (+0.12%)—showing it had learned mean-reversion.

---

#### ⚠️ Critical Limitations:
1. **30-Day Lookback**: The LSTM reads only 30 days of sequences to filter out historical noise, making it blind to long-term macro cycles.
2. **Sentiment Mismatch**: A top-tier model requires **Multimodal Fusion** (training price + news sentiment *together*). Lacking an 8-year daily news API archive, the LSTM is trained strictly on price data, and news is overlaid as a manual heuristic in the backend.

🔗 **GitHub Repository**: [Your Repo Link Here]

![Aegis Markets System Architecture](file:///Users/vipuljain675/Documents/GeoQuant%20AI/system_architecture_diagram.png)

#MachineLearning #DeepLearning #Python #TensorFlow #QuantitativeFinance #FinTech #DataScience
