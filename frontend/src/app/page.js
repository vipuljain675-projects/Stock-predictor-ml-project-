"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  TrendingUp,
  TrendingDown,
  Globe,
  DollarSign,
  AlertTriangle,
  RefreshCw,
  Send,
  MessageSquare,
  Activity,
  Layers,
  ArrowRight
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid
} from "recharts";

export default function AegisDashboard() {
  const [ticker, setTicker] = useState("BZ=F");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("chart"); // chart, macro, news

  // Chat state
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", content: "Aegis analyst online. I have ingested current market prices and today's news feeds. Ask me anything." }
  ]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchDashboardData = async (symbol) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/market-data?symbol=${symbol}`);
      if (!res.ok) {
        throw new Error("Failed to fetch market data from backend server");
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message || "Something went wrong while loading data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData(ticker);
  }, [ticker]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const queryText = chatInput;
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: queryText }]);
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText,
          ticker_symbol: ticker,
          messages: chatMessages
        })
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Gemini API failed to respond");
      }

      const json = await res.json();
      setChatMessages((prev) => [...prev, { role: "assistant", content: json.response }]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `⚠️ Error: ${err.message}` }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchDashboardData(ticker);
  };

  const getCurrencySymbol = () => (ticker === "^NSEI" ? "₹" : "$");

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-zinc-50/50 p-6">
        <div className="flex items-center gap-3 mb-6">
          <Activity className="h-6 w-6 text-zinc-600 animate-pulse" />
          <span className="font-semibold text-zinc-950 tracking-tight">Loading Aegis Markets Intelligence Console...</span>
        </div>
        <div className="w-64 h-1.5 bg-zinc-200 rounded-full overflow-hidden">
          <div className="h-full bg-zinc-900 animate-infinite-loading rounded-full" style={{ width: "60%" }} />
        </div>
      </div>
    );
  }

  // Pre-process chart data
  let chartData = [];
  if (data) {
    const historical = data.historical_chart.slice(-90).map((d) => ({
      date: d.date,
      historical: d.close,
      displayDate: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
    }));
    const forecast = data.forecast_chart.map((d) => ({
      date: d.date,
      forecast: d.close,
      displayDate: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
    }));
    const forecastRaw = data.forecast_chart_raw ? data.forecast_chart_raw.map((d) => ({
      date: d.date,
      forecastRaw: d.close,
      displayDate: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
    })) : [];

    // Connect historical end to forecast start
    if (historical.length > 0 && forecast.length > 0) {
      forecast.unshift({
        date: historical[historical.length - 1].date,
        forecast: historical[historical.length - 1].historical,
        displayDate: historical[historical.length - 1].displayDate
      });
    }

    // Connect historical end to raw forecast start
    if (historical.length > 0 && forecastRaw.length > 0) {
      forecastRaw.unshift({
        date: historical[historical.length - 1].date,
        forecastRaw: historical[historical.length - 1].historical,
        displayDate: historical[historical.length - 1].displayDate
      });
    }

    // Merge lists
    const allDates = [...new Set([
      ...historical.map(h => h.date), 
      ...forecast.map(f => f.date),
      ...forecastRaw.map(fr => fr.date)
    ])].sort();

    chartData = allDates.map((date) => {
      const hMatch = historical.find((h) => h.date === date);
      const fMatch = forecast.find((f) => f.date === date);
      const frMatch = forecastRaw.find((fr) => fr.date === date);
      const matchedDate = hMatch || fMatch || frMatch;
      return {
        date,
        displayDate: matchedDate ? matchedDate.displayDate : "",
        historical: hMatch ? hMatch.historical : null,
        forecast: fMatch ? fMatch.forecast : null,
        forecastRaw: frMatch ? frMatch.forecastRaw : null
      };
    });
  }

  return (
    <div className="max-w-[1600px] mx-auto p-8 flex flex-col gap-8 min-h-screen">
      {/* HEADER SECTION */}
      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 border-b border-zinc-200/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-3xl font-extrabold text-zinc-950 tracking-tight">Aegis Markets</h1>
            <span className="font-mono text-xs font-semibold bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded">v3.5 Live</span>
          </div>
          <p className="text-sm text-zinc-500 font-medium">Real-time market forecasting and geopolitical impact analysis console.</p>
        </div>

        <div className="flex items-center gap-4">
          {/* Asset Picker Segmented Controls */}
          <div className="bg-zinc-100 p-1 rounded-xl flex gap-1 border border-zinc-200/60">
            <button
              onClick={() => setTicker("BZ=F")}
              className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
                ticker === "BZ=F"
                  ? "bg-white text-zinc-950 shadow-sm"
                  : "text-zinc-600 hover:text-zinc-950"
              }`}
            >
              Brent Crude Oil
            </button>
            <button
              onClick={() => setTicker("^NSEI")}
              className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
                ticker === "^NSEI"
                  ? "bg-white text-zinc-950 shadow-sm"
                  : "text-zinc-600 hover:text-zinc-950"
              }`}
            >
              Nifty 50 Index
            </button>
          </div>

          <button
            onClick={handleRefresh}
            className="p-2.5 bg-white border border-zinc-200 hover:bg-zinc-50 rounded-xl transition-all shadow-sm flex items-center justify-center text-zinc-600"
            title="Refresh dashboard data"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-2xl p-5 flex gap-3 items-center">
          <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0" />
          <div className="text-sm font-semibold">{error}</div>
        </div>
      )}

      {data && (
        <>
          {/* KPI STATS GRID */}
          <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* KPI 1 */}
            <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-36">
              <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 font-bold">Current Close</span>
              <div className="flex flex-col mt-2">
                <span className="font-mono text-3xl font-extrabold text-zinc-900">
                  {getCurrencySymbol()}{data.last_close.toLocaleString("en-US", { minimumFractionDigits: 2 })}
                </span>
                <span className="text-[11px] text-zinc-400 font-medium mt-1 font-mono">Real-time quote</span>
              </div>
            </div>

            {/* KPI 2 */}
            <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-36">
              <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 font-bold">Geopolitical Sentiment</span>
              <div className="flex flex-col mt-2">
                <span className={`font-mono text-3xl font-extrabold ${
                  data.avg_sentiment >= 0.05
                    ? "text-emerald-600"
                    : data.avg_sentiment <= -0.05
                    ? "text-rose-600"
                    : "text-zinc-600"
                }`}>
                  {data.avg_sentiment >= 0 ? "+" : ""}{data.avg_sentiment.toFixed(2)}
                </span>
                <span className="text-[11px] text-zinc-500 font-medium mt-1 font-mono uppercase tracking-wider">
                  {data.avg_sentiment >= 0.05
                    ? "Bullish / Stable"
                    : data.avg_sentiment <= -0.05
                    ? "Bearish / Tension"
                    : "Neutral / Calm"}
                </span>
              </div>
            </div>

            {/* KPI 3 */}
            <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-36">
              <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 font-bold">Geopolitical Adjusted Forecast (24h)</span>
              <div className="flex flex-col mt-1">
                {data.model_usable && data.next_predict ? (
                  <>
                    <span className={`font-mono text-2xl font-extrabold ${data.change_pct >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                      {getCurrencySymbol()}{data.next_predict.toLocaleString("en-US", { minimumFractionDigits: 2 })}
                    </span>
                    <div className="flex items-center justify-between mt-1">
                      <span className={`text-[10px] font-bold font-mono ${data.change_pct >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                        {data.change_pct >= 0 ? "▲" : "▼"} {data.change_pct >= 0 ? "+" : ""}{data.change_pct.toFixed(2)}% (Adj)
                      </span>
                      {data.next_predict_raw !== undefined && (
                        <span className="text-[10px] text-zinc-400 font-mono font-medium">
                          Raw ML: {data.change_pct_raw >= 0 ? "+" : ""}{data.change_pct_raw.toFixed(2)}%
                        </span>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <span className="font-mono text-2xl font-bold text-zinc-400">Unavailable</span>
                    <span className="text-[11px] text-zinc-400 font-medium mt-1 font-mono">Model inactive</span>
                  </>
                )}
              </div>
            </div>

            {/* KPI 4 */}
            <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-36">
              <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 font-bold">Model Confidence</span>
              <div className="flex flex-col mt-2">
                {data.model_usable ? (
                  <>
                    <span className="font-mono text-3xl font-extrabold text-zinc-900">
                      {(data.confidence_score * 100).toFixed(0)}%
                    </span>
                    <span className="text-[11px] text-zinc-500 font-medium mt-1 font-mono uppercase tracking-wider">
                      BiLSTM+Attention Active
                    </span>
                  </>
                ) : (
                  <>
                    <span className="font-mono text-2xl font-bold text-zinc-400">Offline</span>
                    <span className="text-[11px] text-zinc-400 font-medium mt-1 font-mono">Inputs mismatch</span>
                  </>
                )}
              </div>
            </div>
          </section>

          {/* SHOCK ALERT SECTION */}
          {data.shock && (
            <section className="bg-amber-50 border-l-4 border-l-amber-500 border border-zinc-200/50 rounded-r-2xl p-6 flex gap-4 items-start shadow-sm animate-pulse-subtle">
              <AlertTriangle className="h-6 w-6 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <div className="font-mono text-[10px] uppercase tracking-wider text-amber-700 font-bold mb-1">
                  ⚡ Live Market Event Detected — {data.shock.type}
                </div>
                <div className="text-sm font-bold text-zinc-950 italic mb-2">
                  "{data.shock.headline}"
                </div>
                <div className="text-xs text-zinc-600 font-medium leading-relaxed max-w-4xl mb-2">
                  {data.shock.reasoning}
                </div>
                <div className="text-[11px] font-bold font-mono uppercase tracking-wider">
                  Bias:{" "}
                  <span className={data.shock.nifty_dir === "up" ? "text-emerald-600" : "text-rose-600"}>
                    {ticker === "^NSEI" ? data.shock.nifty_dir.toUpperCase() : data.shock.crude_dir.toUpperCase()}
                  </span>
                </div>
              </div>
            </section>
          )}

          {/* GEOPOLITICAL ADJUSTMENT DETAILED CARD */}
          {data.model_usable && data.next_predict_raw !== undefined && (
            <section className="bg-zinc-50 border border-zinc-200/80 rounded-2xl p-5 shadow-sm">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h4 className="text-xs font-bold text-zinc-950 uppercase tracking-wider font-mono mb-1 flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" /> 🔧 Geopolitical Adjuster Panel
                  </h4>
                  <p className="text-[11px] text-zinc-500 font-medium">
                    This panel displays how the raw ML model prediction is tactically adjusted using live news sentiment and macroeconomic shocks.
                  </p>
                </div>
                <div className="flex flex-wrap gap-4 items-center font-mono text-xs">
                  <div className="bg-white border border-zinc-200 px-3 py-1.5 rounded-xl text-center shadow-xs">
                    <span className="text-[9px] text-zinc-400 block uppercase font-bold">Raw Model Return</span>
                    <span className="font-bold text-zinc-800">{data.change_pct_raw >= 0 ? "+" : ""}{data.change_pct_raw.toFixed(2)}%</span>
                  </div>
                  <div className="text-zinc-400 font-bold text-sm">+</div>
                  <div className="bg-white border border-zinc-200 px-3 py-1.5 rounded-xl text-center shadow-xs">
                    <span className="text-[9px] text-zinc-400 block uppercase font-bold">Geopolitical Tilt</span>
                    <span className={`font-bold ${data.change_pct - data.change_pct_raw >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                      {data.change_pct - data.change_pct_raw >= 0 ? "+" : ""}{(data.change_pct - data.change_pct_raw).toFixed(2)}%
                    </span>
                  </div>
                  <div className="text-zinc-400 font-bold text-sm">=</div>
                  <div className="bg-zinc-900 border border-zinc-800 text-white px-3 py-1.5 rounded-xl text-center shadow-xs">
                    <span className="text-[9px] text-zinc-400 block uppercase font-bold">Final Adjusted Return</span>
                    <span className={`font-bold ${data.change_pct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      {data.change_pct >= 0 ? "+" : ""}{data.change_pct.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* MAIN GRAPH & NEWS SPLIT */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Charts & Forecast */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col gap-6">
                <div className="flex items-center justify-between border-b border-zinc-100 pb-4">
                  <div>
                    <h2 className="text-lg font-bold text-zinc-950 tracking-tight">Timeline Analytics & 7-Day Forecast</h2>
                    <p className="text-xs text-zinc-500 font-medium">Historical prices and quantitative LSTM predictions.</p>
                  </div>
                  <div className="flex gap-3">
                    <span className="flex items-center gap-1.5 text-xs text-zinc-500 font-semibold font-mono">
                      <span className="h-2 w-2 rounded-full bg-zinc-800" /> Historical Price
                    </span>
                    {data.model_usable && (
                      <>
                        <span className="flex items-center gap-1.5 text-xs text-zinc-500 font-semibold font-mono">
                          <span className="h-2 w-2 rounded-full bg-zinc-400" /> Raw ML Model
                        </span>
                        <span className="flex items-center gap-1.5 text-xs text-zinc-500 font-semibold font-mono">
                          <span className="h-2 w-2 rounded-full bg-blue-500" /> Geopolitical Adjusted
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <div className="h-96 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorHist" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#18181b" stopOpacity={0.06} />
                          <stop offset="95%" stopColor="#18181b" stopOpacity={0.0} />
                        </linearGradient>
                        <linearGradient id="colorFore" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                        </linearGradient>
                        <linearGradient id="colorForeRaw" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#71717a" stopOpacity={0.08} />
                          <stop offset="95%" stopColor="#71717a" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
                      <XAxis
                        dataKey="displayDate"
                        tickLine={false}
                        axisLine={false}
                        tick={{ fontSize: 10, fill: "#71717a", fontFamily: "monospace" }}
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        tickLine={false}
                        axisLine={false}
                        tick={{ fontSize: 10, fill: "#71717a", fontFamily: "monospace" }}
                        tickFormatter={(val) => `${getCurrencySymbol()}${val.toLocaleString()}`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#ffffff",
                          border: "1px solid #e4e4e7",
                          borderRadius: "12px",
                          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
                          fontSize: "11px",
                          fontFamily: "monospace"
                        }}
                        formatter={(value, name) => [
                          `${getCurrencySymbol()}${parseFloat(value).toLocaleString("en-US", {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                          })}`,
                          name
                        ]}
                      />
                      <Area
                        type="monotone"
                        dataKey="historical"
                        stroke="#18181b"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorHist)"
                        dot={false}
                        name="Historical"
                      />
                      {data.model_usable && chartData.some(d => d.forecastRaw !== null) && (
                        <Area
                          type="monotone"
                          dataKey="forecastRaw"
                          stroke="#71717a"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          fillOpacity={1}
                          fill="url(#colorForeRaw)"
                          dot={{ r: 3, fill: "#71717a" }}
                          name="Raw ML Model"
                        />
                      )}
                      {data.model_usable && (
                        <Area
                          type="monotone"
                          dataKey="forecast"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          fillOpacity={1}
                          fill="url(#colorFore)"
                          dot={{ r: 3, fill: "#3b82f6" }}
                          name="Geopolitical Adjusted"
                        />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 7-DAY FORECAST GRID */}
              {data.model_usable && data.forecast_chart.length > 0 && (
                <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col gap-4">
                  <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider font-mono">
                    7-Day Numerical Projection (BiLSTM)
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-7 gap-3">
                    {data.forecast_chart.map((row, idx) => {
                      const pctChange = ((row.close - data.last_close) / data.last_close) * 100;
                      return (
                        <div
                          key={idx}
                          className="bg-zinc-50 border border-zinc-200/50 rounded-xl p-3 text-center flex flex-col gap-1.5"
                        >
                          <div className="font-mono text-[9px] font-bold text-zinc-400">
                            {new Date(row.date).toLocaleDateString("en-US", { weekday: "short", day: "numeric" })}
                          </div>
                          <div className="font-mono text-sm font-extrabold text-zinc-900">
                            {getCurrencySymbol()}{row.close.toFixed(2)}
                          </div>
                          <div className={`font-mono text-[10px] font-bold ${pctChange >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                            {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* News and Macro column */}
            <div className="flex flex-col gap-6">
              {/* Tab Selector */}
              <div className="bg-white border border-zinc-200/80 rounded-2xl p-1 flex gap-1 shadow-sm">
                <button
                  onClick={() => setActiveTab("news")}
                  className={`flex-1 py-2 text-xs font-semibold rounded-xl transition-all ${
                    activeTab === "news" ? "bg-zinc-100 text-zinc-950" : "text-zinc-500 hover:text-zinc-950"
                  }`}
                >
                  Geopolitical Feed
                </button>
                <button
                  onClick={() => setActiveTab("macro")}
                  className={`flex-1 py-2 text-xs font-semibold rounded-xl transition-all ${
                    activeTab === "macro" ? "bg-zinc-100 text-zinc-950" : "text-zinc-500 hover:text-zinc-950"
                  }`}
                >
                  Macro Intelligence
                </button>
              </div>

              {/* Tab Content: News */}
              {activeTab === "news" && (
                <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col gap-4 max-h-[500px] overflow-y-auto">
                  <div className="flex items-center justify-between border-b border-zinc-100 pb-2">
                    <span className="font-mono text-xs font-bold text-zinc-400 uppercase">Headlines scanned today</span>
                    <span className={`font-mono text-[9px] font-bold px-2 py-0.5 rounded ${
                      data.news_is_live ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
                    }`}>
                      {data.news_source}
                    </span>
                  </div>

                  <div className="flex flex-col gap-3">
                    {data.news.map((item, idx) => (
                      <div key={idx} className="border-b border-zinc-100 last:border-b-0 pb-3 last:pb-0">
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <span className="font-mono text-[9px] font-semibold text-zinc-400">{item.date}</span>
                          <div className="flex gap-1.5 items-center">
                            {item.impact && item.impact !== "neutral" && (
                              <span
                                title={item.impact_reason}
                                className={`font-mono text-[9px] font-bold px-1.5 py-0.5 rounded cursor-help ${
                                  item.impact === "bullish"
                                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                    : "bg-rose-50 text-rose-700 border border-rose-200"
                                }`}
                              >
                                {item.impact === "bullish" ? "↑ Buy" : "↓ Sell"}
                              </span>
                            )}
                            <span className={`font-mono text-[9px] font-bold ${item.sentiment >= 0.05 ? "text-emerald-600" : item.sentiment <= -0.05 ? "text-rose-600" : "text-zinc-400"}`}>
                              Tone {item.sentiment >= 0 ? "+" : ""}{item.sentiment.toFixed(2)}
                            </span>
                          </div>
                        </div>
                        <p className="text-xs text-zinc-800 font-semibold leading-normal">{item.title}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tab Content: Macro */}
              {activeTab === "macro" && (
                <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-sm flex flex-col gap-4">
                  <span className="font-mono text-xs font-bold text-zinc-400 uppercase border-b border-zinc-100 pb-2">
                    Macro indicators status
                  </span>
                  <div className="flex flex-col gap-4">
                    {data.macro.map((item, idx) => {
                      const isNifty = ticker === "^NSEI";
                      const assetLabel = isNifty ? "Nifty" : "Crude";
                      
                      // Determine bad vs good trend
                      const isUSDINR = item.key === "USD_INR";
                      const isVIX = item.key === "India_VIX" || item.key === "VIX";
                      const isGold = item.key === "Gold";
                      const isCrude = item.key === "Crude_Oil";

                      let inverseGood = false;
                      if (isNifty) {
                        if (isUSDINR || isVIX || isGold || isCrude) inverseGood = true;
                      } else {
                        if (item.key === "DXY" || isVIX) inverseGood = true;
                      }

                      const isBad = (inverseGood && item.trend === "up") || (!inverseGood && item.trend === "down");

                      // Factual change direction with asset-impact colors
                      const changeIsPositive = item.change_pct >= 0;
                      const changeSign = changeIsPositive ? "+" : "";
                      const changeColorClass = isBad ? "text-rose-600" : "text-emerald-600";

                      return (
                        <div key={idx} className="flex items-center justify-between border-b border-zinc-100 last:border-b-0 pb-3.5 last:pb-0">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-xs font-bold text-zinc-950">{item.label}</span>
                              <span className={`font-mono text-[9px] font-bold px-1.5 py-0.5 rounded ${
                                isBad 
                                  ? "bg-rose-50 text-rose-700 border border-rose-100" 
                                  : "bg-emerald-50 text-emerald-700 border border-emerald-100"
                              }`}>
                                {isBad ? `✗ Bearish` : `✓ Bullish`}
                              </span>
                            </div>
                            <div className="font-mono text-[10px] text-zinc-400">{item.key}</div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono text-xs font-extrabold text-zinc-900">
                              {item.unit === "$" || item.unit === "₹" ? item.unit : ""}{item.value.toLocaleString("en-US", { minimumFractionDigits: 2 })}{item.unit !== "$" && item.unit !== "₹" ? item.unit : ""}
                            </div>
                            <div className={`font-mono text-[10px] font-bold ${changeColorClass}`}>
                              {item.trend === "up" ? "▲" : "▼"} {changeSign}{item.change_pct.toFixed(2)}% (30d)
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* QUANTITATIVE ANALYST CHAT */}
          <section className="bg-white border border-zinc-200/80 rounded-2xl shadow-sm overflow-hidden flex flex-col h-[500px]">
            <div className="bg-zinc-50 border-b border-zinc-200/60 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-zinc-500" />
                <h3 className="text-sm font-bold text-zinc-950 tracking-tight">Geopolitical Analyst Chat Terminal</h3>
              </div>
              <span className="font-mono text-[9px] bg-zinc-200 text-zinc-600 px-2 py-0.5 rounded font-bold uppercase">
                Active Client Context
              </span>
            </div>

            {/* Chat Messages Log */}
            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5 bg-zinc-50/20">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[92%] rounded-2xl px-5 py-4 text-[15px] leading-relaxed shadow-sm ${
                      msg.role === "user"
                        ? "bg-zinc-900 text-white font-medium rounded-tr-none"
                        : "bg-white border border-zinc-200 text-zinc-950 rounded-tl-none font-medium"
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-zinc-200 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-2">
                    <span className="h-1.5 w-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="h-1.5 w-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="h-1.5 w-1.5 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input form */}
            <form onSubmit={handleChatSubmit} className="border-t border-zinc-200 px-6 py-4 flex gap-4 bg-white">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about geopolitical correlations, supply chain shock biases, or macro triggers..."
                className="flex-1 border border-zinc-200 rounded-xl px-4 py-2.5 text-sm text-zinc-950 focus:outline-none focus:ring-1 focus:ring-zinc-900 focus:border-zinc-900"
              />
              <button
                type="submit"
                disabled={chatLoading}
                className="bg-zinc-950 hover:bg-zinc-800 text-white px-5 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>Send</span>
                <Send className="h-3.5 w-3.5" />
              </button>
            </form>
          </section>
        </>
      )}
    </div>
  );
}
