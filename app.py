from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional fallback for first install issues
    st_autorefresh = None


APP_TITLE = "Trading Command Center"
LOCAL_TZ = ZoneInfo("Australia/Brisbane")
NY_TZ = ZoneInfo("America/New_York")

ASSETS = {
    "NVDA": {"name": "NVIDIA", "category": "Core AI"},
    "AVGO": {"name": "Broadcom", "category": "Core AI"},
    "TSM": {"name": "Taiwan Semiconductor", "category": "Core AI"},
    "AMD": {"name": "Advanced Micro Devices", "category": "Core AI"},
    "ASML": {"name": "ASML Holding", "category": "AI Infrastructure"},
    "AMAT": {"name": "Applied Materials", "category": "AI Infrastructure"},
    "LRCX": {"name": "Lam Research", "category": "AI Infrastructure"},
    "MU": {"name": "Micron Technology", "category": "AI Infrastructure"},
    "SNPS": {"name": "Synopsys", "category": "AI Infrastructure"},
    "CDNS": {"name": "Cadence Design Systems", "category": "AI Infrastructure"},
    "APP": {"name": "AppLovin", "category": "Asymmetric AI"},
    "IONQ": {"name": "IonQ", "category": "Asymmetric AI"},
    "CBRS": {"name": "Cerebras Systems", "category": "Asymmetric AI"},
    "NVTS": {"name": "Navitas Semiconductor", "category": "Asymmetric AI"},
    "VST": {"name": "Vistra", "category": "Energy / Nuclear"},
    "LEU": {"name": "Centrus Energy", "category": "Energy / Nuclear"},
    "RKLB": {"name": "Rocket Lab", "category": "Defense / Space"},
    "KTOS": {"name": "Kratos Defense & Security", "category": "Defense / Space"},
    "IBIT": {"name": "iShares Bitcoin Trust", "category": "Crypto"},
    "IREN": {"name": "IREN", "category": "Crypto"},
    "BTC-USD": {"name": "Bitcoin", "category": "Crypto"},
    "USAR": {"name": "USA Rare Earth", "category": "Strategic Materials"},
    "QQQ": {"name": "Invesco QQQ Trust", "category": "ETFs"},
    "PRIO3.SA": {"name": "PRIO", "category": "Brasil"},
    "VALE3.SA": {"name": "Vale", "category": "Brasil"},
}

MACRO = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^VIX": "VIX",
    "BTC-USD": "Bitcoin",
    "AUDBRL=X": "AUD/BRL",
    "BRL=X": "USD/BRL",
    "TIO=F": "Minério",
    "CL=F": "Petróleo",
}


@dataclass
class MarketStatus:
    label: str
    detail: str
    is_open: bool


def load_css() -> None:
    with open("style.css", "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


def fmt_price(value: float | None, currency: str = "$") -> str:
    if value is None or pd.isna(value):
        return "--"
    if abs(value) >= 1000:
        return f"{currency}{value:,.2f}"
    if abs(value) < 1:
        return f"{currency}{value:,.4f}"
    return f"{currency}{value:,.2f}"


def fmt_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:,.0f}"


def market_status() -> MarketStatus:
    now_ny = datetime.now(NY_TZ)
    if now_ny.weekday() >= 5:
        return MarketStatus("Fechado", "Fim de semana nos EUA", False)

    current = now_ny.time()
    if time(4, 0) <= current < time(9, 30):
        return MarketStatus("Pré-market", now_ny.strftime("%H:%M NY"), False)
    if time(9, 30) <= current < time(16, 0):
        return MarketStatus("Aberto", now_ny.strftime("%H:%M NY"), True)
    if time(16, 0) <= current < time(20, 0):
        return MarketStatus("After-hours", now_ny.strftime("%H:%M NY"), False)
    return MarketStatus("Fechado", now_ny.strftime("%H:%M NY"), False)


@st.cache_data(ttl=8, show_spinner=False)
def fetch_history(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=True,
            threads=False,
            timeout=8,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fast_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "market_cap": getattr(info, "market_cap", None) or info.get("marketCap"),
            "currency": getattr(info, "currency", None) or info.get("currency", "USD"),
            "last_price": getattr(info, "last_price", None) or info.get("lastPrice"),
            "previous_close": getattr(info, "regular_market_previous_close", None)
            or info.get("regularMarketPreviousClose")
            or getattr(info, "previous_close", None)
            or info.get("previousClose"),
            "last_volume": getattr(info, "last_volume", None) or info.get("lastVolume"),
        }
    except Exception:
        return {"market_cap": None, "currency": "USD", "last_price": None, "previous_close": None, "last_volume": None}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_macro_snapshot() -> dict:
    rows = {}
    for ticker, label in MACRO.items():
        if ticker == "TIO=F":
            hist = fetch_history(ticker, "5d", "1d")
        else:
            hist = fetch_history(ticker, "5d", "15m")
        rows[ticker] = build_quote(ticker, label, "Macro", hist, fetch_fast_info(ticker))
    return rows


@st.cache_data(ttl=45, show_spinner=False)
def fetch_asset_rows(tickers: tuple[str, ...]) -> list[dict]:
    rows = []
    for ticker in tickers:
        meta = ASSETS[ticker]
        hist = fetch_history(ticker, "5d", "5m")
        info = fetch_fast_info(ticker)
        row = build_quote(ticker, meta["name"], meta["category"], hist, info)
        row["market_cap"] = info.get("market_cap")
        row["currency"] = "R$" if ticker.endswith(".SA") else "$"
        row["rsi"] = latest_rsi(hist)
        row["trend"] = classify_trend(hist, row["pct_change"], row["rsi"])
        row["alerts"] = detect_alerts(row, hist)
        rows.append(row)
    return rows


@st.cache_data(ttl=120, show_spinner=False)
def fetch_bitcoin_extras() -> dict:
    extras = {
        "fear_greed": None,
        "dominance": None,
        "funding": None,
        "etf_flow_proxy": None,
    }

    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=4)
        data = r.json()["data"][0]
        extras["fear_greed"] = f"{data['value']} - {data['value_classification']}"
    except Exception:
        pass

    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=4)
        dominance = r.json()["data"]["market_cap_percentage"]["btc"]
        extras["dominance"] = f"{dominance:.2f}%"
    except Exception:
        pass

    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=4)
        funding = float(r.json()["lastFundingRate"]) * 100
        extras["funding"] = f"{funding:.4f}%"
    except Exception:
        pass

    ibit = fetch_history("IBIT", "5d", "15m")
    if not ibit.empty and len(ibit) > 12:
        recent_volume = ibit["Volume"].tail(12).mean()
        normal_volume = ibit["Volume"].tail(80).mean()
        if normal_volume:
            extras["etf_flow_proxy"] = f"{recent_volume / normal_volume:.2f}x volume"
    return extras


def build_quote(ticker: str, name: str, category: str, hist: pd.DataFrame, info: dict | None = None) -> dict:
    info = info or {}
    if hist.empty or "Close" not in hist:
        price = coalesce(info.get("last_price"), np.nan)
        previous_close = coalesce(info.get("previous_close"), price)
        nominal = price - previous_close if previous_close else 0.0
        pct = nominal / previous_close * 100 if previous_close else 0.0
        return {
            "ticker": ticker,
            "name": name,
            "category": category,
            "price": price,
            "pct_change": pct,
            "nominal_change": nominal,
            "volume": coalesce(info.get("last_volume"), np.nan),
            "series": [],
        }

    close = hist["Close"].dropna()
    price = coalesce(info.get("last_price"), float(close.iloc[-1]) if len(close) else np.nan)
    previous_close = coalesce(info.get("previous_close"), fallback_previous_close(close))
    nominal = price - previous_close if previous_close else 0.0
    pct = (nominal / previous_close * 100) if previous_close else 0.0
    fallback_volume = float(hist["Volume"].dropna().iloc[-1]) if "Volume" in hist and not hist["Volume"].dropna().empty else np.nan

    return {
        "ticker": ticker,
        "name": name,
        "category": category,
        "price": price,
        "pct_change": pct,
        "nominal_change": nominal,
        "volume": coalesce(info.get("last_volume"), fallback_volume),
        "series": close.tail(42).tolist(),
    }


def coalesce(*values: float | None) -> float:
    for value in values:
        if value is not None and not pd.isna(value):
            return float(value)
    return np.nan


def fallback_previous_close(close: pd.Series) -> float:
    if len(close) >= 2:
        return float(close.iloc[-2])
    if len(close) == 1:
        return float(close.iloc[0])
    return np.nan


def latest_rsi(hist: pd.DataFrame, window: int = 14) -> float:
    if hist.empty or "Close" not in hist or len(hist) < window + 2:
        return np.nan
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["RSI"] = compute_rsi(out["Close"])
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def classify_trend(hist: pd.DataFrame, pct_change: float, rsi: float) -> str:
    if hist.empty or len(hist) < 24:
        return "neutra"
    close = hist["Close"].dropna()
    ma_fast = close.tail(8).mean()
    ma_slow = close.tail(24).mean()
    if ma_fast > ma_slow and pct_change > 0.35 and (pd.isna(rsi) or rsi < 78):
        return "bullish"
    if ma_fast < ma_slow and pct_change < -0.35:
        return "bearish"
    return "neutra"


def detect_alerts(row: dict, hist: pd.DataFrame) -> list[tuple[str, str]]:
    alerts = []
    pct = row.get("pct_change", 0) or 0
    if pct >= 4:
        alerts.append(("🚀 Forte alta", f"{row['ticker']} sobe {pct:.2f}% no dia."))
    elif pct <= -4:
        alerts.append(("⚠️ Queda forte", f"{row['ticker']} cai {pct:.2f}% no dia."))

    if hist.empty or len(hist) < 30:
        return alerts

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)
    recent_high = close.tail(6).max()
    resistance = close.tail(80).iloc[:-6].max() if len(close) > 86 else close.iloc[:-6].max()
    if resistance and recent_high > resistance * 1.003:
        alerts.append(("💥 Breakout", f"{row['ticker']} rompe resistência intraday."))

    if len(volume) > 25:
        vol_ratio = volume.tail(5).mean() / max(volume.tail(60).mean(), 1)
        if vol_ratio > 2.2:
            alerts.append(("🔥 Momentum", f"Volume recente em {vol_ratio:.1f}x a média."))
        if vol_ratio > 3.4 and abs(pct) > 2:
            alerts.append(("🐳 Whale movement", f"Fluxo anormal detectado em {row['ticker']}."))

    volatility = close.pct_change().tail(24).std() * math.sqrt(24) * 100
    if volatility > 4.5:
        alerts.append(("⚠️ Alta volatilidade", f"Volatilidade intraday elevada: {volatility:.1f}%."))
    return alerts


def sparkline_svg(values: list[float], positive: bool) -> str:
    if not values:
        return ""
    clean = np.array([v for v in values if not pd.isna(v)], dtype=float)
    if len(clean) < 2:
        return ""
    low, high = float(clean.min()), float(clean.max())
    spread = high - low if high != low else 1
    width, height, pad = 280, 48, 3
    points = []
    for idx, value in enumerate(clean):
        x = pad + idx * ((width - pad * 2) / (len(clean) - 1))
        y = height - pad - ((value - low) / spread) * (height - pad * 2)
        points.append(f"{x:.1f},{y:.1f}")
    color = "#35f2a0" if positive else "#ff4f68"
    fill_points = f"{points[0]} {' '.join(points)} {points[-1].split(',')[0]},{height} {points[0].split(',')[0]},{height}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".32"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<polygon fill="url(#spark)" points="{fill_points}"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}"/>'
        f'</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="sparkline" alt="Mini gráfico intraday" src="data:image/svg+xml;base64,{encoded}">'


def asset_card(row: dict) -> str:
    pct = row.get("pct_change", 0) or 0
    positive = pct >= 0
    color_class = "positive" if positive else "negative"
    hot = "hot" if abs(pct) >= 3 or len(row.get("alerts", [])) >= 2 else ""
    trend = row.get("trend", "neutra")
    trend_class = "positive" if trend == "bullish" else "negative" if trend == "bearish" else "neutral"
    currency = row.get("currency", "$")
    spark = sparkline_svg(row.get("series", []), positive)
    rsi = "--" if pd.isna(row.get("rsi")) else f"{row.get('rsi'):.1f}"
    return (
        f'<div class="asset-card {hot}">'
        f'<div class="asset-head"><div><div class="ticker">{row["ticker"]}</div><div class="company">{row["name"]}</div></div>'
        f'<div class="trend-badge {trend_class}">{trend.upper()}</div></div>'
        f'<div class="price-row"><div class="price">{fmt_price(row.get("price"), currency)}</div>'
        f'<div class="change {color_class}">{pct:+.2f}%<br><span style="font-size:12px">{row.get("nominal_change", 0):+.2f}</span></div></div>'
        f'{spark}'
        f'<div class="card-stats">'
        f'<div class="stat"><span class="card-label">Volume</span><strong>{fmt_number(row.get("volume"))}</strong></div>'
        f'<div class="stat"><span class="card-label">M. Cap</span><strong>{fmt_number(row.get("market_cap"))}</strong></div>'
        f'<div class="stat"><span class="card-label">RSI</span><strong>{rsi}</strong></div>'
        f'</div></div>'
    )


def render_header(status: MarketStatus, macro: dict) -> None:
    now = datetime.now(LOCAL_TZ).strftime("%H:%M:%S Brisbane")
    cards = "".join(macro_card(ticker, label, macro.get(ticker, {})) for ticker, label in MACRO.items())
    html = (
        '<div class="hero-shell">'
        '<div class="topbar">'
        f'<div><h1 class="brand-title">{APP_TITLE}</h1>'
        '<div class="brand-subtitle">AI stocks, Bitcoin, ETFs, Brasil e macro em uma segunda tela.</div></div>'
        f'<div class="status-pill">Mercado: <b>{status.label}</b> · {status.detail} · {now}</div>'
        '</div>'
        f'<div class="macro-grid">{cards}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def macro_card(ticker: str, label: str, row: dict) -> str:
    pct = row.get("pct_change", 0) or 0
    klass = "positive" if pct >= 0 else "negative"
    currency = "R$" if ticker in {"AUDBRL=X", "BRL=X"} else "" if ticker.startswith("^") else "$"
    value = fmt_price(row.get("price"), currency).replace("$", "", 1) if currency == "" else fmt_price(row.get("price"), currency)
    return f'<div class="macro-card"><div class="macro-label">{label}</div><div class="macro-value">{value}</div><div class="macro-change {klass}">{pct:+.2f}%</div></div>'


def market_mood(rows: list[dict], macro: dict) -> tuple[int, str, list[str]]:
    lookup = {row["ticker"]: row for row in rows}
    ai_complex = ["NVDA", "AVGO", "TSM", "AMD", "ASML", "AMAT", "LRCX", "MU", "SNPS", "CDNS", "APP", "IONQ", "CBRS", "NVTS"]
    semis = [lookup[t]["pct_change"] for t in ai_complex if t in lookup]
    semis_score = np.nanmean(semis) if semis else 0
    nasdaq = macro.get("^IXIC", {}).get("pct_change", 0) or 0
    btc = lookup.get("BTC-USD", {}).get("pct_change", 0) or 0
    vix = macro.get("^VIX", {}).get("pct_change", 0) or 0
    volume_hits = sum(1 for row in rows if any("Volume" in alert[1] or "Fluxo" in alert[1] for alert in row.get("alerts", [])))

    raw = 50 + semis_score * 4 + nasdaq * 7 + btc * 1.8 - vix * 1.7 + volume_hits * 2
    score = int(np.clip(raw, 0, 100))
    label = "Bullish" if score >= 62 else "Bearish" if score <= 42 else "Neutro"
    reasons = [
        f"AI complex: {semis_score:+.2f}%",
        f"Nasdaq: {nasdaq:+.2f}%",
        f"Bitcoin: {btc:+.2f}%",
        f"VIX: {vix:+.2f}%",
    ]
    return score, label, reasons


def render_radar(rows: list[dict]) -> None:
    alerts = []
    for row in rows:
        for title, copy in row.get("alerts", []):
            alerts.append((abs(row.get("pct_change", 0)), title, copy))
    alerts = sorted(alerts, reverse=True)[:8]
    if not alerts:
        st.markdown(
            '<div class="alert-card"><div class="alert-title">Radar limpo</div><div class="alert-copy">Nenhum movimento extremo neste ciclo.</div></div>',
            unsafe_allow_html=True,
        )
        return
    for _, title, copy in alerts:
        st.markdown(f'<div class="alert-card"><div class="alert-title">{title}</div><div class="alert-copy">{copy}</div></div>', unsafe_allow_html=True)


def render_btc_center(rows: list[dict], extras: dict) -> None:
    btc = next((row for row in rows if row["ticker"] == "BTC-USD"), {})
    btc_brl = build_quote("BTC-BRL", "Bitcoin BRL", "Cripto", fetch_history("BTC-BRL", "1d", "5m"))
    hist = fetch_history("BTC-USD", "5d", "15m")
    close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    support = close.tail(120).quantile(0.18) if len(close) else np.nan
    resistance = close.tail(120).quantile(0.86) if len(close) else np.nan

    cols = st.columns(6)
    cols[0].metric("BTC/USD", fmt_price(btc.get("price"), "$"), f"{btc.get('pct_change', 0):+.2f}%")
    cols[1].metric("BTC/BRL", fmt_price(btc_brl.get("price"), "R$"), f"{btc_brl.get('pct_change', 0):+.2f}%")
    cols[2].metric("Dominância", extras.get("dominance") or "--")
    cols[3].metric("Fear & Greed", extras.get("fear_greed") or "--")
    cols[4].metric("Funding", extras.get("funding") or "--")
    cols[5].metric("Fluxo ETF", extras.get("etf_flow_proxy") or "--")
    st.caption(f"Suporte técnico: {fmt_price(support, '$')} · Resistência: {fmt_price(resistance, '$')}")
    st.plotly_chart(price_chart("BTC-USD", "5d", "15m"), use_container_width=True, config={"displayModeBar": False})


def price_chart(ticker: str, period: str, interval: str) -> go.Figure:
    df = add_indicators(fetch_history(ticker, period, interval))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.54, 0.16, 0.15, 0.15],
    )
    if df.empty:
        fig.add_annotation(text="Sem dados disponíveis agora", showarrow=False)
        return style_figure(fig)

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Preço", line=dict(color="#50a7ff", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20", line=dict(color="#8b5cf6", width=1.4)), row=1, col=1)
    colors = np.where(df["Close"].diff().fillna(0) >= 0, "#35f2a0", "#ff4f68")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.58), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="#ffbf69", width=1.6)), row=3, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dot", line_color="rgba(255,255,255,.25)", row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dot", line_color="rgba(255,255,255,.25)", row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="#35f2a0", width=1.4)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode="lines", name="Signal", line=dict(color="#ff4f68", width=1.2)), row=4, col=1)
    return style_figure(fig)


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=10, r=10, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font=dict(color="#eef4ff", family="Inter, SF Pro Display, sans-serif"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def sidebar_controls() -> tuple[list[str], str, str]:
    st.sidebar.title("Command Center")
    st.sidebar.caption("Filtros e janelas de análise")
    categories = list(dict.fromkeys(meta["category"] for meta in ASSETS.values()))
    selected_categories = st.sidebar.multiselect("Categorias", categories, default=categories)
    visible = [ticker for ticker, meta in ASSETS.items() if meta["category"] in selected_categories]
    selected_ticker = st.sidebar.selectbox("Ativo para gráfico avançado", visible or list(ASSETS), index=0)
    range_label = st.sidebar.radio("Janela", ["Intraday", "5 dias", "1 mês", "1 ano"], horizontal=False)
    return visible, selected_ticker, range_label


def range_to_yf(label: str) -> tuple[str, str]:
    return {
        "Intraday": ("1d", "5m"),
        "5 dias": ("5d", "15m"),
        "1 mês": ("1mo", "1h"),
        "1 ano": ("1y", "1d"),
    }[label]


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    load_css()

    status = market_status()
    refresh_seconds = 10 if status.is_open else 60
    if st_autorefresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="market_refresh")

    visible_tickers, selected_ticker, range_label = sidebar_controls()
    macro = fetch_macro_snapshot()
    rows = fetch_asset_rows(tuple(visible_tickers))
    rows = sorted(rows, key=lambda row: row.get("pct_change", 0), reverse=True)

    render_header(status, macro)

    score, label, reasons = market_mood(fetch_asset_rows(tuple(ASSETS.keys())), macro)
    m1, m2 = st.columns([0.72, 0.28], gap="large")

    with m1:
        st.markdown('<div class="section-title"><h2>Grid Principal</h2><span class="small-note">Ordenado por maior alta do dia</span></div>', unsafe_allow_html=True)
        st.markdown(f"<div class='asset-grid'>{''.join(asset_card(row) for row in rows)}</div>", unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="section-title"><h2>AI Market Mood</h2></div>', unsafe_allow_html=True)
        mood_class = "positive" if label == "Bullish" else "negative" if label == "Bearish" else "amber"
        st.markdown(
            f"""
            <div class="metric-panel">
              <div class="card-label">Sentimento automático</div>
              <div class="mood-score {mood_class}">{score}</div>
              <div class="{mood_class}" style="font-weight:780;margin-top:6px">{label.upper()}</div>
              <div class="small-note" style="margin-top:11px">{' · '.join(reasons)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title"><h2>Radar de Oportunidades</h2></div>', unsafe_allow_html=True)
        render_radar(fetch_asset_rows(tuple(ASSETS.keys())))

    st.markdown('<div class="section-title"><h2>Bitcoin Center</h2><span class="small-note">Dados públicos gratuitos, com fallback resiliente</span></div>', unsafe_allow_html=True)
    render_btc_center(fetch_asset_rows(tuple(ASSETS.keys())), fetch_bitcoin_extras())

    period, interval = range_to_yf(range_label)
    st.markdown(f'<div class="section-title"><h2>Gráfico Avançado · {selected_ticker}</h2><span class="small-note">{range_label} · MA20 · Volume · RSI · MACD</span></div>', unsafe_allow_html=True)
    st.plotly_chart(price_chart(selected_ticker, period, interval), use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    st.caption(
        f"Atualização automática a cada {refresh_seconds}s. Dados via yfinance e APIs públicas gratuitas. "
        "Para trading ao vivo profissional, valide preços com uma fonte de mercado em tempo real."
    )


if __name__ == "__main__":
    main()
