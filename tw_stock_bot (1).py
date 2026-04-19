from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)

TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8584415843:AAFDvgxqetm_90_M_ZVMm5MB36NCLllTQZM",
)

hot_stocks = {
    "台積電": "2330.TW",
    "鴻海": "2317.TW",
    "聯發科": "2454.TW",
    "台達電": "2308.TW",
    "富邦金": "2881.TW",
    "國泰金": "2882.TW",
    "中華電": "2412.TW",
    "統一": "1216.TW",
    "大立光": "3008.TW",
    "聯電": "2303.TW",
}

DAYTRADE_CANDIDATES = {
    **hot_stocks,
    "中信金": "2891.TW",
    "兆豐金": "2886.TW",
    "玉山金": "2884.TW",
    "元大金": "2885.TW",
    "第一金": "2892.TW",
    "華南金": "2880.TW",
    "永豐金": "2890.TW",
    "合庫金": "5880.TW",
    "廣達": "2382.TW",
    "仁寶": "2324.TW",
    "緯創": "3231.TW",
    "英業達": "2356.TW",
    "華碩": "2357.TW",
    "宏碁": "2353.TW",
    "日月光投控": "3711.TW",
    "欣興": "3037.TW",
    "南亞科": "2408.TW",
    "創意": "3443.TW",
    "世芯": "3661.TW",
    "技嘉": "2376.TW",
    "微星": "2377.TW",
    "長榮": "2603.TW",
    "陽明": "2609.TW",
    "萬海": "2615.TW",
    "中鋼": "2002.TW",
    "台泥": "1101.TW",
    "亞泥": "1102.TW",
    "南亞": "1303.TW",
    "台塑": "1301.TW",
    "台化": "1326.TW",
    "國巨": "2327.TW",
    "華通": "2313.TW",
    "光寶科": "2301.TW",
    "聯詠": "3034.TW",
    "瑞昱": "2379.TW",
    "群創": "3481.TW",
    "友達": "2409.TW",
    "開發金": "2883.TW",
    "臺企銀": "2834.TW",
    "0050": "0050.TW",
    "0056": "0056.TW",
    "006208": "006208.TW",
}

DISCLAIMER = "⚠️ 免責聲明：本內容僅供研究與教學參考，不構成任何投資邀約或獲利保證，實際交易請自行判斷並控管風險。"


@dataclass
class BasicQuote:
    symbol: str
    name: str
    price: float
    previous_close: float
    volume: int
    open_price: float
    high_price: float
    low_price: float
    trade_date: str

    @property
    def change(self) -> float:
        return self.price - self.previous_close

    @property
    def change_percent(self) -> float:
        if self.previous_close == 0:
            return 0.0
        return (self.change / self.previous_close) * 100


class StockDataError(Exception):
    """股價資料取得失敗。"""


def normalize_stock_symbol(raw: str) -> str:
    stock_id = raw.strip().upper()
    if not stock_id:
        return stock_id
    if stock_id in hot_stocks:
        return hot_stocks[stock_id]
    if stock_id in DAYTRADE_CANDIDATES.values():
        return stock_id
    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        return stock_id
    if stock_id.isdigit():
        if stock_id.startswith("7"):
            return f"{stock_id}.TWO"
        return f"{stock_id}.TW"
    return stock_id


def strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    return df


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def format_price(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def signal_emoji(score: int) -> str:
    if score >= 1:
        return "🟢"
    if score <= -1:
        return "🔴"
    return "🟡"


def safe_name_lookup(ticker: yf.Ticker, symbol: str) -> str:
    for attr in ("fast_info", "info"):
        try:
            data = getattr(ticker, attr)
            if isinstance(data, dict):
                name = data.get("shortName") or data.get("longName") or data.get("displayName")
                if name:
                    return str(name)
        except Exception:
            continue
    return symbol


def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
    hist = strip_timezone(hist)
    if hist.empty:
        raise StockDataError(f"無法取得 {symbol} 的歷史資料。")
    hist = hist.dropna(how="all")
    return hist


def get_basic_quote_sync(symbol: str) -> BasicQuote:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="10d", interval="1d", auto_adjust=False, actions=False)
    hist = strip_timezone(hist).dropna(how="all")
    if hist.empty or "Close" not in hist.columns:
        raise StockDataError(f"❌ 無法找到股票代碼 {symbol} 的資訊，請確認代碼是否正確。")

    latest = hist.iloc[-1]
    prev_idx = -2 if len(hist) >= 2 else -1
    previous = hist.iloc[prev_idx]

    price = to_float(latest.get("Close"))
    previous_close = to_float(previous.get("Close"), default=price)
    volume = to_int(latest.get("Volume"), default=0)
    open_price = to_float(latest.get("Open"), default=price)
    high_price = to_float(latest.get("High"), default=price)
    low_price = to_float(latest.get("Low"), default=price)
    trade_date = hist.index[-1].strftime("%Y-%m-%d")

    if pd.isna(price):
        raise StockDataError(f"❌ 無法獲取股票代碼 {symbol} 的完整資訊。")

    name = safe_name_lookup(ticker, symbol)

    return BasicQuote(
        symbol=symbol,
        name=name,
        price=price,
        previous_close=previous_close,
        volume=volume,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        trade_date=trade_date,
    )


def format_basic_quote(quote: BasicQuote) -> str:
    if quote.change > 0:
        emoji = "📈"
        change_str = f"▲ {quote.change:.2f} ({quote.change_percent:+.2f}%)"
    elif quote.change < 0:
        emoji = "📉"
        change_str = f"▼ {abs(quote.change):.2f} ({quote.change_percent:+.2f}%)"
    else:
        emoji = "↔️"
        change_str = "0.00 (0.00%)"

    amplitude = 0.0
    if quote.open_price:
        amplitude = (quote.high_price - quote.low_price) / quote.open_price * 100

    return (
        f"{emoji} {quote.name} ({quote.symbol})\n"
        f"📅 日期: {quote.trade_date}\n"
        f"💰 現價: {quote.price:.2f}\n"
        f"📊 漲跌: {change_str}\n"
        f"🏁 開盤: {quote.open_price:.2f}｜最高: {quote.high_price:.2f}｜最低: {quote.low_price:.2f}\n"
        f"📦 成交量: {quote.volume:,}\n"
        f"📐 振幅: {amplitude:.2f}%"
    )


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def calculate_kd(df: pd.DataFrame, period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    spread = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["Close"] - low_min) / spread) * 100

    k_values: list[float] = []
    d_values: list[float] = []
    last_k = 50.0
    last_d = 50.0
    for value in rsv:
        if pd.isna(value):
            k_values.append(np.nan)
            d_values.append(np.nan)
            continue
        last_k = (2 / 3) * last_k + (1 / 3) * float(value)
        last_d = (2 / 3) * last_d + (1 / 3) * last_k
        k_values.append(last_k)
        d_values.append(last_d)

    k = pd.Series(k_values, index=df.index, name="K")
    d = pd.Series(d_values, index=df.index, name="D")
    return k, d, rsv


def calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calculate_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def compute_support_resistance(df: pd.DataFrame, current_price: float) -> dict[str, Any]:
    recent = df.tail(60)
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())
    diff = recent_high - recent_low
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    fib_levels = {ratio: recent_low + diff * ratio for ratio in fib_ratios}
    levels = sorted(set([recent_low, recent_high, *fib_levels.values()]))

    supports = sorted([level for level in levels if level <= current_price], reverse=True)[:3]
    resistances = sorted([level for level in levels if level >= current_price])[:3]

    return {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "fib_levels": fib_levels,
        "supports": supports,
        "resistances": resistances,
    }


def compute_daily_analysis(symbol: str) -> dict[str, Any]:
    quote = get_basic_quote_sync(symbol)
    hist = fetch_history(symbol, period="450d", interval="1d")
    if len(hist) < 80:
        raise StockDataError(f"{symbol} 的歷史資料不足，暫時無法完整分析。")

    df = hist.copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    close = df["Close"]
    volume = df["Volume"]

    df["RSI14"] = calculate_rsi(close, 14)
    df["K9"], df["D9"], df["RSV9"] = calculate_kd(df, 9)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calculate_macd(close)

    for ma in [5, 10, 20, 60, 120, 240]:
        df[f"MA{ma}"] = close.rolling(ma).mean()

    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = calculate_bollinger(close, 20, 2.0)
    df["VOL_MA5_PREV"] = volume.shift(1).rolling(5).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    support_info = compute_support_resistance(df, float(latest["Close"]))

    signals: list[dict[str, Any]] = []

    rsi = to_float(latest["RSI14"])
    prev_rsi = to_float(prev["RSI14"])
    if rsi <= 30:
        score = 2
        detail = f"RSI {rsi:.2f}，落入超賣區"
    elif rsi < 40:
        score = 1
        detail = f"RSI {rsi:.2f}，偏弱但接近低檔"
    elif rsi >= 70:
        score = -2
        detail = f"RSI {rsi:.2f}，進入超買區"
    elif rsi > 60:
        score = -1
        detail = f"RSI {rsi:.2f}，短線偏熱"
    else:
        score = 0
        detail = f"RSI {rsi:.2f}，位於中性區"
    if prev_rsi < 30 <= rsi:
        score = min(2, score + 1)
        detail += "，且自超賣區反彈"
    elif prev_rsi > 70 >= rsi:
        score = max(-2, score - 1)
        detail += "，且自超買區轉弱"
    signals.append({"name": "RSI(14)", "score": score, "detail": detail})

    k_now = to_float(latest["K9"])
    d_now = to_float(latest["D9"])
    k_prev = to_float(prev["K9"])
    d_prev = to_float(prev["D9"])
    if k_now > d_now and k_prev <= d_prev and k_now < 30:
        score = 2
        detail = f"K {k_now:.2f} / D {d_now:.2f}，低檔黃金交叉"
    elif k_now > d_now and k_now < 50:
        score = 1
        detail = f"K {k_now:.2f} / D {d_now:.2f}，短線偏多"
    elif k_now < d_now and k_prev >= d_prev and k_now > 70:
        score = -2
        detail = f"K {k_now:.2f} / D {d_now:.2f}，高檔死亡交叉"
    elif k_now < d_now and k_now > 50:
        score = -1
        detail = f"K {k_now:.2f} / D {d_now:.2f}，短線偏空"
    else:
        score = 0
        detail = f"K {k_now:.2f} / D {d_now:.2f}，訊號中性"
    signals.append({"name": "KD(9,9)", "score": score, "detail": detail})

    macd_now = to_float(latest["MACD"])
    sig_now = to_float(latest["MACD_SIGNAL"])
    hist_now = to_float(latest["MACD_HIST"])
    macd_prev = to_float(prev["MACD"])
    sig_prev = to_float(prev["MACD_SIGNAL"])
    if macd_now > sig_now and macd_prev <= sig_prev:
        score = 2
        detail = f"MACD 黃金交叉，柱體 {hist_now:.4f}"
    elif macd_now > sig_now and hist_now > 0:
        score = 1
        detail = f"MACD 位於訊號線上方，柱體 {hist_now:.4f}"
    elif macd_now < sig_now and macd_prev >= sig_prev:
        score = -2
        detail = f"MACD 死亡交叉，柱體 {hist_now:.4f}"
    elif macd_now < sig_now and hist_now < 0:
        score = -1
        detail = f"MACD 位於訊號線下方，柱體 {hist_now:.4f}"
    else:
        score = 0
        detail = f"MACD {macd_now:.4f} / Signal {sig_now:.4f}，中性整理"
    signals.append({"name": "MACD(12,26,9)", "score": score, "detail": detail})

    ma_values = {ma: to_float(latest[f"MA{ma}"]) for ma in [5, 10, 20, 60, 120, 240]}
    close_now = to_float(latest["Close"])
    above_count = sum(close_now > value for value in ma_values.values() if not pd.isna(value))
    if close_now > ma_values[20] and ma_values[5] > ma_values[10] > ma_values[20] > ma_values[60]:
        score = 2
        detail = "股價站上中短期均線，且 MA5 > MA10 > MA20 > MA60"
    elif above_count >= 4:
        score = 1
        detail = f"股價位於 {above_count} 條均線之上"
    elif close_now < ma_values[20] and ma_values[5] < ma_values[10] < ma_values[20] < ma_values[60]:
        score = -2
        detail = "股價跌破中短期均線，且 MA5 < MA10 < MA20 < MA60"
    elif above_count <= 2:
        score = -1
        detail = f"股價僅位於 {above_count} 條均線之上"
    else:
        score = 0
        detail = "均線排列無明顯趨勢"
    signals.append({"name": "均線結構", "score": score, "detail": detail})

    bb_upper = to_float(latest["BB_UPPER"])
    bb_mid = to_float(latest["BB_MID"])
    bb_lower = to_float(latest["BB_LOWER"])
    band_width = bb_upper - bb_lower if not any(pd.isna([bb_upper, bb_lower])) else np.nan
    if not pd.isna(bb_lower) and close_now <= bb_lower:
        score = 2
        detail = f"股價觸及布林下軌 {bb_lower:.2f}，短線超跌"
    elif not pd.isna(band_width) and close_now < bb_mid and close_now <= bb_lower + band_width * 0.25:
        score = 1
        detail = "股價靠近布林下緣，留意反彈"
    elif not pd.isna(bb_upper) and close_now >= bb_upper:
        score = -2
        detail = f"股價觸及布林上軌 {bb_upper:.2f}，短線過熱"
    elif not pd.isna(band_width) and close_now > bb_mid and close_now >= bb_upper - band_width * 0.25:
        score = -1
        detail = "股價靠近布林上緣，追價風險升高"
    else:
        score = 0
        detail = "股價位於布林通道中段"
    signals.append({"name": "布林通道", "score": score, "detail": detail})

    vol_ma5_prev = to_float(latest["VOL_MA5_PREV"])
    vol_ratio = close_ratio = np.nan
    if not pd.isna(vol_ma5_prev) and vol_ma5_prev > 0:
        vol_ratio = float(latest["Volume"]) / vol_ma5_prev
    price_change_pct = (close_now - to_float(prev["Close"])) / to_float(prev["Close"], 1.0) * 100
    if not pd.isna(vol_ratio) and vol_ratio >= 1.8 and price_change_pct > 0:
        score = 1
        detail = f"量能放大至近 5 日均量的 {vol_ratio:.2f} 倍，價量偏多"
    elif not pd.isna(vol_ratio) and vol_ratio >= 1.8 and price_change_pct < 0:
        score = -1
        detail = f"量能放大至近 5 日均量的 {vol_ratio:.2f} 倍，但收黑需保守"
    elif not pd.isna(vol_ratio) and vol_ratio < 0.7 and price_change_pct < 0:
        score = -1
        detail = f"量能僅為近 5 日均量的 {vol_ratio:.2f} 倍，動能不足"
    else:
        score = 0
        detail = (
            f"成交量約為近 5 日均量的 {vol_ratio:.2f} 倍"
            if not pd.isna(vol_ratio)
            else "量能資料不足"
        )
    signals.append({"name": "成交量", "score": score, "detail": detail})

    nearest_support = support_info["supports"][0] if support_info["supports"] else support_info["recent_low"]
    nearest_resistance = support_info["resistances"][0] if support_info["resistances"] else support_info["recent_high"]
    support_gap = abs((close_now - nearest_support) / close_now * 100) if close_now else np.nan
    resistance_gap = abs((nearest_resistance - close_now) / close_now * 100) if close_now else np.nan
    if not pd.isna(support_gap) and support_gap <= 2 and resistance_gap >= 4:
        score = 1
        detail = f"股價接近支撐 {nearest_support:.2f}，上方仍有空間"
    elif not pd.isna(resistance_gap) and resistance_gap <= 2 and support_gap >= 4:
        score = -1
        detail = f"股價貼近壓力 {nearest_resistance:.2f}，短線易震盪"
    else:
        score = 0
        detail = f"位於區間中段，支撐 {nearest_support:.2f} / 壓力 {nearest_resistance:.2f}"
    signals.append({"name": "支撐壓力", "score": score, "detail": detail})

    raw_score = sum(item["score"] for item in signals)
    rating = max(0.0, min(10.0, round(((raw_score + 14) / 28) * 10, 1)))
    confidence = max(0, min(100, int(round(abs(raw_score) / 14 * 100))))

    if rating >= 8.0:
        recommendation = "🟢 強力買進"
    elif rating >= 6.2:
        recommendation = "🟢 買進"
    elif rating >= 4.0:
        recommendation = "🟡 觀望"
    elif rating >= 2.0:
        recommendation = "🔴 賣出"
    else:
        recommendation = "🔴 強力賣出"

    return {
        "quote": quote,
        "df": df,
        "latest": latest,
        "signals": signals,
        "support_info": support_info,
        "vol_ratio": vol_ratio,
        "rating": rating,
        "confidence": confidence,
        "recommendation": recommendation,
        "ma_values": ma_values,
    }


def format_daily_analysis_report(analysis: dict[str, Any]) -> str:
    quote: BasicQuote = analysis["quote"]
    latest = analysis["latest"]
    support_info = analysis["support_info"]
    ma_values = analysis["ma_values"]
    fib_levels = support_info["fib_levels"]

    indicator_lines = []
    for item in analysis["signals"]:
        indicator_lines.append(
            f"{signal_emoji(item['score'])} {item['name']}：{item['detail']}"
        )

    ma_text = (
        f"MA5 {format_price(ma_values[5])}｜MA10 {format_price(ma_values[10])}｜MA20 {format_price(ma_values[20])}\n"
        f"MA60 {format_price(ma_values[60])}｜MA120 {format_price(ma_values[120])}｜MA240 {format_price(ma_values[240])}"
    )

    boll_text = (
        f"上軌 {format_price(to_float(latest['BB_UPPER']))}｜中軌 {format_price(to_float(latest['BB_MID']))}｜"
        f"下軌 {format_price(to_float(latest['BB_LOWER']))}"
    )

    support_lines = []
    if support_info["supports"]:
        support_lines.append("支撐位：" + "、".join(f"{x:.2f}" for x in support_info["supports"]))
    else:
        support_lines.append("支撐位：N/A")
    if support_info["resistances"]:
        support_lines.append("壓力位：" + "、".join(f"{x:.2f}" for x in support_info["resistances"]))
    else:
        support_lines.append("壓力位：N/A")
    support_lines.append(
        "費波那契："
        + "｜".join(f"{ratio:.3f}={value:.2f}" for ratio, value in fib_levels.items())
    )
    support_lines.append(
        f"60 日高低區間：高 {support_info['recent_high']:.2f} / 低 {support_info['recent_low']:.2f}"
    )

    report = (
        "📊 完整技術分析報告\n\n"
        f"{format_basic_quote(quote)}\n\n"
        "🧮 技術指標\n"
        f"• RSI(14): {format_price(to_float(latest['RSI14']))}\n"
        f"• KD(9,9): K={format_price(to_float(latest['K9']))}｜D={format_price(to_float(latest['D9']))}\n"
        f"• MACD(12,26,9): MACD={format_price(to_float(latest['MACD']), 4)}｜Signal={format_price(to_float(latest['MACD_SIGNAL']), 4)}｜Hist={format_price(to_float(latest['MACD_HIST']), 4)}\n"
        f"• 均線: {ma_text}\n"
        f"• 布林通道: {boll_text}\n"
        f"• 成交量/5日均量: {format_price(analysis['vol_ratio']) if not pd.isna(analysis['vol_ratio']) else 'N/A'} 倍\n\n"
        "🚦 指標信號\n"
        + "\n".join(indicator_lines)
        + "\n\n📍 支撐與壓力\n"
        + "\n".join(f"• {line}" for line in support_lines)
        + "\n\n🧠 綜合評分\n"
        f"• 總評分：{analysis['rating']:.1f} / 10\n"
        f"• 明確建議：{analysis['recommendation']}\n"
        f"• 信心度：{analysis['confidence']}%\n\n"
        f"{DISCLAIMER}"
    )
    return report


def get_stock_info_sync(symbol: str) -> str:
    try:
        quote = get_basic_quote_sync(symbol)
        return format_basic_quote(quote)
    except StockDataError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("查詢股票 %s 時發生錯誤", symbol)
        return f"❌ 查詢股票 {symbol} 時發生錯誤：{exc}"


def analyze_stock_sync(symbol: str) -> str:
    try:
        analysis = compute_daily_analysis(symbol)
        return format_daily_analysis_report(analysis)
    except StockDataError as exc:
        return f"❌ {exc}"
    except Exception as exc:
        logger.exception("分析股票 %s 時發生錯誤", symbol)
        return f"❌ 分析股票 {symbol} 時發生錯誤：{exc}"


def _extract_symbol_frame(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        lvl0 = downloaded.columns.get_level_values(0)
        lvl1 = downloaded.columns.get_level_values(1)
        if symbol in lvl0:
            df = downloaded[symbol].copy()
            return strip_timezone(df).dropna(how="all")
        if symbol in lvl1:
            df = downloaded.xs(symbol, axis=1, level=1).copy()
            return strip_timezone(df).dropna(how="all")
        return pd.DataFrame()
    return strip_timezone(downloaded.copy()).dropna(how="all")


def fetch_history_quiet(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        return fetch_history(symbol, period=period, interval=interval)
    except Exception:
        return pd.DataFrame()


def get_daytrade_candidates_sync(limit: int = 10) -> str:
    symbols = list(dict.fromkeys(DAYTRADE_CANDIDATES.values()))
    downloaded = yf.download(
        tickers=" ".join(symbols),
        period="1mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if downloaded.empty:
        raise StockDataError("暫時無法取得候選股票資料。")

    rows: list[dict[str, Any]] = []
    for name, symbol in DAYTRADE_CANDIDATES.items():
        df = _extract_symbol_frame(downloaded, symbol)
        if df.empty or len(df) < 6:
            df = fetch_history_quiet(symbol, period="1mo", interval="1d")
        if df.empty or len(df) < 6:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            continue
        latest = df.iloc[-1]
        prev5 = df.iloc[:-1].tail(5)
        open_price = to_float(latest.get("Open"))
        high_price = to_float(latest.get("High"))
        low_price = to_float(latest.get("Low"))
        close_price = to_float(latest.get("Close"))
        volume = to_int(latest.get("Volume"))
        avg_vol = to_float(prev5["Volume"].mean())
        if open_price <= 0 or close_price <= 0 or volume <= 0:
            continue

        amplitude = (high_price - low_price) / open_price * 100
        vol_ratio = volume / avg_vol if avg_vol and not pd.isna(avg_vol) else np.nan
        if not (20 <= close_price <= 3000):
            continue
        if amplitude < 2.5:
            continue
        if not pd.isna(vol_ratio) and vol_ratio < 1.1:
            continue

        rows.append(
            {
                "name": name,
                "symbol": symbol,
                "price": close_price,
                "volume": volume,
                "amplitude": amplitude,
                "vol_ratio": vol_ratio,
            }
        )

    if not rows:
        return (
            "⚠️ 今日暫時沒有明確符合條件的當沖候選股。\n"
            "可能原因是市場波動不足、成交量不夠，或目前並非交易時段。\n\n"
            f"{DISCLAIMER}"
        )

    rows.sort(key=lambda item: (item["volume"], item["amplitude"], item["vol_ratio"]), reverse=True)
    volume_rank = {item["symbol"]: idx + 1 for idx, item in enumerate(rows)}

    scored_rows = []
    universe_size = max(len(rows), 1)
    for item in rows:
        rank_score = (universe_size - volume_rank[item["symbol"]] + 1) / universe_size * 4
        amp_score = min(item["amplitude"], 8.0) / 8.0 * 3
        vr = item["vol_ratio"] if not pd.isna(item["vol_ratio"]) else 1.0
        vol_score = min(vr, 3.0) / 3.0 * 3
        item["score"] = round(rank_score + amp_score + vol_score, 2)
        scored_rows.append(item)

    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    selected = scored_rows[:limit]

    lines = ["⚡ 今日適合關注的當沖候選股", ""]
    for idx, item in enumerate(selected, start=1):
        lines.append(
            f"{idx}. {item['name']} ({item['symbol']})｜現價 {item['price']:.2f}｜振幅 {item['amplitude']:.2f}%｜量比 {item['vol_ratio']:.2f}｜評分 {item['score']:.2f}"
        )
    lines.append("")
    lines.append("篩選邏輯：以候選池中當日成交量、日內振幅、價格區間與相對量能綜合排序。")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def resample_intraday(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(how="any")
    return out


def latest_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dates = pd.to_datetime(df.index).date
    latest_date = dates[-1]
    mask = dates == latest_date
    return df.loc[mask].copy()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_value = (typical * df["Volume"]).cumsum()
    cumulative_volume = df["Volume"].replace(0, np.nan).cumsum()
    return cumulative_value / cumulative_volume


def daytrade_stock_sync(symbol: str) -> str:
    quote = get_basic_quote_sync(symbol)
    intra = fetch_history(symbol, period="5d", interval="5m")
    intra = intra[["Open", "High", "Low", "Close", "Volume"]].dropna()
    intra = latest_session(intra)
    if intra.empty or len(intra) < 12:
        raise StockDataError(f"{symbol} 盤中資料不足，暫時無法提供當沖建議。")

    intra_5m = intra.copy()
    intra_15m = resample_intraday(intra_5m, "15min")
    if len(intra_15m) < 8:
        raise StockDataError(f"{symbol} 15 分鐘資料不足，暫時無法提供當沖建議。")

    intra_5m["RSI14"] = calculate_rsi(intra_5m["Close"], 14)
    intra_5m["MA5"] = intra_5m["Close"].rolling(5).mean()
    intra_5m["MA10"] = intra_5m["Close"].rolling(10).mean()
    intra_5m["VWAP"] = compute_vwap(intra_5m)

    intra_15m["RSI14"] = calculate_rsi(intra_15m["Close"], 14)
    intra_15m["MA5"] = intra_15m["Close"].rolling(5).mean()
    intra_15m["MA10"] = intra_15m["Close"].rolling(10).mean()

    now5 = intra_5m.iloc[-1]
    prev5 = intra_5m.iloc[-2]
    now15 = intra_15m.iloc[-1]

    session_open = to_float(intra_5m.iloc[0]["Open"])
    session_high = float(intra_5m["High"].max())
    session_low = float(intra_5m["Low"].min())
    session_close = to_float(now5["Close"])
    session_amp = (session_high - session_low) / session_open * 100 if session_open else 0.0

    recent_high = float(intra_5m.tail(6)["High"].max())
    recent_low = float(intra_5m.tail(6)["Low"].min())
    opening_range = intra_5m.head(min(6, len(intra_5m)))
    opening_high = float(opening_range["High"].max())
    opening_low = float(opening_range["Low"].min())

    recent_vol = float(intra_5m.tail(3)["Volume"].mean())
    baseline_vol = float(intra_5m.iloc[:-3].tail(12)["Volume"].mean()) if len(intra_5m) > 15 else float(intra_5m["Volume"].mean())
    vol_ratio = recent_vol / baseline_vol if baseline_vol else np.nan

    score = 0
    reasons: list[str] = []

    if session_close > to_float(now5["VWAP"]):
        score += 1
        reasons.append("現價位於 VWAP 之上")
    else:
        score -= 1
        reasons.append("現價跌破 VWAP")

    rsi5 = to_float(now5["RSI14"])
    rsi15 = to_float(now15["RSI14"])
    if rsi5 >= 55:
        score += 1
        reasons.append(f"5 分鐘 RSI {rsi5:.1f} 偏強")
    elif rsi5 <= 45:
        score -= 1
        reasons.append(f"5 分鐘 RSI {rsi5:.1f} 偏弱")

    if rsi15 >= 55:
        score += 1
        reasons.append(f"15 分鐘 RSI {rsi15:.1f} 偏強")
    elif rsi15 <= 45:
        score -= 1
        reasons.append(f"15 分鐘 RSI {rsi15:.1f} 偏弱")

    if to_float(now5["MA5"]) > to_float(now5["MA10"]):
        score += 1
        reasons.append("5 分鐘短均線多頭排列")
    else:
        score -= 1
        reasons.append("5 分鐘短均線偏空")

    if to_float(now15["MA5"]) > to_float(now15["MA10"]):
        score += 1
        reasons.append("15 分鐘短均線多頭排列")
    else:
        score -= 1
        reasons.append("15 分鐘短均線偏空")

    if not pd.isna(vol_ratio) and vol_ratio >= 1.25:
        score += 1
        reasons.append(f"近 3 根 5 分鐘量能放大至 {vol_ratio:.2f} 倍")
    elif not pd.isna(vol_ratio) and vol_ratio <= 0.8:
        score -= 1
        reasons.append(f"近 3 根 5 分鐘量能僅 {vol_ratio:.2f} 倍，追價力道不足")

    if score >= 4:
        bias = "🟢 偏多當沖"
        entry = max(to_float(now5["VWAP"]), to_float(now5["MA10"]), recent_low)
        breakout = recent_high
        stop_loss = min(recent_low, opening_low)
        risk = max(entry - stop_loss, session_close * 0.003)
        take_profit = max(session_high, entry + risk * 1.5)
        action_text = (
            f"建議偏多操作，可留意 {entry:.2f} 附近承接或突破 {breakout:.2f} 後順勢進場。"
        )
    elif score <= -4:
        bias = "🔴 偏空當沖"
        entry = min(to_float(now5["VWAP"]), to_float(now5["MA10"]), recent_high)
        breakout = recent_low
        stop_loss = max(recent_high, opening_high)
        risk = max(stop_loss - entry, session_close * 0.003)
        take_profit = min(session_low, entry - risk * 1.5)
        action_text = (
            f"建議保守或偏空看待，可觀察跌破 {breakout:.2f} 後的弱勢延續；若不可做空，宜以觀望為主。"
        )
    else:
        bias = "🟡 區間震盪"
        entry = max(session_low, to_float(now5["VWAP"]))
        breakout = recent_high
        stop_loss = min(recent_low, opening_low)
        risk = max(entry - stop_loss, session_close * 0.0025)
        take_profit = min(recent_high, entry + risk)
        action_text = "方向不明顯，較適合等待突破開盤區間高低點後再跟進。"

    if session_amp >= 6 or (not pd.isna(vol_ratio) and vol_ratio >= 2):
        risk_level = "高"
    elif session_amp >= 3:
        risk_level = "中"
    else:
        risk_level = "低"

    report = (
        "⚡ 當沖即時建議\n\n"
        f"{format_basic_quote(quote)}\n\n"
        "⏱️ 盤中短線訊號\n"
        f"• 5 分鐘 RSI: {format_price(rsi5)}｜MA5 {format_price(to_float(now5['MA5']))}｜MA10 {format_price(to_float(now5['MA10']))}\n"
        f"• 15 分鐘 RSI: {format_price(rsi15)}｜MA5 {format_price(to_float(now15['MA5']))}｜MA10 {format_price(to_float(now15['MA10']))}\n"
        f"• VWAP: {format_price(to_float(now5['VWAP']))}\n"
        f"• 近 3 根 / 基準量能: {format_price(vol_ratio) if not pd.isna(vol_ratio) else 'N/A'} 倍\n"
        f"• 今日區間: 高 {session_high:.2f} / 低 {session_low:.2f} / 開盤區間高低 {opening_high:.2f}-{opening_low:.2f}\n\n"
        "🧭 即時判讀\n"
        f"• 方向判斷：{bias}\n"
        + "\n".join(f"• {reason}" for reason in reasons)
        + "\n\n🎯 交易建議\n"
        f"• {action_text}\n"
        f"• 參考進場點：{entry:.2f}\n"
        f"• 參考出場點：{take_profit:.2f}\n"
        f"• 參考停損點：{stop_loss:.2f}\n"
        f"• 風險等級：{risk_level}\n\n"
        f"{DISCLAIMER}"
    )
    return report


async def get_stock_info(stock_id: str) -> str:
    return await asyncio.to_thread(get_stock_info_sync, stock_id)


async def analyze_stock(stock_id: str) -> str:
    return await asyncio.to_thread(analyze_stock_sync, stock_id)


async def get_daytrade_candidates() -> str:
    try:
        return await asyncio.to_thread(get_daytrade_candidates_sync)
    except StockDataError as exc:
        return f"❌ {exc}"
    except Exception as exc:
        logger.exception("取得當沖候選清單時發生錯誤")
        return f"❌ 取得當沖候選清單時發生錯誤：{exc}"


async def analyze_daytrade_stock(stock_id: str) -> str:
    try:
        return await asyncio.to_thread(daytrade_stock_sync, stock_id)
    except StockDataError as exc:
        return f"❌ {exc}"
    except Exception as exc:
        logger.exception("分析當沖個股 %s 時發生錯誤", stock_id)
        return f"❌ 分析當沖個股 {stock_id} 時發生錯誤：{exc}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"👋 嗨 {user.first_name}！\n\n"
        f"我是台灣股票查詢機器人 🤖\n\n"
        f"指令：\n"
        f"/stock 2330 → 查詢單支股票\n"
        f"/analyze 2330 或 /a 2330 → 完整技術分析與買賣建議\n"
        f"/dt 或 /daytrade → 列出今日適合關注的當沖股票\n"
        f"/dt 2330 → 取得個股當沖建議\n"
        f"/stocks 2330 2317 2454 → 查詢多支\n"
        f"/list → 熱門股票列表\n"
        f"/help → 幫助"
    )


async def stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("請提供股票代碼。例如: /stock 2330")
        return
    stock_id = normalize_stock_symbol(context.args[0])
    await update.message.reply_text("⏳ 查詢中...")
    message = await get_stock_info(stock_id)
    await update.message.reply_text(message)


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("請提供股票代碼。例如: /analyze 2330 或 /a 2330")
        return
    stock_id = normalize_stock_symbol(context.args[0])
    await update.message.reply_text("⏳ 正在計算技術分析與買賣建議...")
    message = await analyze_stock(stock_id)
    await update.message.reply_text(message)


async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "請提供股票代碼，用空格或逗號分隔。\n例如: /stocks 2330 2317 2454\n或: /stocks 2330,2317,2454"
        )
        return
    raw = " ".join(context.args)
    stock_ids = [normalize_stock_symbol(s.strip()) for part in raw.split(",") for s in part.split() if s.strip()]
    await update.message.reply_text(f"⏳ 查詢 {len(stock_ids)} 支股票中...")
    responses = []
    for stock_id in stock_ids:
        responses.append(await get_stock_info(stock_id))
    await update.message.reply_text("\n\n".join(responses))


async def daytrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("⏳ 正在篩選今日適合關注的當沖股票...")
        message = await get_daytrade_candidates()
        await update.message.reply_text(message)
        return

    stock_id = normalize_stock_symbol(context.args[0])
    await update.message.reply_text("⏳ 正在分析個股當沖建議...")
    message = await analyze_daytrade_stock(stock_id)
    await update.message.reply_text(message)


async def list_hot_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = "🔥 熱門台灣股票列表\n\n"
    for name, stock_id in hot_stocks.items():
        message += f"• {name} ({stock_id})\n"
    message += "\n使用 /stock <代碼> 查詢基本資訊，或用 /analyze <代碼> 查看完整技術分析。"
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📋 指令列表：\n\n"
        "/start → 啟動機器人\n"
        "/stock 2330 → 查詢單支股票基本報價\n"
        "/analyze 2330 → 完整技術分析與買賣建議\n"
        "/a 2330 → /analyze 的簡寫\n"
        "/dt → 列出今日適合關注的當沖股票\n"
        "/daytrade → /dt 的完整指令\n"
        "/dt 2330 → 取得個股當沖建議\n"
        "/stocks 2330 2317 2454 → 查詢多支股票\n"
        "/list → 熱門股票列表\n"
        "/help → 顯示此幫助\n\n"
        f"{DISCLAIMER}"
    )


def main() -> None:
    print("🤖 台灣股票查詢 Bot 啟動中...")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stock", stock))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("a", analyze))
    application.add_handler(CommandHandler("stocks", stocks))
    application.add_handler(CommandHandler("dt", daytrade))
    application.add_handler(CommandHandler("daytrade", daytrade))
    application.add_handler(CommandHandler("list", list_hot_stocks))
    application.add_handler(CommandHandler("help", help_command))
    print("✅ Bot 已啟動！持續監聽中... (Ctrl+C 停止)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
