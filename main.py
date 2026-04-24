from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.ERROR)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

WATCHLIST: dict[str, str] = {
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
    "欣興": "3037.TW",
    "陽明": "2609.TW",
    "長榮": "2603.TW",
    "0050": "0050.TW",
    "0056": "0056.TW",
    "006208": "006208.TW",
}

DISCLAIMER = "⚠️ 本內容僅供研究與教學參考，不構成投資建議；實際交易請自行控管風險。"


@dataclass
class Quote:
    symbol: str
    name: str
    date: str
    close: float
    previous_close: float
    open: float
    high: float
    low: float
    volume: int

    @property
    def change(self) -> float:
        return self.close - self.previous_close

    @property
    def change_percent(self) -> float:
        if self.previous_close == 0:
            return 0.0
        return self.change / self.previous_close * 100


def normalize_symbol(raw: str) -> str:
    value = raw.strip().upper()
    if not value:
        return value
    if value in WATCHLIST:
        return WATCHLIST[value]
    if value.endswith(".TW") or value.endswith(".TWO"):
        return value
    if value.isdigit():
        return f"{value}.TWO" if value.startswith("7") else f"{value}.TW"
    return value


def format_num(value: Optional[float], digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fetch_history(symbol: str, period: str = "450d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d", auto_adjust=False, actions=False)
    if df.empty:
        raise ValueError(f"查無 {symbol} 股價資料")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")


def fetch_quote(symbol: str) -> Quote:
    df = fetch_history(symbol, period="15d")
    if len(df) < 2:
        raise ValueError(f"{symbol} 資料不足")

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    ticker = yf.Ticker(symbol)
    name = symbol
    try:
        info = ticker.fast_info
        name = str(getattr(info, "short_name", None) or symbol)
    except Exception:
        name = symbol

    return Quote(
        symbol=symbol,
        name=name,
        date=df.index[-1].strftime("%Y-%m-%d"),
        close=float(latest["Close"]),
        previous_close=float(previous["Close"]),
        open=float(latest["Open"]),
        high=float(latest["High"]),
        low=float(latest["Low"]),
        volume=int(latest["Volume"]),
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    return value.fillna(50)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def analyze(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    quote = fetch_quote(symbol)
    df = fetch_history(symbol)
    close = df["Close"]
    volume = df["Volume"]

    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["RSI14"] = rsi(close)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(close)
    df["VOL_MA5"] = volume.shift(1).rolling(5).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    notes: list[str] = []

    if latest["Close"] > latest["MA20"] > latest["MA60"]:
        score += 2
        notes.append("股價站上 MA20，且 MA20 高於 MA60，趨勢偏多")
    elif latest["Close"] < latest["MA20"] < latest["MA60"]:
        score -= 2
        notes.append("股價跌破 MA20，且 MA20 低於 MA60，趨勢偏弱")
    else:
        notes.append("均線結構仍偏整理")

    if latest["RSI14"] < 35:
        score += 1
        notes.append(f"RSI {latest['RSI14']:.1f}，短線偏低，留意反彈")
    elif latest["RSI14"] > 70:
        score -= 1
        notes.append(f"RSI {latest['RSI14']:.1f}，短線偏熱，追價風險提高")
    else:
        notes.append(f"RSI {latest['RSI14']:.1f}，屬中性區")

    if latest["MACD"] > latest["MACD_SIGNAL"] and prev["MACD"] <= prev["MACD_SIGNAL"]:
        score += 2
        notes.append("MACD 黃金交叉")
    elif latest["MACD"] < latest["MACD_SIGNAL"] and prev["MACD"] >= prev["MACD_SIGNAL"]:
        score -= 2
        notes.append("MACD 死亡交叉")
    else:
        notes.append("MACD 尚未出現明確轉折")

    vol_ratio = latest["Volume"] / latest["VOL_MA5"] if latest["VOL_MA5"] and not pd.isna(latest["VOL_MA5"]) else np.nan
    if not pd.isna(vol_ratio) and vol_ratio >= 1.5 and quote.change > 0:
        score += 1
        notes.append(f"量能約為 5 日均量 {vol_ratio:.2f} 倍，價量偏多")
    elif not pd.isna(vol_ratio) and vol_ratio >= 1.5 and quote.change < 0:
        score -= 1
        notes.append(f"量能約為 5 日均量 {vol_ratio:.2f} 倍，但價格下跌，需防出貨")
    else:
        notes.append("量能未明顯放大")

    if score >= 3:
        action = "🟢 偏多觀察"
    elif score <= -3:
        action = "🔴 保守避開"
    else:
        action = "🟡 觀望"

    change_mark = "▲" if quote.change > 0 else "▼" if quote.change < 0 else "—"
    lines = [
        f"📊 {quote.name} ({quote.symbol})",
        f"日期：{quote.date}",
        f"收盤：{quote.close:.2f}",
        f"漲跌：{change_mark} {abs(quote.change):.2f} ({quote.change_percent:+.2f}%)",
        f"開高低：{quote.open:.2f} / {quote.high:.2f} / {quote.low:.2f}",
        f"成交量：{quote.volume:,}",
        "",
        "🧮 指標",
        f"MA5：{format_num(latest['MA5'])}｜MA20：{format_num(latest['MA20'])}｜MA60：{format_num(latest['MA60'])}",
        f"RSI14：{format_num(latest['RSI14'])}",
        f"MACD Hist：{format_num(latest['MACD_HIST'], 4)}",
        "",
        "🧠 判斷",
        *[f"• {note}" for note in notes],
        "",
        f"結論：{action}",
        DISCLAIMER,
    ]
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "台股查詢機器人已啟動。\n"
        "用法：\n"
        "/stock 2330\n"
        "/analyze 2330\n"
        "/list"
    )


async def stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("請輸入股票代號，例如：/stock 2330")
        return
    try:
        quote = fetch_quote(normalize_symbol(context.args[0]))
        mark = "▲" if quote.change > 0 else "▼" if quote.change < 0 else "—"
        await update.message.reply_text(
            f"{quote.name} ({quote.symbol})\n"
            f"日期：{quote.date}\n"
            f"收盤：{quote.close:.2f}\n"
            f"漲跌：{mark} {abs(quote.change):.2f} ({quote.change_percent:+.2f}%)\n"
            f"成交量：{quote.volume:,}\n\n{DISCLAIMER}"
        )
    except Exception as exc:
        logger.exception("stock command failed")
        await update.message.reply_text(f"查詢失敗：{exc}")


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("請輸入股票代號，例如：/analyze 2330")
        return
    try:
        await update.message.reply_text(analyze(context.args[0]))
    except Exception as exc:
        logger.exception("analyze command failed")
        await update.message.reply_text(f"分析失敗：{exc}")


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "\n".join(f"{name}：{symbol}" for name, symbol in WATCHLIST.items())
    await update.message.reply_text("常用清單：\n" + text)


def main() -> None:
    if not TOKEN:
        raise RuntimeError("缺少 TELEGRAM_BOT_TOKEN。請在環境變數或 .env 設定。")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stock", stock))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("list", list_cmd))

    logger.info("Taiwan stock bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
