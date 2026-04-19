# 在 Colab 單一儲存格貼上並執行本檔全部內容即可

!pip -q install yfinance openpyxl requests

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.colab import files
from IPython.display import display

# =========================
# 0. 設定區
# =========================
SEND_TELEGRAM = True
TELEGRAM_BOT_TOKEN = "8584415843:AAFDvgxqetm_90_M_ZVMm5MB36NCLllTQZM"
TELEGRAM_CHAT_ID = "6744076686"
STOPLOSS_WARNING_THRESHOLD = 3.0

# =========================
# 1. 上傳 Excel（上傳後立刻讀，不再依賴 /content 持久化）
# =========================
uploaded = files.upload()
if not uploaded:
    raise FileNotFoundError("沒有上傳任何檔案，請重新執行此儲存格。")

file_name = list(uploaded.keys())[0]
file_path = f"/content/{file_name}"
output_path = f"/content/{Path(file_name).stem}_live.xlsx"

print("使用檔案：", file_name)

# =========================
# 2. 工具函式
# =========================
def clean_code(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.upper()


def to_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_dates(series):
    result = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(result.dt, "tz", None) is not None:
            result = result.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    except Exception:
        pass
    return result


def strip_timezone(df):
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    return df


def candidate_tickers(code):
    code = clean_code(code)
    if not code:
        return []
    if code.startswith("^") or code.endswith(".TW") or code.endswith(".TWO"):
        return [code]
    if code.isdigit():
        if code.startswith("7"):
            return [f"{code}.TWO", f"{code}.TW"]
        return [f"{code}.TW", f"{code}.TWO"]
    return [code]


def try_fetch_one(code):
    for ticker in candidate_tickers(code):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="10d", interval="1d", auto_adjust=False, actions=False)
            hist = strip_timezone(hist)
            if hist is None or hist.empty:
                continue
            hist = hist.dropna(subset=["Close"])
            if hist.empty:
                continue
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else np.nan
            chg = last_close - prev_close if pd.notna(prev_close) else np.nan
            chg_pct = (chg / prev_close * 100) if pd.notna(prev_close) and prev_close != 0 else np.nan
            return {
                "ticker": ticker,
                "price": round(last_close, 4),
                "prev_close": round(prev_close, 4) if pd.notna(prev_close) else np.nan,
                "change": round(chg, 4) if pd.notna(chg) else np.nan,
                "change_pct": round(chg_pct, 4) if pd.notna(chg_pct) else np.nan,
            }
        except Exception:
            continue
    return {
        "ticker": None,
        "price": np.nan,
        "prev_close": np.nan,
        "change": np.nan,
        "change_pct": np.nan,
    }


def calculate_bias(close, window):
    ma = close.rolling(window).mean()
    return ((close / ma) - 1) * 100


def rolling_linear_regression_slope(series, window=5):
    x = np.arange(window, dtype=float)
    def _slope(values):
        if np.isnan(values).any():
            return np.nan
        return float(np.polyfit(x, values, 1)[0])
    return series.rolling(window).apply(_slope, raw=True)


def prepare_signal_features(price_df, symbol=None):
    working = price_df.copy()
    if "date" in working.columns:
        working["date"] = normalize_dates(working["date"]).dt.normalize()
    else:
        working = working.reset_index().rename(columns={working.index.name or "index": "date"})
        working["date"] = normalize_dates(working["date"]).dt.normalize()
    working = working.sort_values("date").reset_index(drop=True)
    working["close"] = working["Close"].astype(float)
    working["bias_55"] = calculate_bias(working["close"], 55)
    working["bias_144"] = calculate_bias(working["close"], 144)
    working["bias_233"] = calculate_bias(working["close"], 233)
    working["bias_55_slope"] = rolling_linear_regression_slope(working["bias_55"], 5)
    working["dist_to_zero"] = working["bias_55"].abs()
    rolling_20_high = working["High"].shift(1).rolling(20).max()
    working["breakout_pct"] = ((working["close"] / rolling_20_high) - 1) * 100
    working["volume_value_mil"] = (working["Volume"].astype(float) * working["close"]) / 1_000_000
    working["symbol"] = symbol if symbol is not None else "UNKNOWN"
    needed = [
        "date", "symbol", "close", "bias_55", "bias_144", "bias_233",
        "bias_55_slope", "dist_to_zero", "breakout_pct", "volume_value_mil"
    ]
    return working[needed].copy()


def score_to_signal(score):
    if score >= 80:
        return "buy"
    if score >= 40:
        return "watch"
    return "avoid"


def evaluate_stock_signal(row, market_bias_latest=None):
    raw = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    bias_55 = to_float(raw.get("bias_55"), 0.0)
    bias_144 = to_float(raw.get("bias_144"), 0.0)
    bias_233 = to_float(raw.get("bias_233"), 0.0)
    bias_55_slope = to_float(raw.get("bias_55_slope"), 0.0)
    breakout_pct = to_float(raw.get("breakout_pct"), 0.0)
    volume_value_mil = to_float(raw.get("volume_value_mil"), 0.0)

    breakdown = {
        "hard_filter": 0,
        "core_wave": 0,
        "zero_cross": 0,
        "market_alignment": 0,
        "structure": 0,
        "breakout": 0,
        "liquidity": 0,
    }

    if volume_value_mil < 0.8:
        breakdown["hard_filter"] = -100
        return {
            "score": -100,
            "score_breakdown": breakdown,
            "signal": score_to_signal(-100),
        }

    if bias_55 < 0 and bias_55_slope > 0:
        breakdown["core_wave"] = 35
    if -5 < bias_55 < 8:
        breakdown["zero_cross"] = 20
    if market_bias_latest and "bias_55" in market_bias_latest:
        if to_float(market_bias_latest.get("bias_55"), -999) >= -5:
            breakdown["market_alignment"] = 15

    structure_score = 0
    if bias_144 > -10:
        structure_score += 8
    if bias_233 > -15:
        structure_score += 6
    breakdown["structure"] = structure_score

    if breakout_pct >= 1.5:
        breakdown["breakout"] = 4
    if volume_value_mil >= 3:
        breakdown["liquidity"] = 10

    score = int(sum(breakdown.values()))
    return {
        "score": score,
        "score_breakdown": breakdown,
        "signal": score_to_signal(score),
    }


def latest_signal_snapshot(code):
    for ticker in candidate_tickers(code):
        try:
            hist = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False, actions=False)
            hist = strip_timezone(hist).dropna(how="all")
            if hist.empty or len(hist) < 250:
                continue
            hist = hist.tail(250)
            features = prepare_signal_features(hist[["Open", "High", "Low", "Close", "Volume"]].copy(), symbol=code)
            features = features.dropna(subset=[
                "bias_55", "bias_144", "bias_233", "bias_55_slope",
                "dist_to_zero", "breakout_pct", "volume_value_mil"
            ]).copy()
            if features.empty:
                continue
            row = features.iloc[-1].to_dict()
            row["ticker"] = ticker
            return row
        except Exception:
            continue
    return None


def format_money(value):
    num = to_float(value)
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}"


def format_pct(value, assume_ratio=False):
    num = to_float(value)
    if pd.isna(num):
        return "N/A"
    if assume_ratio:
        num *= 100
    return f"{num:.2f}%"


def send_telegram_message(text):
    if not SEND_TELEGRAM:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok = True
    for chunk in [text[i:i+3500] for i in range(0, len(text), 3500)]:
        try:
            resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=30)
            if resp.status_code >= 400:
                ok = False
                print("Telegram 發送失敗：", resp.text)
        except Exception as e:
            ok = False
            print("Telegram 發送例外：", e)
    return ok

# =========================
# 3. 讀取 Excel
# =========================
xls = pd.ExcelFile(file_path)
print("工作表：", xls.sheet_names)

sheet_dict = {}
for sheet in xls.sheet_names:
    sheet_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)

control = sheet_dict.get("Stock_Control", pd.DataFrame()).copy()
trades = sheet_dict.get("Stock_Trades", pd.DataFrame()).copy()
signals = sheet_dict.get("Stock_Signals", pd.DataFrame()).copy()

for df in [control, trades, signals]:
    df.columns = [str(c).strip() for c in df.columns]
    if "股票代號" in df.columns:
        df["股票代號"] = df["股票代號"].apply(clean_code)

for c in ["收盤價", "停損價", "規則分數", "AI分數"]:
    if c in control.columns:
        control[c] = pd.to_numeric(control[c], errors="coerce")

for c in ["買入價", "張數", "停損價", "賣出價", "現價", "損益(元)", "損益率"]:
    if c in trades.columns:
        trades[c] = pd.to_numeric(trades[c], errors="coerce")

if "最後檢查日" in control.columns:
    control["最後檢查日"] = pd.to_datetime(control["最後檢查日"], errors="coerce")

for c in ["買入日", "賣出日"]:
    if c in trades.columns:
        trades[c] = pd.to_datetime(trades[c], errors="coerce")

# =========================
# 4. 收集股票代號與抓最新價格
# =========================
codes = set()
if "股票代號" in control.columns:
    codes.update(control["股票代號"].dropna().astype(str).tolist())
if "股票代號" in trades.columns:
    codes.update(trades["股票代號"].dropna().astype(str).tolist())
codes = sorted([c.strip() for c in codes if c.strip() and c.strip().lower() != "nan"])

price_map = {}
signal_map = {}
now_ts = pd.Timestamp.now()

market_snapshot = latest_signal_snapshot("^TWII")
market_bias_latest = None
if market_snapshot is not None:
    market_bias_latest = {
        "symbol": "^TWII",
        "bias_55": to_float(market_snapshot.get("bias_55")),
    }

for code in codes:
    price_map[code] = try_fetch_one(code)
    snap = latest_signal_snapshot(code)
    if snap is not None:
        scored = evaluate_stock_signal(snap, market_bias_latest=market_bias_latest)
        signal_map[code] = {
            "snapshot": snap,
            "score": scored["score"],
            "signal": scored["signal"],
            "score_breakdown": json.dumps(scored["score_breakdown"], ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        }
    print(code, price_map[code], signal_map.get(code, {}).get("score"))
    time.sleep(0.3)

price_df = pd.DataFrame([
    {
        "股票代號": code,
        "Yahoo代碼": data["ticker"],
        "最新價": data["price"],
        "昨收": data["prev_close"],
        "漲跌": data["change"],
        "漲跌幅(%)": data["change_pct"],
    }
    for code, data in price_map.items()
])

# =========================
# 5. 更新 Stock_Control
# =========================
if not control.empty and "股票代號" in control.columns:
    control = control.merge(
        price_df[["股票代號", "Yahoo代碼", "最新價", "昨收", "漲跌", "漲跌幅(%)"]],
        on="股票代號",
        how="left"
    )

    if "原始收盤價" not in control.columns and "收盤價" in control.columns:
        control["原始收盤價"] = control["收盤價"]
    if "追蹤中" not in control.columns:
        control["追蹤中"] = "YES"

    control["追蹤中"] = control["追蹤中"].fillna("YES").astype(str).str.upper().str.strip()
    control["資料日期"] = now_ts
    control["market_bias_latest"] = to_float(market_bias_latest.get("bias_55")) if market_bias_latest else np.nan

    control["規則分數"] = control["股票代號"].map(lambda x: signal_map.get(x, {}).get("score", np.nan))
    control["信號等級"] = control["股票代號"].map(lambda x: signal_map.get(x, {}).get("signal", ""))
    control["score_breakdown"] = control["股票代號"].map(lambda x: signal_map.get(x, {}).get("score_breakdown", ""))

    tracking_mask = control["追蹤中"].eq("YES")
    valid_mask = tracking_mask & control["最新價"].notna() & control["停損價"].notna() & (control["停損價"] != 0)

    control["是否跌破停損"] = np.nan
    control["距停損(%)"] = np.nan
    control.loc[valid_mask, "是否跌破停損"] = np.where(
        control.loc[valid_mask, "最新價"] < control.loc[valid_mask, "停損價"],
        "是",
        "否"
    )
    control.loc[valid_mask, "距停損(%)"] = (
        (control.loc[valid_mask, "最新價"] / control.loc[valid_mask, "停損價"] - 1) * 100
    )

# =========================
# 6. 更新 Stock_Trades
# =========================
if not trades.empty and "股票代號" in trades.columns:
    trades = trades.merge(
        price_df[["股票代號", "Yahoo代碼", "最新價", "昨收", "漲跌", "漲跌幅(%)"]],
        on="股票代號",
        how="left"
    )

    if "原始現價" not in trades.columns and "現價" in trades.columns:
        trades["原始現價"] = trades["現價"]

    if "賣出日" in trades.columns:
        open_mask = trades["賣出日"].isna()
    else:
        open_mask = pd.Series([True] * len(trades), index=trades.index)

    if "狀態" in trades.columns:
        open_mask = open_mask & (~trades["狀態"].astype(str).str.contains("已結束", na=False))

    trades.loc[open_mask, "現價"] = trades.loc[open_mask, "最新價"].combine_first(trades.loc[open_mask, "現價"])

    if {"買入價", "張數", "現價"}.issubset(trades.columns):
        trades.loc[open_mask, "損益(元)"] = (
            (trades.loc[open_mask, "現價"] - trades.loc[open_mask, "買入價"]) * trades.loc[open_mask, "張數"] * 1000
        )
        trades.loc[open_mask, "損益率"] = np.where(
            trades.loc[open_mask, "買入價"].notna() & (trades.loc[open_mask, "買入價"] != 0),
            (trades.loc[open_mask, "現價"] / trades.loc[open_mask, "買入價"] - 1),
            np.nan
        )

    trades["資料日期"] = now_ts
    if "停損價" in trades.columns:
        valid_trade_mask = open_mask & trades["現價"].notna() & trades["停損價"].notna() & (trades["停損價"] != 0)
        trades["是否跌破停損"] = np.nan
        trades["距停損(%)"] = np.nan
        trades.loc[valid_trade_mask, "是否跌破停損"] = np.where(
            trades.loc[valid_trade_mask, "現價"] < trades.loc[valid_trade_mask, "停損價"],
            "是",
            "否"
        )
        trades.loc[valid_trade_mask, "距停損(%)"] = (
            (trades.loc[valid_trade_mask, "現價"] / trades.loc[valid_trade_mask, "停損價"] - 1) * 100
        )

# =========================
# 7. 重建 Stock_Signals
# =========================
signal_rows = []
for code in codes:
    payload = signal_map.get(code)
    if not payload:
        continue
    snap = payload["snapshot"]
    signal_rows.append({
        "股票代號": code,
        "Yahoo代碼": snap.get("ticker"),
        "日期": snap.get("date"),
        "close": snap.get("close"),
        "bias_55": snap.get("bias_55"),
        "bias_144": snap.get("bias_144"),
        "bias_233": snap.get("bias_233"),
        "bias_55_slope": snap.get("bias_55_slope"),
        "dist_to_zero": snap.get("dist_to_zero"),
        "breakout_pct": snap.get("breakout_pct"),
        "volume_value_mil": snap.get("volume_value_mil"),
        "規則分數": payload.get("score"),
        "信號等級": payload.get("signal"),
        "score_breakdown": payload.get("score_breakdown"),
        "market_bias_latest": to_float(market_bias_latest.get("bias_55")) if market_bias_latest else np.nan,
    })

signals_new = pd.DataFrame(signal_rows).sort_values(["規則分數", "股票代號"], ascending=[False, True]) if signal_rows else pd.DataFrame()

# =========================
# 8. 新增 Live_Prices
# =========================
price_check = price_df.copy()
price_check["抓取時間"] = now_ts

# =========================
# 9. 輸出到新 Excel
# =========================
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for sheet, df in sheet_dict.items():
        if sheet == "Stock_Control":
            control.to_excel(writer, sheet_name=sheet, index=False)
        elif sheet == "Stock_Trades":
            trades.to_excel(writer, sheet_name=sheet, index=False)
        elif sheet == "Stock_Signals":
            signals_new.to_excel(writer, sheet_name=sheet, index=False)
        else:
            df.to_excel(writer, sheet_name=sheet, index=False)
    if "Stock_Control" not in sheet_dict:
        control.to_excel(writer, sheet_name="Stock_Control", index=False)
    if "Stock_Trades" not in sheet_dict:
        trades.to_excel(writer, sheet_name="Stock_Trades", index=False)
    if "Stock_Signals" not in sheet_dict:
        signals_new.to_excel(writer, sheet_name="Stock_Signals", index=False)
    price_check.to_excel(writer, sheet_name="Live_Prices", index=False)

# =========================
# 10. 顯示摘要
# =========================
tracked = control[control["追蹤中"].eq("YES")].copy() if (not control.empty and "追蹤中" in control.columns) else pd.DataFrame()
open_trades = trades.copy()
if not trades.empty and "賣出日" in trades.columns:
    open_trades = open_trades[open_trades["賣出日"].isna()].copy()
if not trades.empty and "狀態" in trades.columns:
    open_trades = open_trades[~open_trades["狀態"].astype(str).str.contains("已結束", na=False)].copy()

summary_lines = []
summary_lines.append("股票決策控制系統更新摘要")
summary_lines.append(f"來源檔案：{file_path}")
summary_lines.append(f"輸出檔案：{output_path}")
summary_lines.append("")
if market_bias_latest:
    summary_lines.append(f"大盤訊號：^TWII bias_55 = {format_pct(market_bias_latest['bias_55'])}")
    summary_lines.append("")

summary_lines.append("持股現況：")
if tracked.empty:
    summary_lines.append("- 沒有追蹤中的標的")
else:
    for _, row in tracked.iterrows():
        summary_lines.append(
            f"- {row.get('股票代號', '')} {row.get('股票名稱', '') if '股票名稱' in tracked.columns else ''}｜現價 {row.get('最新價', 'N/A')}｜停損 {row.get('停損價', 'N/A')}｜距停損 {format_pct(row.get('距停損(%)'))}｜分數 {row.get('規則分數', 'N/A')}｜{row.get('信號等級', '')}"
        )
summary_lines.append("")

summary_lines.append("未結束交易損益：")
if open_trades.empty:
    summary_lines.append("- 沒有未結束交易")
else:
    total_pnl = open_trades["損益(元)"].fillna(0).sum() if "損益(元)" in open_trades.columns else 0
    summary_lines.append(f"- 總筆數：{len(open_trades)}｜總損益：{format_money(total_pnl)} 元")
    for _, row in open_trades.iterrows():
        summary_lines.append(
            f"- {row.get('股票代號', '')} {row.get('股票名稱', '') if '股票名稱' in open_trades.columns else ''}｜損益 {format_money(row.get('損益(元)'))} 元｜報酬 {format_pct(row.get('損益率'), assume_ratio=True)}"
        )
summary_lines.append("")

summary_lines.append("停損警報：")
if tracked.empty or "是否跌破停損" not in tracked.columns:
    summary_lines.append("- 無可檢查資料")
else:
    breached = tracked[tracked["是否跌破停損"] == "是"]
    near_stop = tracked[(tracked["是否跌破停損"] != "是") & (tracked["距停損(%)"].notna()) & (tracked["距停損(%)"] <= STOPLOSS_WARNING_THRESHOLD)]
    if breached.empty and near_stop.empty:
        summary_lines.append("- 目前沒有跌破或接近停損的追蹤標的")
    else:
        for _, row in breached.iterrows():
            summary_lines.append(f"- 警報：{row['股票代號']} 已跌破停損，距停損 {format_pct(row.get('距停損(%)'))}")
        for _, row in near_stop.iterrows():
            summary_lines.append(f"- 接近：{row['股票代號']} 距停損僅 {format_pct(row.get('距停損(%)'))}")
summary_lines.append("")

summary_lines.append("信號分數排名：")
if signals_new.empty:
    summary_lines.append("- 沒有可用信號資料")
else:
    for i, (_, row) in enumerate(signals_new.head(10).iterrows(), start=1):
        summary_lines.append(
            f"- {i}. {row['股票代號']}｜分數 {row['規則分數']}｜{row['信號等級']}｜bias55 {format_pct(row.get('bias_55'))}｜突破 {format_pct(row.get('breakout_pct'))}｜量額 {row.get('volume_value_mil', 'N/A')}"
        )

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# =========================
# 11. 顯示結果與下載
# =========================
print("\nStock_Control 預覽：")
display(control.head())

print("\nStock_Trades 預覽：")
display(trades.head())

print("\nStock_Signals 預覽：")
display(signals_new.head())

print("\n已輸出：", output_path)
files.download(output_path)

telegram_ok = send_telegram_message(summary_text)
print("Telegram 推播結果：", "成功" if telegram_ok else "未送出或失敗")
