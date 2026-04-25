# taiwan-stock-bot

台股 Telegram 查詢與簡易技術分析機器人。

## 功能

- `/start`：顯示使用方式
- `/stock 2330`：查詢收盤、漲跌、成交量
- `/analyze 2330`：輸出 MA、RSI、MACD 與簡易多空判斷
- `/list`：顯示常用觀察清單

## 安全重點

不要把 Telegram Bot Token 寫進程式碼，也不要提交 `.env` 到 GitHub。

Token 請放在環境變數：

```bash
TELEGRAM_BOT_TOKEN=你的新token
```

## 本機執行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python main.py
```

## Codespaces 執行

1. 到 GitHub repo 按 `Code`
2. 選 `Codespaces`
3. 建立 codespace
4. 在 terminal 執行：

```bash
pip install -r requirements.txt
python main.py
```

如果用 Codespaces Secrets，請設定：

```text
TELEGRAM_BOT_TOKEN
```

## 免責聲明

本專案僅供研究與教學參考，不構成投資建議。
