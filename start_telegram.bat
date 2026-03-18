@echo off
echo Starting Telegram Signal Scraper (FTMO)...
cd /d C:\tradebots
call .venv\Scripts\activate.bat
set "MT5_MODULE=MetaTrader5_FTMO"
set "PYTHONUNBUFFERED=1"
python -u common\tools\telegram_signals.py
pause
