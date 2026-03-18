@echo off
set "ENABLE_LIVE_TRADING=1"
set "PYTHONUNBUFFERED=1"
set "MT5_MODULE=MetaTrader5_FTMO"
C:\tradebots\.venv\Scripts\python.exe -u C:\tradebots\common\live\run_bot.py --live --account-id ftmo_100k