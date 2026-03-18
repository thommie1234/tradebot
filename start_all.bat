@echo off
echo ============================================
echo   SOVEREIGN TRADING SYSTEM - Starting All
echo ============================================

echo.
echo [1/7] Starting BrightFunded MT5 Terminal...
start "" "C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe" /portable

echo [2/7] Starting FTMO MT5 Terminal...
start "" "C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe" /portable

echo Waiting 15 seconds for terminals to connect...
timeout /t 15 /nobreak >nul

echo.
echo [3/7] Starting BrightFunded Live Bot...
start "BF Live" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set ENABLE_LIVE_TRADING=1 && set PYTHONUNBUFFERED=1 && python -u common\live\run_bot.py --live --account-id bright_100k"

echo [4/7] Starting FTMO Live Bot...
start "FTMO Live" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set "ENABLE_LIVE_TRADING=1" && set "PYTHONUNBUFFERED=1" && set "MT5_MODULE=MetaTrader5_FTMO" && python -u common\live\run_bot.py --live --account-id ftmo_100k"

timeout /t 5 /nobreak >nul

echo [5/7] Starting BrightFunded Paper Bot...
start "BF Paper" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set PYTHONUNBUFFERED=1 && python -u common\live\paper_bot.py --account-id bright_100k"

echo [6/7] Starting FTMO Paper Bot...
start "FTMO Paper" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set "PYTHONUNBUFFERED=1" && set "MT5_MODULE=MetaTrader5_FTMO" && python -u common\live\paper_bot.py --account-id ftmo_100k"

echo [7/8] Starting PredMarket Scheduler...
start "PredMarket" cmd /k "cd /d C:\predmarket && call .venv\Scripts\activate.bat && set PYTHONUNBUFFERED=1 && python scheduler.py"

echo [8/10] Starting Telegram Signal Scraper (FTMO)...
start "Telegram Signals" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set "PYTHONUNBUFFERED=1" && set "MT5_MODULE=MetaTrader5_FTMO" && python -u common\tools\telegram_signals.py"

echo [9/10] Starting Trade Copier (BF - copies FTMO trades)...
start "Copier BF" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set "PYTHONUNBUFFERED=1" && python -u common\tools\trade_copier.py --account bright_100k"

echo [10/10] Starting Trade Copier (FTMO - copies BF trades)...
start "Copier FTMO" cmd /k "cd /d C:\tradebots && call .venv\Scripts\activate.bat && set "PYTHONUNBUFFERED=1" && set "MT5_MODULE=MetaTrader5_FTMO" && python -u common\tools\trade_copier.py --account ftmo_100k"

echo.
echo ============================================
echo   All systems started!
echo   2 MT5 Terminals + 4 Bots + PredMarket + Telegram + 2 Copiers
echo ============================================
echo.
pause
