# Migratie Linux → Windows Server

## Hardware
- CPU: Xeon E5-2690 v4 (28 threads)
- RAM: 62 GB
- GPU: Tesla P40 (24GB) + GTX 1050 (2GB)
- NVMe: 238 GB (huidige Linux boot)
- SSD sda1: 916 GB (tick data)
- SSD sdd2: 457 GB (tick data, 418 GB vrij)

## Data inventaris
| Wat | Locatie | Grootte |
|-----|---------|---------|
| tradebots/ (excl .venv) | NVMe | ~25 GB |
| .venv | NVMe | 8.6 GB (NIET kopiëren, opnieuw opbouwen) |
| tick data sda1 | /home/tradebot/ssd_data_1/tick_data/ | 69 GB |
| tick data sdd2 | /home/tradebot/ssd_data_2/tick_data/ | 14 GB |
| tick data NVMe | /home/tradebot/data_1/tick_data/ | 13 GB |
| **Totaal te backuppen** | | **~120 GB** |

---

## Fase 0: Voorbereiding (vrijdagavond)

### 0.1 Bots stoppen
```bash
systemctl --user stop sovereign-bot-ftmo
systemctl --user stop sovereign-bot-bf
systemctl --user stop sovereign-bot-ttp
# Controleer geen open posities!
```

### 0.2 Backup tradebots naar sdd2
```bash
# sdd2 heeft 418 GB vrij
rsync -av --exclude='.venv' /home/tradebot/tradebots/ /home/tradebot/ssd_data_2/backup_tradebots/
rsync -av /home/tradebot/data_1/tick_data/ /home/tradebot/ssd_data_2/backup_tick_data_nvme/
```

### 0.3 Noteer alle credentials
- FTMO MT5 login/password/server
- BF MT5 login/password/server
- TTP MT5 login/password/server
- Discord webhook URL
- Alpaca API keys
- Ollama model (qwen2.5:32b)

---

## Fase 1: Windows Server installeren (zaterdagochtend)

### 1.1 Windows Server 2022/2025 op NVMe
- Boot van USB installer
- Installeer op NVMe (wist Linux partitie)
- Desktop Experience versie (niet Core)
- sda1 en sdd2 NIET aanraken!

### 1.2 Eerste setup
- Admin wachtwoord instellen
- Remote Desktop aanzetten (Systeeminstellingen → Remote Desktop → Aan)
- OpenSSH Server installeren (Settings → Apps → Optional Features)
- Windows Update draaien (één keer, daarna uitschakelen voor stabiliteit)

### 1.3 SATA schijven mounten
- Disk Management: sda1 en sdd2 worden herkend als "Unknown" (ext4)
- Installeer Paragon ExtFS for Windows (of Linux Reader)
- OF: kopieer data via Linux live USB naar NTFS partitie eerst

**Let op**: ext4 lezen vanuit Windows is mogelijk maar niet ideaal.
**Veiligste optie**: vóór Windows installatie, vanuit Linux de backup op sdd2
naar een NTFS-geformatteerde partitie kopiëren.

### 1.4 Alternatief: sdd2 NTFS formatteren vanuit Linux
```bash
# VOOR Windows installatie - alleen als sdd2 backup klaar is op sda1
# Verplaats tick data sdd2 naar sda1 eerst:
rsync -av /home/tradebot/ssd_data_2/tick_data/ /home/tradebot/ssd_data_1/tick_data_sdd2_backup/
# Dan sdd2 NTFS formatteren:
sudo mkfs.ntfs -f /dev/sdd2
sudo mount /dev/sdd2 /mnt/sdd2
# Kopieer alles wat Windows nodig heeft:
rsync -av --exclude='.venv' /home/tradebot/tradebots/ /mnt/sdd2/tradebots/
rsync -av /home/tradebot/ssd_data_1/tick_data/ /mnt/sdd2/tick_data/ssd1/
rsync -av /home/tradebot/data_1/tick_data/ /mnt/sdd2/tick_data/nvme/
```

---

## Fase 2: Software installeren (zaterdagmiddag)

### 2.1 NVIDIA Drivers
- Download NVIDIA driver voor Tesla P40 + GTX 1050
- Game Ready driver of Studio driver (beide werken)
- Herstart na installatie
- Controleer: `nvidia-smi` in PowerShell

### 2.2 CUDA Toolkit
- Download CUDA Toolkit 12.x van developer.nvidia.com
- Installeer met default opties
- Controleer: `nvcc --version`

### 2.3 Python 3.12
- Download Python 3.12.x van python.org (NIET 3.14 — MT5 package ondersteunt tot 3.12)
- ✅ "Add to PATH" aanvinken bij installatie
- Controleer: `python --version`

### 2.4 Git
- Download van git-scm.com
- Default opties bij installatie
- Controleer: `git --version`

### 2.5 Claude Code
```powershell
irm https://claude.ai/install.ps1 | iex
```

### 2.6 NSSM (service manager)
- Download van nssm.cc
- Zet `nssm.exe` in `C:\tools\` en voeg toe aan PATH

### 2.7 Ollama (voor daily reports)
- Download Ollama voor Windows
- `ollama pull qwen2.5:32b`

---

## Fase 3: MT5 Terminals installeren (zaterdagmiddag)

### 3.1 FTMO Terminal
- Installeer in `C:\MT5\FTMO\`
- Login met FTMO credentials
- Sla login op (Remember password)
- Tools → Options → Expert Advisors → Allow algo trading ✅
- Tools → Options → Expert Advisors → Allow DLL imports ✅

### 3.2 BF Terminal
- Installeer in `C:\MT5\BF\`
- Login met BF credentials
- Zelfde instellingen als FTMO

### 3.3 TTP Terminal
- Installeer in `C:\MT5\TTP\`
- Login met TTP credentials
- Zelfde instellingen als FTMO

### 3.4 Auto-start instellen
- Snelkoppelingen naar alle 3 terminals in:
  `C:\Users\<user>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`
- Of via Task Scheduler: "At startup" trigger per terminal

---

## Fase 4: Codebase opzetten (zaterdagavond)

### 4.1 Data kopiëren
```powershell
# Vanuit sdd2 (NTFS) naar definitieve locaties
mkdir C:\tradebots
xcopy D:\tradebots C:\tradebots /E /H
mkdir C:\tick_data
xcopy D:\tick_data C:\tick_data /E /H
```

### 4.2 Python venv opbouwen
```powershell
cd C:\tradebots
python -m venv .venv
.venv\Scripts\activate
pip install xgboost polars numpy pandas scikit-learn MetaTrader5 requests pyyaml optuna discord-webhook
# Plus overige packages uit requirements.txt
```

### 4.3 Code aanpassingen

#### MT5 Bridge verwijderen → Native API
Huidige code (Linux + Wine bridge):
```python
from common.tools.mt5_bridge import MT5BridgeClient
mt5 = MT5BridgeClient(port=5056)
positions = mt5.positions_get()
```

Nieuwe code (Windows native):
```python
import MetaTrader5 as mt5
mt5.initialize(path=r"C:\MT5\FTMO\terminal64.exe")
positions = mt5.positions_get()
```

#### Bestanden die aangepast moeten worden:
- `common/tools/mt5_bridge.py` → wrapper die native MT5 aanroept ipv TCP
- `common/live/run_bot.py` → initialize per account met terminal path
- `common/execution/order_router.py` → geen bridge meer
- `common/execution/position_manager.py` → geen bridge meer
- `common/live/paper_tracker.py` → geen bridge meer
- `common/tools/mt5_bridge_proxy.py` → VERWIJDEREN (niet meer nodig)
- `live/run_wine.sh` → VERWIJDEREN
- `live/launcher.py` → VERWIJDEREN

#### Pad aanpassingen
```python
# Linux paden → Windows paden
# /home/tradebot/tradebots/ → C:\tradebots\
# /home/tradebot/ssd_data_1/tick_data/ → C:\tick_data\
```
Config files updaten: config/loader.py, DATA_ROOTS, BAR_ROOTS

#### Slimme aanpak: MT5BridgeClient wrapper behouden
In plaats van alle code te herschrijven, kunnen we `MT5BridgeClient`
aanpassen zodat het onder water native MT5 aanroept. Dan hoeft de
rest van de code NIET te veranderen.

```python
# common/tools/mt5_bridge.py (nieuwe versie voor Windows)
import MetaTrader5 as mt5

class MT5BridgeClient:
    """Drop-in replacement: wraps native MT5 API with same interface."""

    TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
    TRADE_ACTION_SLTP = mt5.TRADE_ACTION_SLTP
    TRADE_RETCODE_DONE = mt5.TRADE_RETCODE_DONE

    def __init__(self, port=None, terminal_path=None):
        if terminal_path:
            mt5.initialize(path=terminal_path)

    def positions_get(self, **kwargs):
        return mt5.positions_get(**kwargs)

    def symbol_info(self, symbol):
        return mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol):
        return mt5.symbol_info_tick(symbol)

    def order_send(self, request):
        return mt5.order_send(request)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return mt5.copy_rates_from_pos(symbol, timeframe, start, count)

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        return mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    def account_info(self):
        return mt5.account_info()

    def history_deals_get(self, date_from, date_to, **kwargs):
        return mt5.history_deals_get(date_from, date_to, **kwargs)
```

Met deze wrapper hoeft GEEN ENKELE andere file aangepast te worden.
Alleen mt5_bridge.py zelf + de initialisatie in run_bot.py.

---

## Fase 5: Services opzetten (zondagochtend)

### 5.1 NSSM services aanmaken
```powershell
# FTMO bot
nssm install sovereign-bot-ftmo "C:\tradebots\.venv\Scripts\python.exe"
nssm set sovereign-bot-ftmo AppParameters "-u C:\tradebots\common\live\run_bot.py --live --account-id ftmo_100k"
nssm set sovereign-bot-ftmo AppDirectory "C:\tradebots"
nssm set sovereign-bot-ftmo AppStdout "C:\tradebots\ftmo\logs\service.log"
nssm set sovereign-bot-ftmo AppStderr "C:\tradebots\ftmo\logs\service_err.log"
nssm set sovereign-bot-ftmo AppRestartDelay 5000

# BF bot
nssm install sovereign-bot-bf "C:\tradebots\.venv\Scripts\python.exe"
nssm set sovereign-bot-bf AppParameters "-u C:\tradebots\common\live\run_bot.py --live --account-id bright_100k"
nssm set sovereign-bot-bf AppDirectory "C:\tradebots"
nssm set sovereign-bot-bf AppStdout "C:\tradebots\bf\logs\service.log"
nssm set sovereign-bot-bf AppStderr "C:\tradebots\bf\logs\service_err.log"
nssm set sovereign-bot-bf AppRestartDelay 5000

# TTP bot
nssm install sovereign-bot-ttp "C:\tradebots\.venv\Scripts\python.exe"
nssm set sovereign-bot-ttp AppParameters "-u C:\tradebots\common\live\run_bot.py --live --account-id ttp_demo"
nssm set sovereign-bot-ttp AppDirectory "C:\tradebots"
nssm set sovereign-bot-ttp AppStdout "C:\tradebots\ttp\logs\service.log"
nssm set sovereign-bot-ttp AppStderr "C:\tradebots\ttp\logs\service_err.log"
nssm set sovereign-bot-ttp AppRestartDelay 5000
```

### 5.2 Services starten
```powershell
nssm start sovereign-bot-ftmo
nssm start sovereign-bot-bf
nssm start sovereign-bot-ttp
```

### 5.3 Task Scheduler: Ritual timers
- `ritual-ftmo`: Zaterdag 00:00
- `ritual-bf`: Zaterdag 12:00
- `daily-report`: Dagelijks 00:00

---

## Fase 6: Testen (zondagmiddag)

### 6.1 Checklist
- [ ] Alle 3 MT5 terminals open en ingelogd
- [ ] `nvidia-smi` toont beide GPU's
- [ ] Python + venv werkt met alle packages
- [ ] MT5 Python API werkt: `python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.account_info())"`
- [ ] Tick data bereikbaar op juiste paden
- [ ] Modellen laden correct
- [ ] Paper mode test per bot (1 uur draaien)
- [ ] Discord notificaties komen door
- [ ] NSSM services herstarten na crash
- [ ] Claude Code werkt via PowerShell
- [ ] Ollama draait + daily report test

### 6.2 Live gaan
- Zondagavond PAPER mode aanzetten op alle bots
- Maandagochtend controleren of alles stabiel draait
- Als alles goed is: LIVE mode aanzetten
- Linux NVMe partitie bewaren als backup (of later wissen)

---

## Risico's en fallback

| Risico | Mitigatie |
|--------|----------|
| Windows installatie mislukt | Linux live USB, data is op sda1/sdd2 |
| CUDA driver problemen | GTX 1050 als fallback display, P40 voor compute |
| Python packages werken niet | requirements.txt + pip freeze van Linux als referentie |
| MT5 login issues | Credentials genoteerd, broker support bereikbaar |
| Data verloren | Backup op sdd2, tick data op sda1 onaangeraakt |

## Tijdschatting
| Fase | Tijd |
|------|------|
| 0: Voorbereiding + backup | 2-3 uur |
| 1: Windows installeren | 1 uur |
| 2: Software installeren | 2-3 uur |
| 3: MT5 terminals | 1 uur |
| 4: Code + venv | 2-3 uur |
| 5: Services | 1 uur |
| 6: Testen | 2-3 uur |
| **Totaal** | **~12-16 uur (1 weekend)** |
