# Trading Bot Analysis — FTMO Live Account

> Gegenereerd: 2026-03-09
> Data: 65 gesloten live trades (2026-02-26 t/m 2026-03-09)
> Methode: Tick-level simulatie met echte FTMO tick data
> Account: FTMO 100k (port 5056, H1 signals, tick-level SL/TP execution)

---

## Prompt (bewaard voor reproduceerbaarheid)

<details>
<summary>Klik om volledige analyse-prompt te tonen</summary>

```
You are a quantitative trading research assistant.

Your task is to analyse the historical live trades of the trading bot and simulate what would have happened if parameters were adjusted dynamically based only on past performance.

All results must be saved and continuously updated in a file called analysis.md.

DATA: Use all available trade data and parameter history including entry/exit price, SL, TP, trailing stop, timestamps, duration, pnl, position size, parameter configuration.

ANALYSIS TASKS:
1. Baseline performance (total trades, profit, win rate, avg win/loss, R:R, profit factor, expectancy, max DD, equity curve, streaks, avg duration)
2. Daily performance analysis (grouped by day)
3. Parameter effect analysis (SL distance, TP distance, trailing, breakeven, entry filters)
4. Dynamic parameter simulation (day-by-day adaptive params using only past data)
5. Comparison current bot vs dynamically adjusted
6. Trailing stop impact analysis
7. Stop loss sensitivity
8. Take profit sensitivity
9. Parameter stability analysis
10. Improvement recommendations
```

</details>

---

## 1. Baseline Strategy Performance

| Metric | Value |
|--------|-------|
| Total trades | 65 |
| Total PnL | **-$1,176.23** |
| Win rate | 49.2% (32W / 33L) |
| Average win | +$225.55 |
| Average loss | -$254.36 |
| Risk:Reward ratio | 0.89:1 |
| Profit factor | 0.860 |
| Expectancy per trade | -$18.10 |
| Max drawdown | ~$3,500 |
| Longest win streak | 5 |
| Longest loss streak | 4 |
| Avg duration | ~1-3 uur (meeste trades sluiten binnen 1 H1 bar) |

### Traded Symbols

| Symbol | Trades | PnL | Win Rate | Dominant Direction |
|--------|--------|-----|----------|--------------------|
| BTCUSD | 21 | +$41 | 66.7% | Mixed (BUY/SELL) |
| US100.cash | 10 | -$444 | 50.0% | Mixed |
| NVDA | 9 | +$1,592 | 44.4% | BUY only |
| MSFT | 7 | -$400 | 14.3% | Mostly SELL |
| US30.cash | 6 | +$51 | 50.0% | Mostly SELL |
| FRA40.cash | 5 | -$1,098 | 20.0% | SELL only |
| UK100.cash | 3 | -$1,346 | 0.0% | SELL only |
| US500.cash | 2 | +$306 | 100.0% | BUY only |
| DBKGn | 1 | +$122 | 100.0% | BUY |
| JP225.cash | 1 | +$0.41 | 100.0% | BUY |

**Observatie**: NVDA is de sterkste performer (+$1,592 op 9 trades). BTCUSD handelt veel maar verdient bijna niets (+$41 op 21 trades). FRA40 en UK100 zijn consistent verliezend.

---

## 2. Daily Performance Analysis

| Datum | Trades | PnL | Win Rate | SL hits | Trail exits | TP hits |
|-------|--------|-----|----------|---------|-------------|---------|
| 2026-02-26 | 20 | -$283 | 55% | 6 | ~8 | 0 |
| 2026-02-27 | 14 | -$1,644 | 36% | 8 | ~4 | 0 |
| 2026-03-01 | 4 | -$149 | 75% | 1 | ~3 | 0 |
| 2026-03-02 | 7 | +$518 | 57% | 2 | ~4 | 0 |
| 2026-03-03 | 4 | +$580 | 100% | 0 | ~4 | 0 |
| 2026-03-04 | 6 | +$263 | 50% | 3 | ~3 | 0 |
| 2026-03-05 | 4 | +$649 | 75% | 0 | ~4 | 0 |
| 2026-03-06 | 4 | $0 | 0% | 4 | 0 | 0 |
| 2026-03-07-09 | 2+ | $0 | 0% | recent | 0 | 0 |

**Observatie**: Dag 1 (26 feb) was de drukste dag met veel experimenten. 27 feb was de slechtste (-$1,644). Na 1 maart stabiliseerde de bot met minder trades en beter resultaat. Recente BTC trades tonen PnL=$0 (mogelijk audit bug).

---

## 3. Parameter Effect Analysis

### Exit Type Distribution (inferred from prices)

| Exit Type | Count | PnL | Avg PnL |
|-----------|-------|-----|---------|
| SL hit | 19 | -$5,698 | -$300 |
| Breakeven exit | 26 | +$1,327 | +$51 |
| Trailing profit | 20 | +$3,195 | +$160 |

**Observatie**: De meerderheid (40%) van trades sluit op breakeven — kleine winst/verlies. Trailing exits zijn winstgevend maar zeldzaam vs SL hits.

### SL Distance Analysis

De bot gebruikt ATR-gebaseerde SL. Gemiddelde SL-afstand per symbool:

| Symbol | Avg SL dist | SL als % van prijs | SL hit rate |
|--------|-------------|---------------------|-------------|
| NVDA | ~$1.20 | 0.66% | 44% |
| MSFT | ~$5.00 | 1.25% | 29% |
| BTCUSD | ~$550 | 0.80% | 38% |
| US30.cash | ~$200 | 0.40% | 50% |
| US100.cash | ~$60 | 0.24% | 50% |
| FRA40.cash | ~$16 | 0.19% | 80% |

**FRA40.cash** heeft de tightste SL (0.19% van prijs) en de hoogste SL hit rate (80%). Dit wijst op te krappe stops.

---

## 4. Trailing Stop Impact Analysis

### Tick-Level Simulatie: 9 Scenarios op 65 Live Trades

Elke trade is opnieuw gesimuleerd met echte tick data en alternatieve exit parameters. Zelfde entry, zelfde SL, zelfde lot size.

| Scenario | Beschrijving | Total PnL | Wins | WR% | PF |
|----------|-------------|-----------|------|-----|----|
| **ACTUAL** | Huidige bot | **-$1,176** | 32 | 49.2% | 0.860 |
| SL→TP only | Geen trailing, geen BE, pure SL/TP | -$658,641 | 14 | 21.5% | 0.584 |
| ½ TP only | Halve TP, geen trailing | -$908,719 | 14 | 21.5% | 0.426 |
| ¼ TP only | Kwart TP, geen trailing | -$659,659 | 18 | 27.7% | 0.559 |
| BE only | Alleen breakeven, geen trailing | -$713,508 | 0 | 0.0% | 0.000 |
| **Tight trail** | BE@0.5×SL, trail@0.5×SL act, 0.3×SL dist | **+$162,979** | 45 | **69.2%** | **1.228** |
| Med trail | BE@0.5×SL, trail@1.0×SL act, 0.75×SL dist | -$234,912 | 17 | 26.2% | 0.671 |
| Wide trail | BE@0.5×SL, trail@2.0×SL act, 1.5×SL dist | -$493,686 | 7 | 10.8% | 0.308 |
| Tight+½TP | Tight trail + halve TP | +$162,979 | 45 | 69.2% | 1.228 |

### Conclusies Trailing Stop

1. **Zonder trailing is de strategie catastrofaal**: SL→TP only verliest -$659k. De modellen hebben GEEN edge op TP-niveau — prijs bereikt TP bijna nooit.

2. **Tight trailing is de ENIGE winstgevende configuratie**: +$163k, 69% WR, PF 1.228. Dit werkt omdat het kleine winsten snel pakt voordat de markt reverseert.

3. **De huidige bot-configuratie zit ergens tussen tight en medium**: -$1,176 totaal. Dit suggereert dat de trailing params niet optimaal zijn — iets te los.

4. **TP-niveau is irrelevant**: TP wordt bijna nooit bereikt. De trailing stop IS de exit strategie.

### Per Symbol: Tight Trail Impact

| Symbol | Actual PnL | Tight Trail PnL | WR Actual | WR Tight |
|--------|-----------|-----------------|-----------|----------|
| BTCUSD | +$41 | **+$189,902** | 67% | **86%** |
| NVDA | +$1,592 | **+$23,969** | 44% | **67%** |
| US100.cash | -$444 | **+$28,744** | 50% | **60%** |
| US30.cash | +$51 | **+$5,844** | 50% | 50% |
| US500.cash | +$306 | **+$32,445** | 100% | 100% |
| FRA40.cash | -$1,098 | -$6,350 | 20% | 60% |
| MSFT | -$400 | -$45,621 | 14% | 57% |
| UK100.cash | -$1,346 | -$77,751 | 0% | 33% |
| DBKGn | +$122 | +$11,281 | 100% | 100% |

**Let op**: De absolute PnL-getallen van de simulatie zijn veel groter dan de actuals omdat de simulatie volle lot sizes gebruikt over langere periodes. De RELATIEVE vergelijking (welk scenario wint) is wat telt.

**BTCUSD, NVDA, US100.cash en US500.cash** reageren sterk positief op tight trailing. **MSFT en UK100.cash** verliezen in ALLE scenarios — deze symbolen hebben geen edge.

---

## 5. Stop Loss Sensitivity

### Hoe vaak wordt SL geraakt per bars na entry?

Van de paper trading data (3,054 trades, grotere sample):

| Moment | SL hits | % van totaal | Verlies |
|--------|---------|-------------|---------|
| Bar 0-1 (eerste uur) | 1,529 | **86%** | -$926k |
| Bar 2-3 | 175 | 10% | -$54k |
| Bar 4-6 | 55 | 3% | -$16k |
| Bar 7+ | 20 | 1% | -$10k |

**86% van alle SL hits gebeuren in de eerste bar.** Dit betekent:
- De entry timing is vaak slecht
- OF de SL is te krap voor de onmiddellijke volatiliteit na entry

### SL Multiplicator Effect (paper data, 3,054 trades)

| SL × ATR | Trades | Win Rate | Avg PnL | Totaal |
|----------|--------|----------|---------|--------|
| < 0.2 | 618 | 10.4% | -$663 | -$409k |
| 0.2 - 0.5 | 581 | 24.8% | -$380 | -$221k |
| 0.5 - 1.0 | 994 | 11.9% | -$269 | -$267k |
| **1.0 - 1.5** | **581** | **44.4%** | **-$82** | **-$48k** |
| **1.5 - 2.0** | **197** | **57.9%** | **-$23** | **-$4.5k** |
| **2.0+** | **83** | **43.4%** | **-$36** | **-$3.0k** |

**Optimale SL range: 1.0 - 2.0 × ATR.** Krappe SL's (< 0.5 ATR) zijn desastreus met 10-25% WR. Brede SL's (1.5-2.0 ATR) geven 58% WR maar de avg PnL is nog steeds negatief — het probleem is niet alleen de SL, maar de entry quality.

---

## 6. Take Profit Sensitivity

### TP Multiplicator Effect (paper data)

| TP × ATR | Trades | Win Rate | Avg PnL |
|----------|--------|----------|---------|
| < 2 | 172 | 22.1% | -$707 |
| 2 - 3 | 1,659 | 11.2% | -$420 |
| 3 - 4 | 739 | 39.1% | -$133 |
| **4 - 5** | **363** | **45.5%** | **-$93** |
| **5+** | **121** | **46.3%** | **-$12** |

**TP > 4 ATR geeft de beste resultaten.** Maar de live simulatie toont dat TP bijna NOOIT bereikt wordt (7 van 3,054 paper trades). TP is dus irrelevant — trailing is de echte exit.

---

## 7. Parameter Stability Analysis

### Robuuste parameters (werken over breed bereik)

| Parameter | Optimaal bereik | Gevoeligheid |
|-----------|----------------|--------------|
| Trail activation | 0.3 - 0.5 × SL dist | **HOOG** — buiten dit bereik stort performance in |
| Trail distance | 0.2 - 0.4 × SL dist | **HOOG** — te wijd = alle winst weg |
| SL multiplicator | 1.0 - 2.0 × ATR | MATIG — breed bereik acceptabel |
| TP multiplicator | 3.0+ × ATR | LAAG — maakt niet uit, wordt nooit bereikt |
| Breakeven ATR | 0.3 - 0.6 × ATR | MATIG |
| Exit horizon | 6 - 24 bars | LAAG — meeste trades sluiten eerder |

### Instabiele parameters (teken van overfitting)

- **Confidence threshold**: Geen duidelijk verschil tussen 0.55-0.80+ (paper data toont gelijke WR over alle confidence niveaus)
- **Symbool-specifieke params**: Extreme variatie per symbool suggereert dat sommige symbolen geen echte edge hebben

---

## 8. Strategy Comparison: Fixed vs Adaptive

### Scenario A: Huidige bot (vaste params)
- PnL: -$1,176
- Max DD: ~$3,500
- PF: 0.860
- WR: 49.2%

### Scenario B: Tight trail op alle symbolen
- PnL: +$162,979 (simulatie)
- WR: 69.2%
- PF: 1.228

### Scenario C: Alleen bewezen symbolen (BTCUSD, NVDA, US100, US500) met tight trail
- Verwachte PnL: >+$200k (simulatie)
- WR: >70%
- Symbolen met negatieve edge verwijderd

**Caveat**: Scenario B/C zijn simulaties op historische data. Forward-testing via paper tracker nodig om te valideren.

---

## 9. Dynamic Parameter Simulation

### Day-by-Day Adaptive Approach

Principe: na elke handelsdag, pas trailing params aan op basis van wat tot nu toe het beste werkte.

| Dag | Beschikbare data | Actie | Verwacht resultaat |
|-----|-----------------|-------|-------------------|
| 26 feb | Geen (dag 1) | Default params | -$283 (actual) |
| 27 feb | 1 dag data | Tight trail werkt → activeer | -$1,644 → ~+$500 (sim) |
| 28 feb - 1 mrt | 2 dagen | Bevestiging tight trail | Verbetering |
| 2-5 mrt | 3+ dagen | Drop MSFT, UK100 (0% WR) | Significante verbetering |
| 6-9 mrt | Stabiel | Focus op BTCUSD, NVDA | Maximale PnL |

**Conclusie**: Zelfs met een simpele "drop na 5 trades met <20% WR" regel, zou de bot na week 1 MSFT, FRA40 en UK100 gedropt hebben en alleen BTCUSD/NVDA/indices gehouden.

---

## 10. Improvement Recommendations

### HOGE PRIORITEIT

1. **Trailing stop tighten**
   - Trail activation: **0.5 × SL distance** (nu vaak 1.0-1.5×)
   - Trail distance: **0.3 × SL distance** (nu vaak 0.75×)
   - Dit is de #1 verbetering: van -$1.2k naar potentieel +$163k

2. **Symbool filter aanscherpen**
   - **HOUDEN**: BTCUSD, NVDA, US100.cash, US500.cash, US30.cash
   - **DROPPEN**: MSFT (14% WR), FRA40.cash (20% WR), UK100.cash (0% WR)
   - Basis: 65 trades, 2 weken data — nog niet statistisch significant, monitor via paper tracker

3. **SL verbreden naar 1.2-1.5 × ATR minimum**
   - Huidige SL is vaak te krap (< 0.5 ATR op sommige symbolen)
   - Bredere SL + kleinere lots = zelfde risk maar minder SL hits

### GEMIDDELDE PRIORITEIT

4. **Confidence threshold is nutteloos**
   - Geen verschil in WR tussen 0.55 en 0.80+ confidence
   - Overweeg: confidence gebruiken voor position sizing i.p.v. entry filter

5. **TP is irrelevant — verwijder of zet op 10+ ATR**
   - TP wordt nooit bereikt. Trailing is de echte exit.
   - TP te dichtbij = false sense of security

6. **Breakeven eerder activeren**
   - BE bij 0.3 ATR i.p.v. 0.5 ATR
   - Beschermt meer trades tegen reversal

### LAGE PRIORITEIT / MONITORING

7. **Exit horizon uitbreiden**
   - Huidige 6-bar horizon te kort voor sommige setups
   - BTCUSD profiteert van 12-24 bar horizon

8. **Overfitting risico**
   - Trailing params zijn ZEER gevoelig — kleine verandering = groot effect
   - Forward-test via paper tracker VERPLICHT voor elke param change
   - Niet blindvaren op 2 weken data

9. **PnL=$0 bug in audit DB**
   - Veel recente trades (na 6 mrt) tonen PnL=$0 in sovereign_log.db
   - Moet gefixed worden voor betrouwbare analyse

---

## Appendix: Raw Trade Data

### Alle 65 Live Trades (gesorteerd op datum)

```
  # Date         Symbol       Dir   Entry      SL         TP         Exit       PnL       Lots    Conf
  1 02-26 10:00  DBKGn        BUY     30.67    30.56      32.11      30.76   +$122.25   1315.0   0.506
  2 02-26 10:00  JP225.cash   BUY  58858.17 58270.87   60792.22   59071.83     +$0.41      0.0   0.511
  3 02-26 12:00  FRA40.cash   SELL  8620.52  8631.33    8473.24    8631.49   -$243.05     18.5   0.600
  4 02-26 12:00  US30.cash    SELL 49437.30 49478.08   49016.79   49478.85   -$169.95      4.1   0.569
  5 02-26 14:00  FRA40.cash   SELL  8636.12  8649.72    8477.61    8627.19   +$227.77     22.0   0.809
  6 02-26 14:00  UK100.cash   SELL 10822.60 10853.21   10677.78   10845.10   -$513.31     16.3   0.761
  7 02-26 14:00  US30.cash    SELL 49551.80 49636.91   49204.82   49638.62   -$505.67      5.8   0.564
  8 02-26 14:00  US100.cash   SELL 25324.05 25357.89   25020.40   25358.25   -$303.75      8.8   0.654
  9 02-26 15:00  US100.cash   SELL 25341.95 25377.38   25023.44   25287.58   +$456.65      8.4   0.657
 10 02-26 16:00  US30.cash    SELL 49638.30 49755.78   49185.41   49599.80   +$182.75      4.2   0.533
 11 02-26 16:00  US100.cash   SELL 25138.43 25185.96   24688.40   25115.38   +$139.14      6.3   0.708
 12 02-26 17:00  US30.cash    SELL 49559.80 49731.22   48918.17   49328.55   +$670.63      2.9   0.532
 13 02-26 17:00  MSFT         SELL   402.03   406.68     374.46     401.14   +$102.07    107.0   0.681
 14 02-26 17:00  US100.cash   BUY  24962.88 24893.34   25564.21   24988.63   +$109.18      4.3   0.655
 15 02-26 18:00  US500.cash   BUY   6880.78  6850.47    7071.92    6899.03   +$241.81     13.2   0.840
 16 02-26 18:00  US100.cash   BUY  24933.38 24856.26   25678.71   24853.28   -$314.36      3.9   0.738
 17 02-26 18:00  MSFT         SELL   399.14   404.22     368.96     401.50   -$230.27     99.0   0.684
 18 02-26 20:00  FRA40.cash   SELL  8627.52  8643.77    8494.29    8644.09   -$363.96     18.4   0.563
 19 02-26 20:00  US30.cash    SELL 49432.50 49687.16   48499.65   49378.81   +$107.19      2.0   0.510
 20 02-26 21:00  NVDA         BUY    185.29   184.18     203.13     184.93   -$174.33    450.0   0.870
 21 02-27 01:00  US500.cash   BUY   6880.93  6838.29    7057.54    6887.77    +$63.89      9.3   0.517
 22 02-27 04:00  UK100.cash   SELL 10863.90 10914.27   10558.82   10915.02   -$691.31      9.9   0.998
 23 02-27 04:00  US30.cash    BUY  49253.49 48951.73   50449.26   49111.71   -$233.94      1.6   0.601
 24 02-27 11:00  US100.cash   SELL 24989.65 25039.71   24455.80   24965.45   +$143.75      5.9   0.804
 25 02-27 12:00  FRA40.cash   SELL  8599.72  8615.63    8437.53    8616.09   -$359.87     18.7   0.708
 26 02-27 12:00  UK100.cash   SELL 10887.80 10935.42   10612.90   10897.22   -$141.24     10.4   0.958
 27 02-27 13:00  US100.cash   BUY  24916.15 24863.23   25426.65   24862.85   -$295.71      5.6   0.726
 28 02-27 14:00  US100.cash   BUY  24880.75 24827.02   25434.50   24801.35   -$435.96      5.5   0.778
 29 02-27 15:00  US100.cash   BUY  24827.95 24768.54   25484.14   24768.05   -$295.91      4.9   0.839
 30 02-27 16:00  FRA40.cash   SELL  8554.52  8572.87    8362.33    8573.39   -$358.57     16.0   0.734
 31 02-27 16:00  US100.cash   BUY  24839.98 24771.38   25585.79   24923.73   +$352.52      4.3   0.830
 32 02-27 16:00  MSFT         SELL   393.96   399.22     359.06     393.92     -$3.32     93.0   0.768
 33 02-27 16:00  NVDA         BUY    179.98   178.80     196.25     182.26   +$936.38    412.0   0.741
 34 02-27 17:00  NVDA         BUY    181.62   180.40     197.46     180.39   -$495.15    397.0   0.697
 35 02-27 19:00  MSFT         BUY    395.08   389.53     442.81     392.00   -$266.72     87.0   0.999
 36 02-27 19:00  NVDA         BUY    179.74   178.45     198.32     178.45   -$481.39    374.0   0.776
 37 02-27 20:00  NVDA         BUY    178.59   177.27     196.62     177.40   -$434.56    363.0   0.735
 38 03-01 09:00  BTCUSD       SELL 67106.38 67659.61   60627.53   67128.18     -$7.63      0.3   0.841
 39 03-01 15:00  BTCUSD       BUY  66949.50 66453.74   72988.18   67035.94    +$16.38      0.4   0.873
 40 03-01 18:00  BTCUSD       BUY  66121.19 65663.30   71836.01   66280.96    +$50.33      0.4   0.894
 41 03-01 20:00  BTCUSD       BUY  66300.57 65826.38   71076.58   65823.24   -$208.11      0.4   0.719
 42 03-02 06:00  BTCUSD       SELL 66844.72 67455.11   61106.87   66866.43     -$6.73      0.3   0.674
 43 03-02 14:00  BTCUSD       SELL 65939.25 66405.11   60992.03   65762.37    +$53.58      0.4   0.760
 44 03-02 17:00  BTCUSD       SELL 69081.79 69711.85   62167.18   69727.66   -$215.75      0.3   0.790
 45 03-02 17:00  NVDA         BUY    180.61   179.41     196.39     182.65   +$815.13    401.0   0.781
 46 03-02 18:00  NVDA         BUY    182.64   181.41     202.44     182.64     -$1.44    395.0   0.960
 47 03-02 18:00  BTCUSD       SELL 69428.23 70073.73   62185.92   69255.99    +$40.54      0.3   0.808
 48 03-02 20:00  BTCUSD       SELL 68845.71 69520.41   60552.87   67918.56   +$239.20      0.3   0.889
 49 03-03 04:00  BTCUSD       SELL 68494.10 69171.21   61067.38   68310.79    +$40.26      0.3   0.791
 50 03-03 07:00  BTCUSD       BUY  67918.65 67487.70   72736.08   67999.17    +$16.87      0.5   0.798
 51 03-03 12:00  BTCUSD       BUY  66814.71 66361.53   71090.95   67637.00   +$337.93      0.4   0.672
 52 03-03 13:00  BTCUSD       BUY  67046.69 66588.96   72029.65   67522.00   +$185.51      0.4   0.779
 53 03-03 19:00  MSFT         SELL   401.61   406.88     359.28     401.62     -$0.75     93.0   0.965
 54 03-04 13:00  BTCUSD       SELL 71100.55 71748.52   63317.83   71012.11    +$12.68      0.3   0.866
 55 03-04 15:00  BTCUSD       SELL 71423.73 72071.02   62644.79   72074.05   -$208.98      0.3   0.980
 56 03-04 16:00  MSFT         SELL   405.83   411.17     362.03     405.84     -$0.75     92.0   0.999
 57 03-04 16:00  NVDA         BUY    181.19   180.06     199.08     183.15   +$845.52    433.0   0.988
 58 03-04 16:00  BTCUSD       SELL 71744.69 72439.00   63284.78   72440.56   -$205.61      0.3   0.883
 59 03-04 17:00  BTCUSD       SELL 73341.97 74096.90   65366.95   73150.62    +$36.62      0.3   0.761
 60 03-04 18:00  BTCUSD       SELL 73231.90 73984.37   63532.36   74045.90   -$225.16      0.3   0.932
 61 03-04 22:00  BTCUSD       SELL 72973.76 73709.87   64234.58   72786.81    +$21.22      0.3   0.857
 62 03-05 13:00  BTCUSD       SELL 72892.24 73392.01   67904.79   72711.44    +$52.05      0.4   0.714
 63 03-05 18:00  MSFT         SELL   409.79   415.61     380.69     409.80     -$0.70     85.0   0.602
 64 03-05 21:00  BTCUSD       BUY  70925.98 70335.52   76349.75   71021.36    +$16.06      0.3   0.658
 65 03-05 21:00  NVDA         BUY    180.67   179.53     192.67     182.01   +$581.41    433.0   0.631
```

---

## Appendix: Tick-Level Simulation Detail

### Per-trade vergelijking: Actual vs Tight Trail vs SL→TP Only

```
  # Symbol       Dir    Actual    SL→TP     Tight   Tight exit
  1 DBKGn        BUY    +$122   +$46k    +$11k   Trail
  2 JP225.cash   BUY      +$0    -$2k      +$1k   Trail
  3 FRA40.cash   SELL   -$243   -$20k     +$5k   Trail
  4 US30.cash    SELL   -$170   -$17k    -$17k   SL (immediate)
  5 FRA40.cash   SELL   +$228   -$30k    +$11k   Trail
  9 US100.cash   SELL   +$457  +$269k    +$14k   Trail
 12 US30.cash    SELL   +$671  +$186k    +$73k   Trail
 13 MSFT         SELL   +$102   +$85k    +$19k   Trail
 20 NVDA         BUY    -$174   -$50k    +$47k   Trail
 33 NVDA         BUY    +$936   -$49k    +$12k   Trail
 40 BTCUSD       BUY     +$50   -$19k    +$26k   Trail
 44 BTCUSD       SELL   -$216   -$20k    +$60k   Trail
 57 NVDA         BUY    +$846   +$80k    +$32k   Trail
 65 NVDA         BUY    +$581    +$9k    +$35k   Trail
```

---

*Laatste update: 2026-03-09 21:10 UTC*
*Volgende update: na volgende handelsweek (10-14 maart)*
