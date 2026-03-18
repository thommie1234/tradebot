# FTMO $100k Post-Mortem — Account 541095703

**Date failed:** 2026-02-25
**Reason:** Hit -10% max drawdown limit ($100k → $90,003.65)

## Key stats
- 242 trades (133 long, 109 short) over ~5 weeks (Jan 25 - Feb 25)
- Win rate: 48.76%, Profit Factor: 0.59, Sharpe: -0.14
- Gross profit: +$14,520 | Gross loss: -$24,516 | Fees: -$861
- Average P/L per trade: -$41.05
- Max deposit load: 100.24% (all margin used)
- Trades/week: 48, Avg hold: 4h 47m
- Jan: 17 trades (-$706) → Feb: 467 trades (-$9,290) = 27x escalation

## Root causes (ranked by impact)

### 1. Structureel negatieve verwachting
PF 0.59 — het systeem had geen edge na kosten. Zelfs zonder bugs of emotionele fouten verliest dit portfolio geld. De OOS validatie was positief, maar de live execution verschilde door fees, bugs, en timing.

### 2. Weekly portfolio churn (meta-overfitting)
Elke week Optuna → nieuw portfolio → max 5-7 dagen per config. Individuele OOS was positief, maar:
- Strategie krijgt nooit de kans om verwachte waarde te realiseren
- Herhaald kiezen uit duizenden trials = selection bias op meta-niveau
- Afkappen bij eerste drawdown → je realiseert alleen de losing start van elke strategie

### 3. Crypto fees
40% van trades was crypto. Fees waren moorddadig:
- DOGEUSD: $161 fees op 32 trades ($5/trade)
- BNBUSD, UNIUSD, ICPUSD, BCHUSD, SOLUSD: elk $49-57 aan fees
- Veel crypto had positieve gross profit maar netto verlies door fees

### 4. Handmatig ingrijpen ("handen jeukten")
- ML threshold 0.55 → toch signalen van 0.54 handmatig uitvoeren
- Spread te hoog → toch traden
- Parameters live aangepast op gevoel
- Bypassed alle safeguards die ingebouwd waren

### 5. Risk te hoog (1% per trade)
- 1% van $100k = $1000 risk per trade
- FTMO limiet = $10,000 (10%)
- Slechts 10 consecutive losers = game over
- Met 48.76% win rate en gecorreleerde posities is dat realistisch

### 6. Bugs in productie
- SL/TP niet correct gezet
- Bot crashes/disconnects
- Position sizing te groot (meer dan bedoeld)
- Een buggy bot + 1% risk + geen guards = tijdbom

### 7. Geen functionerende guards
- FTMO guard status onbekend ("weet niet zeker")
- Max deposit load bereikte 100.24%
- Geen circuit breaker bij oplopende DD

## Equity curve patroon
```
Jan 25 - Feb 16:  Vlak met kleine schommelingen (~$0 netto)
Feb 16 - Feb 21:  Gestage daling (~-$3k)
Feb 22 - Feb 25:  Vrije val (~-$5k in 3 dagen) → FAIL
```

## Regels voor BrightFunded (en elke toekomstige prop firm)

| Regel | Implementatie |
|---|---|
| Max 0.25-0.5% risk per trade | 20-40 losers buffer tot DD limiet |
| Minimum 4 weken zelfde config | Geen weekly re-optimization |
| Crypto fee filter | Ban symbolen met >$3/trade in fees, of weeg fees mee in WFO |
| NOOIT handmatig ingrijpen | Als je bot niet vertrouwt, zet hem uit. Nooit threshold/spread bypassen |
| Harde daily loss guard | Bot stopt bij -2% dagverlies |
| Position size cap | Max 2% margin per trade, hard cap in code |
| Guard verificatie | Test FTMO/DD guards VOOR je live gaat, bewijs dat ze werken |
| OOS moet robuust zijn | Gebruik purged WFO gemiddeld over alle folds, niet 1 goede fold |
| Bugfix voor live | Geen bekende bugs deployen. SL/TP en sizing MOETEN werken |

## OOS → Live gap analyse
De OOS was positief maar live faalde. Oorzaken:
1. **Fees niet volledig meegenomen** — crypto spreads live breder dan historisch
2. **Meta-overfitting** — wekelijks best-portfolio kiezen = selection bias
3. **Execution bugs** — SL/TP falen, sizing bugs bestonden niet in backtest
4. **Geen tijd voor mean reversion** — elke strategie afgekapt bij eerste DD
