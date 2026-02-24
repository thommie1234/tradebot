# Portfolio Correlation Analysis — 2026-02-18

## WFO Risk-Calibrated Portfolio (13 symbols)

### Correlation Groups (WFO selector, max 2 per group)

| Group | Symbols | Count |
|-------|---------|-------|
| us_tech | AMZN, MSFT | 2/2 |
| eu_equity | LVMH | 1/2 |
| us_idx | US100, US30 | 2/2 |
| apac_idx | AUS200, JP225 | 2/2 |
| jpy_fx | CAD_JPY | 1/2 |
| usd_fx | NZD_USD | 1/2 |
| silver | XAG_AUD | 1/2 |
| oil | UKOIL | 1/2 |

8 groups, none exceeding max-2 limit.

### Real-World Correlation Concern

The selector uses narrow correlation groups, but in practice:

- **AMZN + MSFT + BABA + US100 + US30** = 5 symbols strongly correlated with S&P 500. During a US market crash, all 5 draw down simultaneously.
- **US100 and US30** are ~0.95 correlated — nearly identical.
- **LVMH** also partially moves with US risk-on/off sentiment.

This makes **~6 of 13 symbols** effectively "long US risk".

### True Diversifiers

- **JP225 / AUS200** — Asian session, own dynamics
- **CAD_JPY / NZD_USD** — Forex, carry trades
- **BCHUSD** — Crypto, low correlation with equities
- **XAG_AUD / UKOIL** — Commodities

### Risk Assessment

During a strong US sell-off (-5%), ~5 symbols would draw down simultaneously.
With 0.4%-1.2% risk per symbol, combined US exposure is effectively ~3.5% risk on one "factor".

### Potential Improvements

1. **Broader correlation group**: Create a "US_all" super-group covering us_tech + us_idx + us_equity, cap at 3-4 total
2. **Stricter max-corr**: Set --max-corr 1 for closely related groups
3. **Factor-based correlation**: Instead of static groups, compute rolling correlation from returns
4. **Portfolio-level DD sim**: Simulate concurrent DD across correlated symbols to estimate true max DD

### Symbols That Didn't Make FTMO Cut (close)

| Symbol | DD% | Reason |
|--------|-----|--------|
| HK50.cash | 10.8% | Just above 10% limit |
| GER40.cash | 10.6% | Just above 10% limit |
| XAU_USD | 10.1% | Just above 10% limit |
| GOOG | 8.4% | FTMO-OK but excluded by correlation group (us_tech full) |
| NVDA | 8.9% | FTMO-OK but excluded by correlation group (us_tech full) |
| META | 9.2% | FTMO-OK but excluded by correlation group (us_tech full) |
