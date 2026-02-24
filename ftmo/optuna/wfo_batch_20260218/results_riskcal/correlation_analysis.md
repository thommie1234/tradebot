# Correlatie-analyse WFO Portfolio (13 symbolen)

## Correlatie-groepen (WFO selector)

| Groep | Symbolen | Max |
|-------|----------|-----|
| us_tech | AMZN, MSFT | 2/2 |
| eu_equity | LVMH | 1/2 |
| us_idx | US100, US30 | 2/2 |
| apac_idx | AUS200, JP225 | 2/2 |
| jpy_fx | CAD_JPY | 1/2 |
| usd_fx | NZD_USD | 1/2 |
| silver | XAG_AUD | 1/2 |
| oil | UKOIL | 1/2 |

## Werkelijke correlatie-risico

De selector gebruikt smalle groepen, maar in de praktijk:

- **AMZN + MSFT + BABA + US100 + US30** = 5 symbolen die sterk bewegen met S&P 500
- **US100 en US30** zijn ~0.95 gecorreleerd
- **LVMH** beweegt deels mee met US risk-on/off sentiment
- ~6 van 13 symbolen effectief "long US risk"

## Echte diversificatie

- JP225 / AUS200 — Aziatische sessie, eigen dynamiek
- CAD_JPY / NZD_USD — forex, carry trades
- BCHUSD — crypto, laag gecorreleerd met equities
- XAG_AUD / UKOIL — commodities

## Risico bij US sell-off

Bij -5% US markt: ~5 symbolen tegelijk in drawdown.
Gecombineerde US-blootstelling: ~3.5% risk op een "factor".

## TODO

- [ ] Overweeg bredere "US_all" correlatie-groep
- [ ] Of max-corr strenger zetten (max 1 per groep)
- [ ] Of US-blootstelling cappen op max 3 symbolen totaal
