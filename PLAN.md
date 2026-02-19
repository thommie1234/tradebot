# Multi-Timeframe Feature Expansion — Implementatieplan

## Doel
Higher-timeframe context (H4, D1) toevoegen als extra features aan het ML model.
Voorbeeld: het H1 model ziet nu alleen H1 bars. Met HTF features weet het ook of
de D1 trend bullish is, of H4 volatiliteit stijgt, etc.

## Welke features per HTF (8 per timeframe)
Van elke hogere timeframe (H4 + D1) nemen we 8 samengevatte features:

| Feature          | Beschrijving                              |
|------------------|-------------------------------------------|
| `htf_ret1`       | 1-bar return (richting)                   |
| `htf_ret3`       | 3-bar return (korte momentum)             |
| `htf_ma_cross`   | MA5/MA20 ratio (trend)                    |
| `htf_vol20`      | 20-bar volatiliteit                       |
| `htf_atr_ratio`  | ATR5/ATR20 ratio (vol expansion/contract) |
| `htf_z20`        | Z-score (mean-reversion druk)             |
| `htf_adx_proxy`  | ADX proxy (trendsterkte)                  |
| `htf_regime`     | Regime label (0/1/2)                      |

Prefix: `h4_` en `d1_` → 16 nieuwe features, totaal 39 + 16 = 55.

## Bestanden die wijzigen

### 1. `engine/feature_builder.py`
- Nieuwe functie `build_htf_features(htf_bars, prefix)` → berekent 8 features
- Nieuwe functie `merge_htf_features(base_feat, htf_feat, prefix)` → asof-join op time
- `FEATURE_COLUMNS` uitbreiden met 16 HTF kolommen
- Shift(1) discipline: HTF features worden shift(1) toegepast in `build_htf_features`

### 2. `engine/signal.py` — `get_h1_features()`
- Na H1 bars ophalen, ook H4 + D1 bars ophalen via MT5
- `build_htf_features()` + `merge_htf_features()` aanroepen
- Fallback: als HTF data niet beschikbaar → fill met 0.0

### 3. `engine/multi_tf_scanner.py` — `_get_features()`
- Zelfde HTF enrichment als signal.py

### 4. `engine/inference.py` — `train_model()`
- Bij training: H4 + D1 bar data laden van parquets (bar_roots)
- `merge_htf_features()` toevoegen aan feature pipeline
- Model traint nu op 55 features

### 5. `research/optuna_orchestrator.py` — `run_symbol()`
- HTF bar data laden naast primary timeframe
- Merge HTF features voor Optuna trials
- `FEATURE_COLUMNS` import automatisch bijgewerkt

### 6. `research/train_ml_strategy.py`
- Zelfde HTF merge toevoegen aan WFO training

### 7. `research/backtest_engine.py`
- HTF features toevoegen aan backtest pipeline

## Data flow

```
Bar parquets (ssd_data_2/bars/)
  ├── H1/{SYMBOL}/*.parquet  →  build_bar_features()     → 39 base features
  ├── H4/{SYMBOL}/*.parquet  →  build_htf_features("h4") → 8 features
  └── D1/{SYMBOL}/*.parquet  →  build_htf_features("d1") → 8 features
                                        ↓
                              merge_htf_features() via asof join on time
                                        ↓
                              55-feature DataFrame → XGBoost
```

## Asof join logica
HTF bars hebben grotere tijdsintervallen. Voor elke H1 bar zoeken we de
meest recente H4/D1 bar die VOOR die H1 bar sloot (backward asof join).
Dit is inherent leak-safe: we gebruiken alleen HTF bars die al afgesloten zijn.

## Implementatievolgorde

1. **`feature_builder.py`** — `build_htf_features()` + `merge_htf_features()` + FEATURE_COLUMNS
2. **`research/optuna_orchestrator.py`** — HTF bar loading + merge in Optuna pipeline
3. **`research/train_ml_strategy.py`** — HTF merge in WFO training
4. **`engine/inference.py`** — HTF merge in live training
5. **`engine/signal.py`** — HTF bars ophalen bij live inference
6. **`engine/multi_tf_scanner.py`** — HTF enrichment
7. **Backtest** — run op alle 116 symbolen, vergelijk 39-feat vs 55-feat resultaten

## Backtest strategie (voordat het live gaat)
- Eerst Optuna draaien op actieve symbolen met 55 features
- Walk-forward backtest vergelijken: oud (39 feat) vs nieuw (55 feat)
- Alleen live zetten als EV verbetert en geen overfitting signalen

## Risico's
- **Overfitting**: 16 extra features bij weinig data → XGBoost regularisatie + Optuna tuning vangt dit op
- **Correlatie**: HTF features zijn gecorreleerd met base features (bijv. h4_ret1 ≈ ret12) → colsample_bytree handelt dit af
- **Missing data**: Sommige symbolen hebben geen 20 jaar D1 data → graceful fallback naar 0.0
