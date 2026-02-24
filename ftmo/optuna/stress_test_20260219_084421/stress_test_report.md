# Extreme Stress Test Report

**Generated:** 2026-02-19 08:48:50

---

## Phase 1: Monte Carlo Extreme

- **extreme_500k**: ruin=100.00%, median_sharpe=-11.07, engine=CPU, time=24.9s
- **deep_2k_trades**: ruin=100.00%, median_sharpe=-11.07, engine=CPU, time=20.0s
- **million_sims**: ruin=100.00%, median_sharpe=-11.04, engine=CPU, time=25.3s
- **gtx1050_50k**: ruin=100.00%, median_sharpe=-11.04, engine=CPU, time=1.3s
- **gtx1050_100k**: ruin=100.00%, median_sharpe=-11.04, engine=CPU, time=2.6s

## Phase 2: Scenario Injection

| Symbol | Scenario | Baseline EV | Stress EV | Change% | Max Loss | Consec Losses |
|--------|----------|-------------|-----------|---------|----------|---------------|
| AMZN | flash_crash_8pct | 0.0013 | 0.0013 | +0.0% | -0.0264 | 7 |
| AMZN | flash_crash_15pct | 0.0013 | 0.0013 | +0.0% | -0.0264 | 7 |
| AMZN | gap_up_5pct | 0.0013 | 0.0011 | -16.6% | -0.0252 | 11 |
| AMZN | gap_down_5pct | 0.0013 | 0.0011 | -18.6% | -0.0277 | 7 |
| AMZN | vol_explosion_5x | 0.0013 | 0.0013 | -5.6% | -0.0264 | 7 |
| AMZN | vol_explosion_10x | 0.0013 | 0.0013 | +0.0% | -0.0264 | 7 |
| AMZN | spread_blowout_10x | 0.0013 | 0.0027 | +101.6% | -0.0893 | 7 |
| AMZN | spread_blowout_50x | 0.0013 | 0.0028 | +108.4% | -0.0816 | 7 |
| AMZN | slippage_20x | 0.0013 | 0.0011 | -17.7% | -0.0264 | 9 |
| AMZN | dead_market | 0.0013 | 0.0013 | -0.5% | -0.0264 | 7 |
| AMZN | whipsaw | 0.0013 | 0.0024 | +78.2% | -0.0257 | 7 |
| TSLA | flash_crash_8pct | 0.0024 | 0.0024 | +0.0% | -0.0440 | 11 |
| TSLA | flash_crash_15pct | 0.0024 | 0.0024 | +0.0% | -0.0440 | 11 |
| TSLA | gap_up_5pct | 0.0024 | 0.0027 | +13.5% | -0.0420 | 9 |
| TSLA | gap_down_5pct | 0.0024 | 0.0027 | +15.0% | -0.0463 | 11 |
| TSLA | vol_explosion_5x | 0.0024 | 0.0024 | +0.0% | -0.0440 | 11 |
| TSLA | vol_explosion_10x | 0.0024 | 0.0024 | +0.0% | -0.0440 | 11 |
| TSLA | spread_blowout_10x | 0.0024 | -0.0078 | -426.9% | -0.2113 | 6 |
| TSLA | spread_blowout_50x | 0.0024 | 0.0105 | +340.6% | -0.2234 | 7 |
| TSLA | slippage_20x | 0.0024 | 0.0027 | +12.1% | -0.0440 | 11 |
| TSLA | dead_market | 0.0024 | 0.0023 | -4.4% | -0.0440 | 11 |
| TSLA | whipsaw | 0.0024 | 0.0025 | +4.4% | -0.0453 | 9 |
| JP225.cash | flash_crash_8pct | -0.0007 | -0.0007 | +0.0% | -0.0319 | 19 |
| JP225.cash | flash_crash_15pct | -0.0007 | -0.0007 | +0.0% | -0.0319 | 19 |
| JP225.cash | gap_up_5pct | -0.0007 | -0.0007 | +6.5% | -0.0305 | 19 |
| JP225.cash | gap_down_5pct | -0.0007 | -0.0007 | -2.6% | -0.0336 | 19 |
| JP225.cash | vol_explosion_5x | -0.0007 | -0.0007 | -0.0% | -0.0319 | 19 |
| JP225.cash | vol_explosion_10x | -0.0007 | -0.0007 | -0.0% | -0.0319 | 19 |
| JP225.cash | spread_blowout_10x | -0.0007 | -0.0024 | -242.2% | -0.0557 | 8 |
| JP225.cash | spread_blowout_50x | -0.0007 | -0.0025 | -265.5% | -0.0557 | 8 |

## Phase 3: Backtest Blast (28 workers)

- **Completed:** 0/28
- **Failed:** AMZN, TSLA, JP225.cash, LVMH, NVDA, PFE, RACE, BTCUSD, ETHUSD, DASHUSD, XAU_USD, USOIL.cash, EUR_USD, USD_JPY, GBP_USD, AUD_JPY, US100.cash, GER40.cash, AAPL, META, MSFT, GOOG, BABA, XAG_USD, UK100.cash, HK50.cash, EU50.cash, US30.cash

## Phase 4: FTMO Guardrail Torture

**Result: 13/14 tests passed**

- [PASS] daily_loss_limit_36pct: equity -3.6% of balance
- [PASS] daily_loss_limit_34pct: equity -3.4% of balance (should pass)
- [PASS] profit_lock_31pct: equity +3.1% — should lock profits
- [PASS] dd_recovery_mode_45pct: DD 4.5% — should enter recovery (halve lots)
- [PASS] dd_recovery_exit: DD 0.5% after recovery — should exit recovery
- [PASS] null_account: None account_info — should allow
- [PASS] lot_calc_amzn: AMZN lot calc: equity=100k risk=0.25% SL=3.0
- [PASS] lot_calc_zero_sl: SL distance = 0 — should return 0
- [PASS] kelly_no_mt5: Kelly without MT5 — should return fallback 0.3%
- [PASS] kelly_zero_wr: fractional_kelly(WR=0.0, PF=1.5, frac=0.5)
- [PASS] kelly_perfect_wr: fractional_kelly(WR=1.0, PF=2.0, frac=0.5)
- [PASS] kelly_bad_pf: fractional_kelly(WR=0.5, PF=0.0, frac=0.5)
- [PASS] kelly_normal: fractional_kelly(WR=0.55, PF=1.8, frac=0.5)
- [**FAIL**] rl_adjust_no_sizer: RL adjust without RL sizer — should return base risk

---

## System Metrics

See `system_metrics.csv` for per-10s CPU/GPU/RAM data.
