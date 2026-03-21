# Daily Trading Report — Sunday 2026-03-01

## Summary

| Account | Trades | Gross P&L | Commission | Net P&L | Win Rate |
|---------|--------|-----------|------------|---------|----------|
| FTMO    | 4      | +$111.39  | -$33.68    | +$77.71 | 75% (3W/1L) |
| BF      | 5      | -$1,031.77| $0.00      | -$1,031.77 | 0% (0W/5L) |
| **Combined** | **9** | **-$920.38** | **-$33.68** | **-$954.06** | **33%** |

**Market context**: Weekend crypto session — low liquidity, BTC whipsawing between $65,700–$67,100. Classic weekend chop with multiple false breakouts.

---

## FTMO Trades (port 5056)

Starting balance: ~$95,595 | Ending balance: ~$95,672 | **Net: +$77.71**

### Trade 1 — BTC SHORT ✅
| Field | Value |
|-------|-------|
| Entry | 10:00 UTC @ $67,106.38 |
| Exit  | 15:51 UTC @ $66,533.41 |
| Volume | 0.35 lot |
| Hold time | 5h 51m |
| Confidence | 0.841 |
| Gross P&L | **+$200.54** |
| Commission | -$15.20 |
| Net P&L | **+$185.34** |

Best trade of the day. Model correctly identified BTC top, trailed stop locked profit on the way down.

### Trade 2 — BTC LONG ✅
| Field | Value |
|-------|-------|
| Entry | 16:00 UTC @ $66,950.38 |
| Exit  | 16:09 UTC @ $67,035.94 |
| Volume | 0.39 lot |
| Hold time | 9 min |
| Confidence | 0.873 |
| Gross P&L | **+$33.37** |
| Commission | -$16.99 |
| Net P&L | **+$16.38** |

Quick scalp — trailing stop moved to breakeven fast, caught a small bounce.

### Trade 3 — BTC LONG ✅
| Field | Value |
|-------|-------|
| Entry | 19:00 UTC @ $66,118.09 |
| Exit  | 20:32 UTC @ $66,280.96 |
| Volume | 0.42 lot |
| Hold time | 1h 32m |
| Confidence | 0.894 |
| Gross P&L | **+$68.41** |
| Commission | -$18.08 |
| Net P&L | **+$50.33** |

Model caught the local bottom with high confidence. Trailing stop locked modest profit.

### Trade 4 — BTC LONG ❌
| Field | Value |
|-------|-------|
| Entry | 21:00 UTC @ $66,300.57 |
| Exit  | 21:44 UTC @ $65,823.24 |
| Volume | 0.40 lot |
| Hold time | 44 min |
| Confidence | 0.719 |
| Gross P&L | **-$190.93** |
| Commission | -$17.18 |
| Net P&L | **-$208.11** |

Re-entry after trade 3 with lower confidence (0.719 vs 0.894). BTC dumped through SL. This was the weakest signal of the day — lesson: lower confidence + weekend = higher risk.

---

## BrightFunded Trades (port 5057)

Starting balance: ~$99,603 | Ending balance: ~$98,571 | **Net: -$1,031.77**

### Trade 1 — BTC LONG ❌
| Field | Value |
|-------|-------|
| Entry | 16:00 UTC @ $66,885.84 |
| Exit  | 16:47 UTC @ $66,487.21 |
| Volume | 0.46 lot |
| Hold time | 47 min |
| Confidence | 0.742 |
| Gross P&L | **-$183.37** |
| Commission | -$7.38 |
| Net P&L | **-$190.75** |

Same signal window as FTMO trade 2, but BF entered at a **higher price** ($66,886 vs $66,950 on FTMO). Different OHLCV data from different brokers = different entry prices. While FTMO caught a quick +$33 bounce, BF entered late and got stopped out.

### Trade 2 — BTC LONG ❌
| Field | Value |
|-------|-------|
| Entry | 17:00 UTC @ $66,169.19 |
| Exit  | 17:17 UTC @ $65,709.61 |
| Volume | 0.45 lot |
| Hold time | 17 min |
| Confidence | 0.854 |
| Gross P&L | **-$206.81** |
| Commission | -$7.15 |
| Net P&L | **-$213.96** |

High confidence signal but BTC continued dumping. SL hit within 17 minutes. BF's wider spreads meant the effective SL was tighter before the spread compensation fix.

### Trade 3 — BTC LONG ❌
| Field | Value |
|-------|-------|
| Entry | 19:00 UTC @ $66,338.34 |
| Exit  | 19:44 UTC @ $65,858.72 |
| Volume | 0.42 lot |
| Hold time | 44 min |
| Confidence | 0.837 |
| Gross P&L | **-$201.44** |
| Commission | -$6.69 |
| Net P&L | **-$208.13** |

Same window as FTMO trade 3 (+$68.41 on FTMO). BF entered at $66,338 while FTMO entered at $66,118 — a $220 difference in entry price. BF's worse fill meant the trailing stop couldn't protect the position before the whipsaw reversed.

### Trade 4 — DOGE LONG ❌ (split fill)
| Field | Value |
|-------|-------|
| Entry | 19:00 UTC @ $0.09315 / $0.09310 |
| Exit  | 20:35 UTC @ $0.09000 |
| Volume | 100,000 + 40,372 units |
| Hold time | 1h 35m |
| Confidence | 0.704 |
| Gross P&L | **-$315.00 + -$125.15 = -$440.15** |
| Commission | -$3.14 |
| Net P&L | **-$443.29** |

DOGE followed BTC's dump. Split order fill (100k + 40.4k). Spread was 16% of SL distance on DOGE — the spread compensation fix was deployed AFTER this trade, so it didn't benefit from the wider SL buffer.

---

## Key Observations

### 1. FTMO vs BF Divergence
Same model, same signals — opposite results. FTMO: +$77.71, BF: -$1,031.77. Root causes:
- **Different broker price feeds**: BF consistently enters at worse prices ($100-$220 worse on BTC)
- **Wider spreads on BF**: Eats into SL distance, causing premature stops
- **Risk scale**: BF uses 0.8x vs FTMO 1.0x, but the spread difference more than negates this

### 2. Weekend Crypto Whipsaw
7 of 9 trades were BTC, all during low-volume weekend hours. The market chopped between $65,700–$67,100 with no clear trend. Mean-reversion signals (high z-scores from extreme moves) generated high confidence values (0.72–0.89), but the follow-through was insufficient for profitable exits.

### 3. Spread Compensation Fix (deployed mid-day)
The `sl_distance += spread_abs` fix in `order_router.py` was deployed during this session. BF's first 3 trades did NOT benefit from this fix. Future trades will have wider SL buffers that automatically reduce position size to maintain the same dollar risk.

### 4. DOGE Amplified Losses
DOGE had 16% of SL distance consumed by spread alone. Combined with weekend low liquidity, this produced the single largest loss of the day (-$440.15). DOGE on BF during weekends is particularly risky due to wide spreads.

### 5. Confidence vs Outcome
| Confidence | Trades | Win Rate |
|------------|--------|----------|
| > 0.80     | 5      | 60% (3W/2L) |
| 0.70-0.80  | 4      | 0% (0W/4L) |

Higher confidence signals performed significantly better, even on a chaotic weekend. The profit gate (min 0.80 after realizing +1%) would have filtered 4 of the 5 losing trades.

---

## Bug Fixes Deployed During Session

1. **Weekend schedule fix** (`bf_sessions.csv` + `ftmo_guard.py`): Crypto was marked `closed` on weekends in BF session override. Fixed CSV + parser to read Saturday/Sunday columns. Without this fix, BF would never have scanned BTC on Sunday.

2. **Spread compensation** (`order_router.py`): Added `spread_abs = abs(tick.ask - tick.bid)` to `sl_distance` before position sizing. This widens the SL by the spread amount while reducing volume proportionally, maintaining the same dollar risk per trade.

3. **Trailing SL notifications** (`position_manager.py`): Removed 0.5×ATR throttle filter. Now sends Discord notification for every trailing SL move.

---

## Account Status (end of day)

| Metric | FTMO | BF |
|--------|------|-----|
| Balance | $95,672 | $98,571 |
| Drawdown from $100k | -$4,328 (4.3%) | -$1,429 (1.4%) |
| Max allowed DD | 10% ($10,000) | 10% ($10,000) |
| Internal warning at | 8% ($8,000) | 8% ($8,000) |
| Headroom | $5,672 | $8,571 |

Note: FTMO's larger drawdown is mostly from Feb 26-27 (-$4,358 in 2 days), not from Sunday. BF had been profitable before Sunday (-$252 net Feb 23-28, then -$1,032 on Sunday).

---

## Recommendations (post-lock period)

1. **Weekend risk reduction**: Consider halving `risk_scale` during weekend sessions (Sat/Sun) for crypto pairs. Low liquidity + wide spreads = adverse conditions.
2. **DOGE on BF**: Monitor spread-to-SL ratio. If consistently >10%, consider excluding DOGE from BF weekend scanning.
3. **Confidence floor**: The 0.80 profit gate already helps. Consider raising minimum confidence for weekend entries from default to 0.80 across the board.
4. **Spread compensation validation**: Now that the fix is live, compare BF SL-hit rates before/after over the next 2 weeks.
