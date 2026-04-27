# BTC/USDT M15 Hybrid Regime-Switching Strategy

Pine Script v6 implementation. Two files:

- `pine/btc_regime_hybrid_indicator.pine` — visualization + alerts only
- `pine/btc_regime_hybrid_strategy.pine` — full backtest with martingale, partial TP, daily DD lock

Historical data for backtest is in `BTCUSDT-15m-YYYY-MM.csv` (Binance Futures, 2021-01 → 2026-03).

---

## 1. Edge & rationale

A single fixed logic loses on BTC M15 because the asset alternates between two structurally different regimes:

- **Trending sessions** (US open, news-driven hours, CPI/FOMC days): momentum persists for hours, mean reversion bleeds.
- **Asian / overnight ranges** (≈ 22:00–06:00 UTC) and post-impulse consolidations: oscillation around VWAP, mean reversion works, breakouts fail.

The edge is **not predicting direction** — it is **picking the right tool for the current regime**. The exploited inefficiencies:

1. **Volatility clustering** (GARCH-style autocorrelation in absolute returns) → ATR / BB-width are predictable enough to classify regime one bar ahead.
2. **Intraday liquidity cycle** — order-flow imbalance during US/EU sessions creates persistent moves; thin-book Asian hours mean-revert.
3. **Behavioral overshoots** at BB±2σ in flat regimes — short-term liquidity hunts that revert within 1–6 bars.
4. **HTF trend bias** — when H1/H4 EMA200 slope is strong, M15 counter-trend mean reversion has negative expectancy; aligning M15 trend entries with HTF removes this.

Hybrid switching means we accept lower trade frequency for higher per-trade EV: instead of forcing entries, we *only trade when the regime favors a specific playbook*.

---

## 2. Detailed rules

### 2.1 Regime detector (runs every bar, classification used on next bar)

```text
adx, +DI, -DI = ta.dmi(14, 14)
ema50, ema200  = ta.ema(close, 50), ta.ema(close, 200)
bbWidth        = (bbUpper - bbLower) / bbBasis      // bbLen=20, stdev=2
slope50        = ta.linreg(ema50, 20, 0) - ta.linreg(ema50, 20, 5)

regime = TREND_UP   if adx > 25 and close > ema200 and slope50 > 0 and bbWidth > bbWidth_p40
       = TREND_DOWN if adx > 25 and close < ema200 and slope50 < 0 and bbWidth > bbWidth_p40
       = RANGE      if adx < 20 and bbWidth < bbWidth_p60
       = NEUTRAL    otherwise (no trades)
```

Classification only on **closed bars** (`barstate.isconfirmed`) — no repaint. HTF EMA200 is fetched with `lookahead = barmerge.lookahead_off`.

### 2.2 Trend regime — pullback entries

- **Long** (regime == TREND_UP):
  - Pullback condition: `low ≤ ema20` within last 3 bars
  - Reclaim: current `close > ema20`
  - Momentum: `RSI(14) ∈ (50, 70)` (avoid chasing exhaustion)
  - HTF align: H1 close > H1 EMA200
- **Short**: symmetric (TREND_DOWN, RSI ∈ (30, 50), H1 below EMA200).
- **SL**: `entry − 1.0 × ATR(14)` for long, with floor at last 5-bar swing low − 0.2×ATR buffer.
- **TP1**: `entry + 1.0 × ATR` → close 50% (move SL to entry).
- **TP2**: `entry + 2.0 × ATR` or trailing-stop on 1×ATR Chandelier from peak.

### 2.3 Range regime — mean reversion

- **Long**: regime == RANGE, `close ≤ bbLower`, `RSI(14) < 30`, prior bar low was the lowest of last 10.
- **Short**: symmetric at `bbUpper`, `RSI > 70`.
- **SL**: `entry − 0.8 × ATR` (tighter — flat regime, smaller expected adverse).
- **TP**: `bbBasis` (mid-band). Typical R/R ≈ 1:1 to 1:1.3. No partial — single TP.

### 2.4 Filters (all must pass)

| Filter | Rule |
|---|---|
| HTF trend (trend regime only) | M15 trend direction must match H1 EMA200 side |
| Time-of-day | Block 02:00–06:00 UTC (low-liquidity Asian dead zone) |
| Volatility floor | `ATR(14) > p20(ATR over 100 bars)` — skip dead market |
| Volatility ceiling | `ATR(14) < p95(ATR over 100 bars)` — skip flash-crash spikes |
| Funding (optional, manual input) | Block longs if est. funding > +0.05%; block shorts if < −0.05% |
| News window | User-configured session input: ±15 min around scheduled events |
| One position at a time | `pyramiding = 0`, no stacking |

### 2.5 Risk management & martingale

| Layer | Rule |
|---|---|
| Base risk | 3% of current equity per trade |
| Martingale | After a loss: next trade risk × 2 (steps cap = 2 → max 12%) |
| Reset | On any win, or after `martingale_step == max` |
| Streak stop | After 4 consecutive losses → pause trading 24h |
| Daily DD lock | If equity drops ≥ 15% from day-open → flat & no new entries until 00:00 UTC |
| Position sizing | `qty = (equity × risk_pct) / SL_distance`, capped by `qty × price ≤ equity × max_leverage` |
| Min lot | `qty = max(qty, 0.001 BTC)` (Binance/Bybit min) |
| Max leverage | 10x hard cap (configurable up to 20x) |
| Commission | 0.055% per side (Bybit/Binance taker), modeled in `strategy(...)` |
| Slippage | 3 ticks |

Important: **2 martingale steps × 2× = 12% on the third trade.** Combined with the 15% daily DD lock, a 3-loss streak (3% + 6% + 12% = 21%) will trip the daily lock during the third position, which is intentional — this caps single-day catastrophe at ~15–18%.

---

## 3. Parameters table (defaults & optimization ranges)

| Group | Parameter | Default | Range | Notes |
|---|---|---|---|---|
| Regime | ADX length | 14 | 10–20 | |
| Regime | ADX trend threshold | 25 | 20–30 | Higher = fewer but cleaner trends |
| Regime | ADX range threshold | 20 | 15–22 | |
| Regime | EMA fast | 50 | 34–89 | |
| Regime | EMA slow | 200 | 144–250 | |
| Regime | BB length | 20 | 14–30 | |
| Regime | BB stdev | 2.0 | 1.8–2.5 | |
| Regime | Slope lookback | 20 | 10–30 | |
| Trend entry | EMA pullback period | 20 | 13–34 | |
| Trend entry | RSI period | 14 | 10–21 | |
| Trend entry | RSI long band | (50, 70) | fixed | |
| Range entry | BB extreme touches | required | — | |
| Range entry | RSI extreme | 30 / 70 | 25–35 / 65–75 | |
| Exits | ATR length | 14 | 10–21 | |
| Exits | SL ATR mult (trend) | 1.0 | 0.8–1.5 | |
| Exits | TP1 ATR mult (trend) | 1.0 | 0.8–1.5 | |
| Exits | TP2 ATR mult (trend) | 2.0 | 1.5–3.0 | |
| Exits | SL ATR mult (range) | 0.8 | 0.6–1.2 | |
| Exits | Trailing ATR mult | 1.0 | 0.5–1.5 | |
| Risk | Base risk % | 3.0 | 1.0–5.0 | |
| Risk | Martingale × | 2.0 | 1.5–2.5 | |
| Risk | Max martingale steps | 2 | 0–3 | 0 disables martingale |
| Risk | Max consecutive losses | 4 | 3–6 | |
| Risk | Daily DD limit % | 15 | 5–25 | |
| Risk | Max leverage | 10 | 5–20 | |
| Filters | HTF timeframe | 60 | 60 / 240 | |
| Filters | Block session | 02:00–06:00 UTC | configurable | |
| Filters | ATR percentile floor | 20 | 10–30 | |
| Filters | ATR percentile ceiling | 95 | 90–99 | |

---

## 4. Expected metrics (target, not guarantee)

Based on prior published research on regime-switching crypto systems (BTC M15, 2020–2024) and conservative assumptions:

| Metric | Trend mode | Range mode | Combined target |
|---|---|---|---|
| Win Rate | 38–45% | 55–62% | ~50% |
| Avg R/R | 1.7 : 1 | 1.0 : 1 | ~1.4 : 1 |
| Profit Factor | 1.4–1.7 | 1.2–1.4 | **> 1.4** target |
| Expectancy / trade | +0.25R | +0.15R | +0.20R |
| Trades / day | 1–3 | 1–3 | 2–6 |
| Max DD (no martingale) | — | — | 25–30% |
| Max DD (martingale ×2, 2 steps) | — | — | 40–55% (realistic) |
| Sharpe (annualized) | — | — | 1.0–1.6 |
| Sortino | — | — | 1.5–2.2 |

These are **plausible** numbers, not promises. Real out-of-sample results often degrade 30–50% versus in-sample.

---

## 5. Backtest plan

1. **Data**: `BTCUSDT-15m-2021-01.csv` … `BTCUSDT-15m-2026-03.csv` (~5 years).
2. **Splits**:
   - Train / param search: 2021-01 → 2023-06 (bull → bear → recovery)
   - Validation: 2023-07 → 2024-12 (bull + ATH)
   - Out-of-sample: 2025-01 → 2026-03 (post-cycle)
3. **Metrics to record per split**: PF, WR, expectancy in R, max DD, MAR ratio, longest losing streak, equity curve smoothness (R² of log-equity vs time).
4. **Walk-forward**: 6 anchors, 90-day train / 30-day test, no re-optimization inside test.
5. **Stress tests**:
   - Commission ×1.5 and slippage ×2
   - Funding rate as additional 0.01%/8h drag on holds > 8h
   - Random ±25% perturbation on every parameter — strategy must keep PF > 1.1 (robustness)
6. **Pass criteria** (combined OOS): PF > 1.3, max DD < 50%, no parameter collapses PF below 1.0 in stress.

---

## 6. Honest assessment of weaknesses

### 6.1 Where the strategy bleeds

- **Regime transitions** — first 1–3 bars of a regime change are misclassified; trend entries get stopped near tops, range entries trapped in breakouts. Expect a cluster of small losses 2–4 times per week.
- **Flash crashes / liquidations** — ATR ceiling filter catches most, but one-bar wicks (e.g., Aug 2024 yen-carry unwind) can blow through SL with slippage > 50 ticks. Real loss can be 1.5–2× planned.
- **Low-volume drift weekends** — even with session filter, weekend ranges sometimes morph into slow trends that the regime detector flags as RANGE → mean reversion entries fight a slow trend.
- **Funding rate drag** on positions held overnight — at +0.1%/8h on overcrowded longs, holding two funding windows costs 0.2% which is meaningful at 3% risk.
- **Self-similarity decay** — BTC M15 microstructure changes (more derivatives flow, more index-arb). Parameters that worked in 2021–2022 are weaker in 2024–2026.

### 6.2 The martingale math at $200 (this is the part that matters)

Assume per-trade win probability `p = 0.50`, max martingale steps = 2 (×1, ×2, ×4 risk), base risk 3%.

A "martingale chain" loses fully only if 3 consecutive losses occur. With independent trades:

- P(3 losses in a row) = (1 − p)³ = **0.125** at p=0.5 → **12.5% per chain**
- Cumulative loss in failed chain = 3% + 6% + 12% = **21%** of equity (clipped to ~15% by daily DD lock)
- Expected chains per 100 trades: ~33
- Expected chain failures per 100 trades: ~4

**Probability of at least one ruin event (50% drawdown) in 100 trades**: not catastrophically high if daily DD lock works as designed (~10–15%), but still meaningful. **Over 500 trades (~3 months at 6/day), that probability rises to 35–50%.**

If `p` is overestimated and real WR is 0.45:
- P(3-loss streak) = 0.166 → ~17% chains fail → expected ~5–6 chain failures per 100 trades.
- Compounding losses degrade equity faster than wins recover (asymmetry of percent moves).

**Conclusion**: at $200 with these settings the most likely outcome distribution is bimodal — either 2–3× growth in 1–2 months OR account reduced below $50 within 2 months. Median outcome is closer to "down 30–60%" than to "up 200%".

### 6.3 Friction at $200 deposit

- Per-trade round-trip cost = 0.11% commission + ~0.04% slippage = **0.15%**.
- At avg trade move of 0.4×ATR ≈ 0.6% on TP1, friction eats **~25%** of gross win.
- Funding can add another 0.1–0.3% on overnight holds.
- **Net effect**: a strategy that backtests at PF 1.6 gross may print PF 1.25–1.35 net. Marginally profitable.

### 6.4 Recommendation

If the goal is to actually grow $200 to $600–$1000 in 1–3 months, this strategy is **plausible but not likely** — math says expected value is positive but variance is high enough that median user blows up before reaching the goal. Suggested mitigations, in order of impact:

1. **Disable martingale entirely** (`max_martingale_steps = 0`). Keeps PF roughly the same, cuts max DD from ~50% to ~25%, removes ruin risk.
2. **Drop base risk to 1.5%**. Halves variance; growth slows but survival probability rises sharply.
3. **Paper-trade for 4 weeks first** to verify the regime detector classifies your live broker's price feed the same way the backtest does (different exchanges have different microstructure noise).
4. **Increase deposit to $1000+** if martingale must stay — same percent rules, but min-lot constraint disappears and you can use partial fills more flexibly.

The script implements all settings; the defaults match the prompt's aggressive profile. Tune via the inputs.
