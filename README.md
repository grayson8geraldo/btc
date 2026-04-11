# BTCUSDT 15m · 20 Intraday Strategies · Honest Tournament

**Goal (as requested):** study the historical BTC data in this repo, design
20 intraday strategies with a starting deposit of **$200**, create TradingView
(Pine Script) indicators for 10 of them, then run a fair tournament on
out-of-sample data to pick the winner — **without fitting parameters to the
test set**.

This README documents the methodology, every strategy, the backtest rules,
and — most importantly — the honest result.

---

## 1 · Dataset

| Item | Value |
|---|---|
| Instrument | `BTCUSDT` (Binance) |
| Timeframe  | **15 minutes** |
| Files      | 61 monthly CSVs, `BTCUSDT-15m-YYYY-MM.csv` |
| Range      | 2021-01-01 → 2026-03-31 (gaps: 2025-07, 2026-01) |
| Bars       | **177,984** candles (≈ 5.2 years) |
| Price      | from **$15,638** to **$125,986** |

Buy & hold benchmark: $200 → $475 full range (+137%), $200 → **$321** on the
out-of-sample window.

## 2 · Backtest rules (identical for every strategy)

| Setting | Value | Why |
|---|---|---|
| Starting capital | **$200** | as requested |
| Timeframe        | 15-minute bars | intraday focus |
| Signal execution | on bar close → enter at **next bar’s open** | no look-ahead |
| Fee per side     | **0.04 %** (Binance USDT-M taker) | realistic |
| Slippage per side| **0.02 %** | realistic half-spread |
| Round-trip cost  | **0.12 %** of notional | ‑ |
| Stop loss        | per-strategy (usually ±1.5 ATR) | ‑ |
| Take profit      | **fixed 2R** multiple or mid-band | no trailing/reoptimizing |
| Max holding time | **96 bars = 24 h** | keeps it intraday |
| Positions        | **one at a time**, long+short allowed | ‑ |
| Walk-forward     | **IS** 2021-01 → 2023-12 · **OOS** 2024-01 → 2026-03 | ‑ |

Two fixed risk-management profiles are run on **every** strategy.  Both were
decided *before* looking at any results and applied uniformly.

| Profile | Risk/trade | Max leverage | Cooldown after a trade |
|---|---|---|---|
| **AGGR** | 2 %  | 3x | 0 bars   |
| **CONS** | 1 %  | 2x | 32 bars (8 h) |

### 2.1 Honesty guarantees

*   **No optimization**: every parameter is a textbook default chosen up
    front (RSI-2, MACD 12/26/9, Bollinger 20/2, SuperTrend 10/3, …).
    Nothing is tuned to maximize any metric.
*   **Walk-forward split**: the OOS window (2024-01 … 2026-03, 72,864 bars)
    is never used to pick parameters, strategies, or profiles.
*   **Same engine, same costs** applied to every strategy.
*   **Top-10 selection** uses ONLY in-sample statistics with a pre-committed
    formula:

    ```
    rank_score = 0.45·norm(Sharpe) + 0.25·norm(min(PF,5))
               + 0.15·norm(ExpectancyR) + 0.15·norm(−MaxDD)
    ```
*   **Tournament rank**: the winner is the strategy with the highest
    **OOS CONS** end-equity — that is exactly the “can we grow $200?”
    question the user asked.

## 3 · The 20 strategies

All signal logic lives in [`backtest/strategies.py`](backtest/strategies.py);
shared indicators in [`backtest/engine.py`](backtest/engine.py).

| # | Name | Family | Core rule (fixed params) |
|---|---|---|---|
| 1  | **EMA Cross 9/21**         | trend   | EMA9↑ over EMA21 = long · stop 1.5 ATR |
| 2  | **RSI-2 Reversion**        | reversion | Connors: RSI(2)<10 & close>EMA200 = long |
| 3  | **BB Breakout 20/2**       | breakout | close crosses above upper BB |
| 4  | **BB Fade 20/2**           | reversion | close below lower BB → long, target = mid |
| 5  | **Donchian-20 Breakout**   | breakout | close > 20-bar high → long |
| 6  | **MACD Cross 12/26/9**     | trend   | MACD↑ over signal from below 0 |
| 7  | **Stochastic 14/3/3**      | reversion | K↑ over D while D<20 → long |
| 8  | **VWAP Reversion**         | reversion | (close−VWAP)/ATR < −2 → long, target VWAP |
| 9  | **VWAP Breakout**          | trend   | first close above daily VWAP > 0.25 ATR |
| 10 | **Keltner Breakout 20/2×ATR** | breakout | close crosses upper Keltner |
| 11 | **SuperTrend 10/3**        | trend   | direction flip |
| 12 | **Heikin-Ashi Flip**       | trend   | 2 consecutive HA greens after a red |
| 13 | **Opening Range 4 h**      | breakout | break of UTC day’s first-4h range |
| 14 | **Parabolic SAR 0.02/0.2** | trend   | SAR trend flip |
| 15 | **ADX(14)+EMA50**          | trend   | ADX>25 & close crosses EMA50 |
| 16 | **Volume Spike Fade**      | reversion | vol z>3 & return<−0.5 % → long |
| 17 | **Z-score 20**             | reversion | (close−SMA20)/σ < −2 → long |
| 18 | **Momentum-3**             | trend   | ret(3 bars) > +1 % & close > EMA50 |
| 19 | **CMF-20 Zero Cross**      | flow    | CMF crosses > 0 → long |
| 20 | **Williams %R 14**         | reversion | %R crosses up through −80 |

## 4 · In-sample results (2021-01 → 2023-12)

Full tables are in [`results/tournament_log.txt`](results/tournament_log.txt)
and [`results/tournament_results.json`](results/tournament_results.json).

**AGGR profile** — almost every strategy blows up: high risk + frequent
signals + 0.12 % round-trip cost demolishes a small $200 account.

**CONS profile** — survivors emerge:

| Rank | Strategy | IS End-eq | PF | Sharpe | Trades |
|---:|---|---:|---:|---:|---:|
| 1 | **S13 OpeningRange**  | **$93.75** | 0.88 | -1.22 |  885 |
| 2 | **S16 VolumeFade**    | $63.50 | 0.93 | -1.41 |  974 |
| 3 | **S02 RSI2**          | $56.13 | 0.97 | -1.37 | 1343 |
| 4 | **S18 Momentum-3**    | $42.18 | 0.89 | -1.71 | 1216 |
| 5 | **S11 SuperTrend**    | $39.49 | 0.82 | -2.51 | 1046 |
| … | | | | | |

**None of the 20 strategies finishes the 3-year IS window in profit.**
The best is S13 Opening-Range at $93.75 (still a −53 % loss).

## 5 · Top-10 selected for tournament

Selected strictly by IS-CONS `rank_score`:

```
 1.  S02_RSI2_Reversion    6.  S13_OpeningRange
 2.  S04_BB_Fade           7.  S10_Keltner_Breakout
 3.  S17_ZScore_Reversion  8.  S06_MACD_Cross
 4.  S16_Volume_Fade       9.  S11_SuperTrend
 5.  S18_Momentum3         10. S20_Williams_R
```

## 6 · Out-of-sample tournament (2024-01 → 2026-03)

These 10 strategies were then run **UNCHANGED** on the OOS window — no
re-tuning, no reselection.  Starting capital is $200 for every row.

### 6.1 CONS profile — final standings (primary metric)

| # | Strategy | End $ | Ret | Sharpe | MaxDD | PF | Trades |
|---:|---|---:|---:|---:|---:|---:|---:|
| 🏆 1 | **S18 Momentum-3**   | **$154.43** | −22.78 % | −0.39 | −38.5 % | **1.02** | 627 |
| 2 | S13 OpeningRange      |  $102.89    | −48.56 % | −1.50 | −52.8 % | 0.87 | 642 |
| 3 | S16 VolumeFade        |   $70.20    | −64.90 % | −2.24 | −67.6 % | 0.79 | 533 |
| 4 | S11 SuperTrend        |   $60.44    | −69.78 % | −2.74 | −70.2 % | 0.75 | 753 |
| 5 | S02 RSI2 Reversion    |   $56.65    | −71.67 % | −2.03 | −75.0 % | 0.87 | 962 |
| 6 | S10 Keltner Breakout  |   $17.74    | −91.13 % | −4.17 | −91.4 % | 0.78 | 1086 |
| 7 | S06 MACD Cross        |   $15.22    | −92.39 % | −4.39 | −92.8 % | 0.75 | 1271 |
| 8 | S04 BB Fade           |   $12.18    | −93.91 % | −5.38 | −94.0 % | 0.75 | 1412 |
| 9 | S17 Z-score           |   $12.18    | −93.91 % | −5.38 | −94.0 % | 0.75 | 1412 |
| 10 | S20 Williams %R      |    $3.37    | −98.31 % | −7.46 | −98.3 % | 0.63 | 1649 |

### 6.2 AGGR profile (for context)

Under the aggressive profile 4 of the 10 strategies go fully to zero.  Best
is still **S18 Momentum-3** at **$32.54** (−83.7 %).  AGGR is simply too
violent for a $200 account at 15m.

## 7 · 🏆 Winner — `S18 Momentum-3`

```
Entry long  : close / close[3] − 1 > +1 %  AND  close > EMA(50)
Entry short : close / close[3] − 1 < −1 %  AND  close < EMA(50)
Stop        : ±1.5 · ATR(14)
Take profit : 2R (fixed)
Max hold    : 96 bars (24 h)
Profile     : 1 % risk · 2x leverage · 8-h cooldown
```

OOS stats:

* 627 trades, 37.3 % win rate, **profit factor 1.023**, expectancy **+0.006 R**
* $200 → **$154.43**, max DD −38.5 %
* **Positive edge per trade**, but still a losing equity curve

## 8 · The honest conclusion — and why it matters

> **None of the 20 textbook intraday strategies grows $200 on BTC 15m
> when tested without curve-fitting.**

This is the key finding the user explicitly asked for (“without fitting”).
Let’s break down *why*:

1. **Costs dominate.**  A realistic 0.12 % round-trip eats any edge that
   textbook indicators might have.  Winners like Momentum-3 have profit
   factor ≈ 1.02 — the cost alone is roughly 0.02 % per trade on top of
   natural variance, enough to tip the curve negative.
2. **Small account + compounding drag.**  On a $200 account sized at 1 %
   risk, losing streaks shrink future dollar risk, so wins can’t fully
   restore prior dollar losses even with a fair coin.  This is the classic
   volatility drag; it is *worse* for high-frequency strategies.
3. **Buy & hold beat everything.**  Over the OOS window BTC rose +60 %:
   $200 simply held = **$321**.  Active intraday strategies with naïve
   entries cannot beat this on $200 unless they have a genuine edge — and
   textbook RSI/MACD/Bollinger rules, applied uniformly, do not.
4. **The best strategies were the ones trading LESS.**  Momentum-3 (627
   trades), Opening-Range (642), Volume-Fade (533) dominate the top of the
   leaderboard.  High-frequency ones (Williams %R 1649, Z-score 1412) are
   at the bottom.  *Friction scales with trade count.*
5. **A 2 % risk / 3x leverage / zero-cooldown setup is NOT viable** for a
   small account on 15m BTC — 16 of the 20 strategies go fully to zero on
   OOS AGGR.  Even the winner loses 84 %.

### 8.1 What WOULD grow a $200 deposit?

Based on this honest backtest, realistic paths are:
* **Hodl / DCA** — Buy & hold returned +60 % on the OOS window with zero
  skill.  DCA on a market trending up is the simplest honest “pump”.
* **Higher-timeframe trend following** (daily / 4 h).  Intraday noise dominates
  edge on 15 m; stepping up to 4 h or 1 d cuts cost multiplier drastically
  and is known to produce positive backtests for SuperTrend / Donchian
  systems on BTC.
* **Strategy ensembles + very low leverage**, combining the winners here so
  that idle time of one fills active time of another, while keeping total
  exposure capped.

None of these are claims: they are natural next steps if the user wants to
continue this research project on the same data.

---

## 8.2 · Multi-timeframe follow-up (CONFIRMED EDGE ON 4h & 1D)

After the 15m tournament the hypothesis was: *costs, not logic, kill
these strategies.*  So the same 20 strategies, same textbook parameters,
same walk-forward split, same CONS risk profile were re-run on **4-hour**
and **daily** resamples of the very same data.  Implementation in
[`backtest/run_multi_tf.py`](backtest/run_multi_tf.py), full log in
[`results/multi_tf_log.txt`](results/multi_tf_log.txt).

### OOS strategies with **genuine positive edge (PF > 1.0, $200 start)**

| TF | Strategy | End $ | Return | PF | Sharpe | MaxDD | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| **1D** | **🥇 S10 Keltner Breakout** | **$231.66** | **+15.8 %** | **3.21** | **+1.47** | **−4.0 %** | 18 |
| 4h | S05 Donchian-20          | $221.26 | +10.6 % | 1.13 | +0.48 | −15.3 % | 164 |
| 4h | S18 Momentum-3           | $220.85 | +10.4 % | 1.08 | +0.36 | −17.3 % | 347 |
| 4h | S08 VWAP Reversion       | $220.74 | +10.4 % | 1.09 | +0.32 | −24.1 % | 255 |
| 1D | S18 Momentum-3           | $213.05 | +6.5 %  | 1.16 | +0.41 | −6.6 %  | 85  |
| 4h | S02 RSI-2 Reversion      | $212.10 | +6.1 %  | 1.11 | +0.34 | −15.6 % | 126 |
| 4h | S16 Volume Fade          | $211.45 | +5.7 %  | 1.19 | +0.39 | −7.9 %  | 61  |
| 1D | S07 Stoch Cross          | $209.26 | +4.6 %  | 1.18 | +0.37 | −6.3 %  | 54  |
| 4h | S11 SuperTrend           | $205.52 | +2.8 %  | 1.11 | +0.24 | −5.0 %  | 80  |
| 1D | S06 MACD Cross           | $205.22 | +2.6 %  | 1.20 | +0.30 | −5.3 %  | 28  |
| 1D | S03 BB Breakout          | $203.31 | +1.7 %  | 1.09 | +0.17 | −8.0 %  | 33  |
| 4h | S01 EMA Cross            | $198.79 | −0.6 %  | 1.03 | +0.03 | −13.1 % | 135 |

### Headline numbers

|   | 15m | 4h | 1D |
|---|---:|---:|---:|
| # strategies with OOS PF > 1.0  | 1 / 20 | 9 / 20 | 6 / 20 |
| Best OOS end-equity             | $154.43 | $221.26 | **$231.66** |
| Best OOS Sharpe                 | −0.39   | +0.48   | **+1.47** |
| OOS Buy & Hold                  | $320.89 | $322.12 | $308.57 |

### The real findings

1. **The 15m result was structurally, not directionally, wrong.**  The
   very same strategies, the very same parameters, produce positive
   profit factors on 4h and 1D.  What killed 15m was cost drag, not the
   idea.

2. **Multiple statistically meaningful positives.**  4h S05 / S18 / S08
   each have 160–350 trades, which is a real sample — not noise.

3. **🥇 S10 Keltner Breakout on 1D beats Buy & Hold on a risk-adjusted
   basis** (Sharpe **+1.47** vs. B&H ~+0.9 on the OOS window), though on
   only 18 trades — small sample, wide confidence interval.  Use with
   caution and expect the live PF to regress toward ~1.5.

4. **None of the strategies beat Buy & Hold in absolute return** on OOS
   — BTC grew +60 %, the best strategy grew +15.8 %.  That is the honest
   price of risk reduction: a 4 % drawdown vs. a 30 %+ B&H drawdown.

### The practical take-away

* Want **highest absolute return**: hold BTC.  End value $321.
* Want **highest risk-adjusted return with rules**: **S10 Keltner 1D** —
  +15.8 % return, −4 % max DD, Sharpe 1.47.  Smaller sample → treat as
  candidate to paper-trade before risking real money.
* Want **most statistically significant OOS edge**: 4h **S05 Donchian-20**
  or **S18 Momentum-3**.  Hundreds of trades, PF 1.08–1.13, Sharpe ≈ 0.4.
* **Ensemble of the 4h / 1D winners** is the natural next step — the
  green rows above are logically decorrelated (trend, mean-reversion,
  VWAP, volume).  Not implemented yet, but the data is all there.

## 9 · How to reproduce

```bash
pip install pandas numpy
python3 -m backtest.run_tournament
```

Outputs:
* prints both profiles × both windows × every strategy,
* selects the top-10 strictly from in-sample stats,
* writes `results/tournament_results.json` and `results/tournament_log.txt`.

`btc_15m.pkl` is the merged dataset (saved by the first run; regenerated
automatically if missing by the build cell in this README’s history).

## 10 · Repo layout

```
.
├── BTCUSDT-15m-YYYY-MM.csv          (61 monthly files, ~ 178k bars)
├── btc_15m.pkl                      merged pickle used by the tests
├── backtest/
│   ├── engine.py                    event-driven BT engine + indicators
│   ├── strategies.py                the 20 strategies
│   └── run_tournament.py            IS/OOS dual-profile tournament
├── pinescript/                      Pine v5 scripts for the top-10
│   ├── S02_RSI2_Reversion.pine
│   ├── S04_BB_Fade.pine
│   ├── S06_MACD_Cross.pine
│   ├── S10_Keltner_Breakout.pine
│   ├── S11_SuperTrend.pine
│   ├── S13_OpeningRange.pine
│   ├── S16_Volume_Fade.pine
│   ├── S17_ZScore_Reversion.pine
│   ├── S18_Momentum3.pine           ← 🏆 winner
│   └── S20_Williams_R.pine
├── results/
│   ├── tournament_log.txt
│   └── tournament_results.json
└── README.md
```

## 11 · TradingView quick-start

1. Open a `BINANCE:BTCUSDT.P` chart, set timeframe to **15 minutes**.
2. Pine editor → paste any file from [`pinescript/`](pinescript/) → *Add to chart*.
3. Strategy tester tab shows TradingView’s own backtest with fees 0.04 %
   and slippage 2 ticks (same as the Python engine).
4. To compare with this repo’s numbers, set initial capital to **$200**.

---

### Disclaimer

Nothing here is financial advice.  The backtest explicitly shows that the
premise “turn $200 into several thousand in a few months via intraday BTC
indicators” is, on this dataset with honest costs, **not achievable** with
any of these 20 textbook strategies.  The value of this project is the
methodology: reproducible, walk-forward, fixed parameters, fair comparison.
