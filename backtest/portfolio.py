"""
Portfolio backtesting engine.

Runs multiple strategies sharing a single $200 account.  Each strategy
gets its own independent position slot; positions can be open
simultaneously.  Total exposure is capped by max_leverage on the sum of
all open notionals.  This is how real multi-strategy systems run.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .engine import (
    Trade, BTResult,
    STARTING_CAPITAL, FEE_PER_SIDE, SLIPPAGE_PER_SIDE, BARS_PER_YEAR
)


def run_portfolio_backtest(df: pd.DataFrame,
                           strategies: list,     # [(name, signal_fn), ...]
                           name: str = "Portfolio",
                           risk_pct_per_strat: float = 0.01,
                           max_leverage: float = 2.0,
                           cooldown_bars: int = 2,
                           max_hold_bars: int = 30,
                           bars_per_year: float = BARS_PER_YEAR,
                           r_multiple_tp: float = 2.0) -> BTResult:
    """
    * One shared equity curve.
    * Each strategy holds at most one position at a time.
    * Total notional at any moment ≤ max_leverage × equity.
    * Fills at next-bar open (no look-ahead).
    * Fees/slippage identical to the single-strategy engine.
    """
    N = len(strategies)

    # pre-compute every strategy's signals once
    sig_dfs = [fn(df) for _, fn in strategies]
    sig_arrs  = [s['signal'].values  for s in sig_dfs]
    stop_arrs = [s['stop'].values    for s in sig_dfs]
    tp_arrs   = [s['tp'].values if 'tp' in s.columns else np.full(len(df), np.nan)
                 for s in sig_dfs]

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    t = df['open_time'].values

    equity = STARTING_CAPITAL
    equity_curve = np.empty(len(df), dtype=float)
    equity_curve[0] = equity

    in_pos         = [False] * N
    side           = [0]     * N
    entry          = [0.0]   * N
    stop           = [0.0]   * N
    tp             = [0.0]   * N
    qty            = [0.0]   * N
    entry_idx      = [0]     * N
    cooldown_until = [0]     * N
    trades = []

    for i in range(1, len(df)):
        # --- first: evaluate exits on currently open positions ---
        for k in range(N):
            if not in_pos[k]:
                continue
            hi, lo = h[i], l[i]
            exit_px = None; reason = ''
            if side[k] > 0:
                if lo <= stop[k]:
                    exit_px = stop[k] * (1 - SLIPPAGE_PER_SIDE); reason = 'stop'
                elif hi >= tp[k]:
                    exit_px = tp[k]   * (1 - SLIPPAGE_PER_SIDE); reason = 'target'
            else:
                if hi >= stop[k]:
                    exit_px = stop[k] * (1 + SLIPPAGE_PER_SIDE); reason = 'stop'
                elif lo <= tp[k]:
                    exit_px = tp[k]   * (1 + SLIPPAGE_PER_SIDE); reason = 'target'
            if exit_px is None and (i - entry_idx[k]) >= max_hold_bars:
                exit_px = c[i] * (1 - SLIPPAGE_PER_SIDE * side[k]); reason = 'time'
            if exit_px is not None:
                pnl = (exit_px - entry[k]) * qty[k] * side[k]
                equity += pnl - qty[k] * exit_px * FEE_PER_SIDE
                trades.append(Trade(
                    entry_time=pd.to_datetime(t[entry_idx[k]], unit='ms', utc=True),
                    exit_time =pd.to_datetime(t[i],            unit='ms', utc=True),
                    side=side[k], entry=entry[k], exit=exit_px, qty=qty[k],
                    pnl_usd=pnl - qty[k] * exit_px * FEE_PER_SIDE,
                    reason=f"{strategies[k][0]}:{reason}",
                ))
                in_pos[k] = False
                side[k] = 0
                cooldown_until[k] = i + cooldown_bars
                if equity <= 0:
                    equity_curve[i:] = 0
                    return _package(name, trades, equity_curve, equity, t, bars_per_year)

        # current total notional after exits
        total_notional = sum(qty[k] * c[i] for k in range(N) if in_pos[k])

        # --- then: evaluate entries for each strategy ---
        for k in range(N):
            if in_pos[k] or i < cooldown_until[k]:
                continue
            s = sig_arrs[k][i-1]
            if s == 0:
                continue
            entry_px = o[i] * (1 + SLIPPAGE_PER_SIDE * s)
            st  = stop_arrs[k][i-1]
            tgt = tp_arrs[k][i-1]
            if np.isnan(st) or (s > 0 and st >= entry_px) or (s < 0 and st <= entry_px):
                continue
            risk_per_unit = abs(entry_px - st)
            usd_risk = equity * risk_pct_per_strat
            q = usd_risk / risk_per_unit
            # respect portfolio-level leverage cap
            available_notional = max_leverage * equity - total_notional
            if available_notional <= 0:
                continue
            max_q = available_notional / entry_px
            if q > max_q: q = max_q
            if q <= 0:
                continue
            equity -= q * entry_px * FEE_PER_SIDE
            in_pos[k]    = True
            side[k]      = s
            entry[k]     = entry_px
            stop[k]      = st
            tp[k]        = tgt if not np.isnan(tgt) else entry_px + s * r_multiple_tp * risk_per_unit
            qty[k]       = q
            entry_idx[k] = i
            total_notional += q * entry_px

        equity_curve[i] = equity

    # close remaining positions at final close
    for k in range(N):
        if in_pos[k]:
            exit_px = c[-1] * (1 - SLIPPAGE_PER_SIDE * side[k])
            pnl = (exit_px - entry[k]) * qty[k] * side[k]
            equity += pnl - qty[k] * exit_px * FEE_PER_SIDE
            trades.append(Trade(
                entry_time=pd.to_datetime(t[entry_idx[k]], unit='ms', utc=True),
                exit_time =pd.to_datetime(t[-1],           unit='ms', utc=True),
                side=side[k], entry=entry[k], exit=exit_px, qty=qty[k],
                pnl_usd=pnl - qty[k] * exit_px * FEE_PER_SIDE,
                reason=f"{strategies[k][0]}:eod",
            ))
    equity_curve[-1] = equity
    return _package(name, trades, equity_curve, equity, t, bars_per_year)


def _package(name, trades, eq_arr, final_eq, t, bars_per_year):
    eq = pd.Series(eq_arr, index=pd.to_datetime(t, unit='ms', utc=True))
    return BTResult(name=name, trades=trades, equity=eq,
                    start_eq=STARTING_CAPITAL, end_eq=float(final_eq),
                    bars_per_year=bars_per_year)
