"""
Tournament runner.

Honest walk-forward design:
  * Split:  IN-SAMPLE  2021-01-01 → 2023-12-31    (≈ 3 years)
            OUT-OF-SAMPLE 2024-01-01 → 2026-03-31 (≈ 2.25 years, 27 months)
  * Two fixed risk-management profiles (chosen a-priori, NOT tuned to data):
        AGGR  — 2% risk, 3x max leverage, no cooldown
        CONS  — 1% risk, 2x max leverage, 8-hour cooldown between trades
  * All 20 strategies are run on IN-SAMPLE under BOTH profiles.
  * Top-10 selection: rank by CONS-profile IS rank_score
        rank_score = 0.45·Sharpe + 0.25·ProfitFactor
                   + 0.15·ExpectancyR + 0.15·(-MaxDD)   (min-max normalized)
  * Those 10 strategies, UNCHANGED, are then run on OOS under BOTH profiles.
    The winner is the strategy whose OOS CONS end-equity is highest.
  * No optimization, no peeking.
"""
from __future__ import annotations
import sys, os, json, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest, STARTING_CAPITAL
from backtest.strategies import ALL_STRATEGIES


IS_END_MS = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp() * 1000)

AGGR = dict(risk_pct=0.02, max_leverage=3.0, cooldown_bars=0,  profile='AGGR')
CONS = dict(risk_pct=0.01, max_leverage=2.0, cooldown_bars=32, profile='CONS')


def load_data(path='btc_15m.pkl'):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_pickle(os.path.join(here, path))
    return df.reset_index(drop=True)


def _norm(x):
    x = np.array(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12: return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rank_score(rows):
    sharpe = [r['sharpe'] for r in rows]
    pf     = [min(r['profit_factor'], 5.0) for r in rows]
    er     = [r['expectancy_R'] for r in rows]
    dd     = [-r['max_dd'] for r in rows]
    return (0.45*_norm(sharpe) + 0.25*_norm(pf)
          + 0.15*_norm(er)     + 0.15*_norm(dd))


def run_phase(df, window_label, strategies, profile):
    rows = []
    for name, fn in strategies:
        res = run_backtest(
            df, fn, name,
            risk_pct=profile['risk_pct'],
            max_leverage=profile['max_leverage'],
            cooldown_bars=profile['cooldown_bars'],
        )
        s = res.summary()
        s['window']  = window_label
        s['profile'] = profile['profile']
        rows.append(s)
    return rows


def fmt_row(r):
    return (f"  {r['strategy']:24s}  end=${r['end_equity']:8.2f}  "
            f"ret={r['total_return']*100:+8.2f}%  "
            f"sharpe={r['sharpe']:+7.2f}  "
            f"dd={r['max_dd']*100:+7.2f}%  "
            f"pf={r['profit_factor']:5.2f}  "
            f"trades={r['trades']:5d}")


def main():
    df = load_data()
    df_is  = df[df['open_time'] <  IS_END_MS].reset_index(drop=True).copy()
    df_oos = df[df['open_time'] >= IS_END_MS].reset_index(drop=True).copy()
    print(f'IS  bars: {len(df_is):,}  '
          f'({pd.to_datetime(df_is.open_time.iloc[0],  unit="ms", utc=True).date()} → '
          f'{pd.to_datetime(df_is.open_time.iloc[-1], unit="ms", utc=True).date()})')
    print(f'OOS bars: {len(df_oos):,}  '
          f'({pd.to_datetime(df_oos.open_time.iloc[0],  unit="ms", utc=True).date()} → '
          f'{pd.to_datetime(df_oos.open_time.iloc[-1], unit="ms", utc=True).date()})')

    # ===================  PHASE 1 — IN-SAMPLE, all 20  ===================
    print('\n================================================================')
    print('PHASE 1 — 20 strategies on IN-SAMPLE  (2021-01 … 2023-12)')
    print('================================================================')

    print('\n---- AGGR profile (2% risk · 3x lev · no cooldown) ----')
    is_aggr = run_phase(df_is, 'IS', ALL_STRATEGIES, AGGR)
    for r in is_aggr: print(fmt_row(r))

    print('\n---- CONS profile (1% risk · 2x lev · 8h cooldown) ----')
    is_cons = run_phase(df_is, 'IS', ALL_STRATEGIES, CONS)
    for r in is_cons: print(fmt_row(r))

    # --- select top-10 strictly by CONS in-sample rank_score ---
    scores = rank_score(is_cons)
    order  = np.argsort(-scores)
    top10_idx   = order[:10]
    top10_names = [is_cons[i]['strategy'] for i in top10_idx]
    print('\n---- TOP-10 selected by CONS-IS rank_score ----')
    for rank, idx in enumerate(top10_idx, 1):
        r = is_cons[idx]
        print(f"  {rank:2d}. {r['strategy']:24s}  "
              f"score={scores[idx]:.3f}  sharpe={r['sharpe']:+.2f}  "
              f"pf={r['profit_factor']:.2f}  end=${r['end_equity']:.2f}")
    top10_strats = [s for s in ALL_STRATEGIES if s[0] in top10_names]

    # ===================  PHASE 2 — OUT-OF-SAMPLE, top-10  ===================
    print('\n================================================================')
    print('PHASE 2 — top-10 on OUT-OF-SAMPLE  (2024-01 … 2026-03)')
    print('================================================================')

    print('\n---- OOS  AGGR profile ----')
    oos_aggr = run_phase(df_oos, 'OOS', top10_strats, AGGR)
    for r in oos_aggr: print(fmt_row(r))

    print('\n---- OOS  CONS profile ----')
    oos_cons = run_phase(df_oos, 'OOS', top10_strats, CONS)
    for r in oos_cons: print(fmt_row(r))

    # ===================  FINAL RANK (OOS CONS end equity)  ===================
    print('\n================================================================')
    print('FINAL OOS CONS TOURNAMENT STANDINGS (ranked by end equity)')
    print('================================================================')
    standings = sorted(oos_cons, key=lambda r: r['end_equity'], reverse=True)
    for pos, r in enumerate(standings, 1):
        tag = ' <-- WINNER' if pos == 1 else ''
        print(f"{pos:2d}. {fmt_row(r).strip()}{tag}")

    winner = standings[0]
    print('\nWINNER:')
    print(json.dumps(winner, indent=2, default=str))

    out = dict(
        config = dict(
            starting_capital=STARTING_CAPITAL,
            is_window='2021-01-01..2023-12-31',
            oos_window='2024-01-01..2026-03-31',
            fee_per_side=0.0004,
            slippage_per_side=0.0002,
            roundtrip_cost=0.0012,
            AGGR=AGGR, CONS=CONS,
            r_multiple_tp=2.0,
            max_hold_bars=96,
        ),
        in_sample_aggr  = is_aggr,
        in_sample_cons  = is_cons,
        top10_names     = top10_names,
        out_of_sample_aggr = oos_aggr,
        out_of_sample_cons = oos_cons,
        standings_oos_cons = standings,
        winner            = winner,
    )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'results', 'tournament_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('\nresults/tournament_results.json saved.')


if __name__ == '__main__':
    main()
