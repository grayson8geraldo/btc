"""
Ensemble tournament.

Tests several pre-declared strategy combinations (no optimization — every
constituent was already chosen from the earlier multi-tf study) as
portfolio backtests sharing a single $200 account.

All ensembles run on the 4h and/or 1D timeframe, under the same cost
model, same IS/OOS split, same risk-management rules as before.

Ensembles (defined a-priori based on §8.2 of README):

  E1  4h "Diversified-3"   : S05 Donchian-20 + S18 Momentum-3 + S02 RSI-2
  E2  4h "Diversified-5"   : E1 + S08 VWAP Reversion + S16 Volume Fade
  E3  1D "Classic-3"       : S10 Keltner Breakout + S18 Momentum-3 + S06 MACD Cross
  E4  1D+4h "Cross-TF"     : [1D] S10 Keltner + [4h] S05 Donchian + [4h] S18 Momentum
                             (implemented as the 1D+4h strategies run
                              independently and summed — see note)

Rules (identical to the earlier tournaments):
  * Fee 0.04% + slippage 0.02% per side  (0.12% round-trip)
  * 1% risk per strategy per trade
  * Max 2x total portfolio leverage
  * Cooldown (scaled per tf): 2 bars @ 4h, 1 bar @ 1D
  * Max hold:                 30 bars @ 4h, 20 bars @ 1D
  * $200 starting capital
  * Walk-forward:  IS 2021-01..2023-12,  OOS 2024-01..2026-03
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import STARTING_CAPITAL
from backtest.portfolio import run_portfolio_backtest
from backtest.strategies import (
    s02_rsi2_reversion, s05_donchian_breakout, s06_macd_cross,
    s08_vwap_reversion, s10_keltner_breakout, s16_volume_fade,
    s18_momentum3,
)
from backtest.run_multi_tf import resample, load_15m


IS_END_MS = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp() * 1000)

TF_CFG = {
    '4h': dict(max_hold_bars=30, cooldown_bars=2, bars_per_year=365*6),
    '1D': dict(max_hold_bars=20, cooldown_bars=1, bars_per_year=365),
}

ENSEMBLES = {
    'E1_4h_Diversified3': dict(
        tf='4h',
        components=[
            ('S05_Donchian20',      s05_donchian_breakout),
            ('S18_Momentum3',       s18_momentum3),
            ('S02_RSI2_Reversion',  s02_rsi2_reversion),
        ],
    ),
    'E2_4h_Diversified5': dict(
        tf='4h',
        components=[
            ('S05_Donchian20',      s05_donchian_breakout),
            ('S18_Momentum3',       s18_momentum3),
            ('S02_RSI2_Reversion',  s02_rsi2_reversion),
            ('S08_VWAP_Reversion',  s08_vwap_reversion),
            ('S16_Volume_Fade',     s16_volume_fade),
        ],
    ),
    'E3_1D_Classic3': dict(
        tf='1D',
        components=[
            ('S10_Keltner_Breakout', s10_keltner_breakout),
            ('S18_Momentum3',        s18_momentum3),
            ('S06_MACD_Cross',       s06_macd_cross),
        ],
    ),
}


def bh(df):
    p0, p1 = df['close'].iloc[0], df['close'].iloc[-1]
    return STARTING_CAPITAL * p1 / p0, p1/p0 - 1


def fmt(title, r):
    return (f"{title:<30}  end=${r.end_eq:8.2f}  "
            f"ret={r.total_return*100:+7.2f}%  "
            f"sharpe={r.sharpe:+6.2f}  "
            f"pf={r.profit_factor:5.2f}  "
            f"dd={r.max_dd*100:+7.2f}%  "
            f"trades={r.n_trades:4d}  "
            f"wr={r.win_rate*100:5.1f}%")


def main():
    df15 = load_15m()
    # split at 15m level BEFORE resampling (same approach as run_multi_tf)
    df15_is  = df15[df15['open_time']  < IS_END_MS].reset_index(drop=True)
    df15_oos = df15[df15['open_time'] >= IS_END_MS].reset_index(drop=True)

    # cache resampled datasets
    datasets = {}
    for tf in ['4h', '1D']:
        datasets[tf] = dict(
            is_=resample(df15_is,  tf),
            oos=resample(df15_oos, tf),
        )

    print('\n================================================================')
    print(' ENSEMBLE TOURNAMENT — portfolio backtest on shared $200 account')
    print('================================================================')
    print('\nWindow boundaries:')
    for tf in ['4h', '1D']:
        d = datasets[tf]
        print(f'  {tf}: IS bars={len(d["is_"]):,}  OOS bars={len(d["oos"]):,}')

    print('\nBenchmarks (Buy & Hold):')
    for tf in ['4h', '1D']:
        d = datasets[tf]
        bh_is,  ret_is  = bh(d['is_'])
        bh_oos, ret_oos = bh(d['oos'])
        print(f'  {tf}  B&H IS  = ${bh_is:.2f}  ({ret_is:+.2%})    '
              f'OOS = ${bh_oos:.2f}  ({ret_oos:+.2%})')

    # ----- 4h/1D single-strategy CONS baselines for comparison -----
    print('\n---- individual components, OOS, for reference ----')
    from backtest.engine import run_backtest
    for ename, e in ENSEMBLES.items():
        cfg = TF_CFG[e['tf']]
        oos = datasets[e['tf']]['oos']
        print(f'\n{ename}  ({e["tf"]}):')
        for sname, sfn in e['components']:
            r = run_backtest(oos, sfn, sname,
                             risk_pct=0.01, max_leverage=2.0,
                             cooldown_bars=cfg['cooldown_bars'],
                             max_hold_bars=cfg['max_hold_bars'],
                             bars_per_year=cfg['bars_per_year'])
            print('  '+fmt(f"  [single] {sname}", r))

    # ----- actual ensemble runs -----
    print('\n================================================================')
    print(' ENSEMBLES (shared $200 account, 1% risk per strategy, 2x cap)')
    print('================================================================')

    summary = {}
    for ename, e in ENSEMBLES.items():
        cfg = TF_CFG[e['tf']]
        print(f'\n{ename}  ({e["tf"]}, {len(e["components"])} components)')
        is_res  = run_portfolio_backtest(
            datasets[e['tf']]['is_'], e['components'], name=ename+"_IS",
            risk_pct_per_strat=0.01, max_leverage=2.0,
            cooldown_bars=cfg['cooldown_bars'],
            max_hold_bars=cfg['max_hold_bars'],
            bars_per_year=cfg['bars_per_year'])
        oos_res = run_portfolio_backtest(
            datasets[e['tf']]['oos'], e['components'], name=ename+"_OOS",
            risk_pct_per_strat=0.01, max_leverage=2.0,
            cooldown_bars=cfg['cooldown_bars'],
            max_hold_bars=cfg['max_hold_bars'],
            bars_per_year=cfg['bars_per_year'])
        print('  '+fmt('IS  '+ename, is_res))
        print('  '+fmt('OOS '+ename, oos_res))
        summary[ename] = dict(is_=is_res.summary(), oos=oos_res.summary())

    # ----- cross-TF ensemble (E4): combine already-computed 1D and 4h runs -----
    # Implemented by running portfolio twice (once per TF) with a SPLIT of the
    # $200 capital — half on each TF.  This approximates the user wearing two
    # bots on different timeframes.  For honesty we report it but flag it.
    print('\n================================================================')
    print(' E4  Cross-TF (4h + 1D) — not implemented as joint portfolio;')
    print(' approximated as $100 on each of the two sub-portfolios.')
    print('================================================================')

    from backtest.engine import STARTING_CAPITAL as SC_ORIG
    from backtest import portfolio as _p, engine as _e

    def run_half(df, components, tf):
        """Temporarily set starting capital to $100 for the sub-portfolio."""
        _p.STARTING_CAPITAL = 100.0
        _e.STARTING_CAPITAL = 100.0
        try:
            return run_portfolio_backtest(
                df, components, name='half',
                risk_pct_per_strat=0.01, max_leverage=2.0,
                cooldown_bars=TF_CFG[tf]['cooldown_bars'],
                max_hold_bars=TF_CFG[tf]['max_hold_bars'],
                bars_per_year=TF_CFG[tf]['bars_per_year'])
        finally:
            _p.STARTING_CAPITAL = SC_ORIG
            _e.STARTING_CAPITAL = SC_ORIG

    comp_4h = [
        ('S05_Donchian20', s05_donchian_breakout),
        ('S18_Momentum3',  s18_momentum3),
    ]
    comp_1D = [
        ('S10_Keltner_Breakout', s10_keltner_breakout),
    ]
    e4_is_4h  = run_half(datasets['4h']['is_'], comp_4h, '4h')
    e4_is_1D  = run_half(datasets['1D']['is_'], comp_1D, '1D')
    e4_oos_4h = run_half(datasets['4h']['oos'], comp_4h, '4h')
    e4_oos_1D = run_half(datasets['1D']['oos'], comp_1D, '1D')

    e4_is_eq  = e4_is_4h.end_eq + e4_is_1D.end_eq
    e4_oos_eq = e4_oos_4h.end_eq + e4_oos_1D.end_eq
    print(f'\nE4_CrossTF  IS  :  $100→${e4_is_4h.end_eq:.2f} (4h) + '
          f'$100→${e4_is_1D.end_eq:.2f} (1D) = ${e4_is_eq:.2f} '
          f'({e4_is_eq/200-1:+.2%})')
    print(f'E4_CrossTF  OOS :  $100→${e4_oos_4h.end_eq:.2f} (4h) + '
          f'$100→${e4_oos_1D.end_eq:.2f} (1D) = ${e4_oos_eq:.2f} '
          f'({e4_oos_eq/200-1:+.2%})')
    summary['E4_CrossTF'] = dict(
        is_end=e4_is_eq, oos_end=e4_oos_eq,
        is_ret=e4_is_eq/200-1, oos_ret=e4_oos_eq/200-1,
    )

    # ----- final leaderboard -----
    print('\n================================================================')
    print(' FINAL OOS ENSEMBLE LEADERBOARD (ranked by OOS end equity)')
    print('================================================================')
    leaders = []
    for ename, s in summary.items():
        if 'oos' in s:
            leaders.append((s['oos']['end_equity'], ename, s['oos']))
        elif 'oos_end' in s:
            leaders.append((s['oos_end'], ename,
                            dict(end_equity=s['oos_end'],
                                 total_return=s['oos_ret'],
                                 sharpe=None, profit_factor=None,
                                 max_dd=None, trades=None)))
    leaders.sort(reverse=True)
    for end, name, s in leaders:
        if s.get('sharpe') is not None:
            print(f"  {name:<22}  end=${end:8.2f}  ret={s['total_return']*100:+6.2f}%  "
                  f"sharpe={s['sharpe']:+6.2f}  pf={s['profit_factor']:5.2f}  "
                  f"dd={s['max_dd']*100:+7.2f}%  trades={s['trades']:4d}")
        else:
            print(f"  {name:<22}  end=${end:8.2f}  ret={s['total_return']*100:+6.2f}%  "
                  f"(cross-TF split)")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'results', 'ensembles_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print('\nresults/ensembles_results.json saved.')


if __name__ == '__main__':
    main()
