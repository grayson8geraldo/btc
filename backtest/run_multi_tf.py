"""
Multi-timeframe tournament.

Runs the SAME 20 strategies with the SAME textbook parameters on
15m, 4h and 1D BTCUSDT bars, under the same CONS risk profile,
with walk-forward IS / OOS split identical to the 15m tournament.

Hypothesis (see README §8): higher timeframes drastically reduce
cost drag and should reveal whether textbook strategies have a real
edge on BTC once friction is removed.

Rules (fixed before seeing results):
  * IS  = 2021-01-01 .. 2023-12-31  (≈3y)
  * OOS = 2024-01-01 .. 2026-03-31  (≈2.25y)
  * Risk profile: 1% risk, 2x leverage, 8h cooldown (scaled per tf)
  * max_hold = 24h on 15m, 5 days on 4h, 20 days on 1D
  * No parameter optimization
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest, STARTING_CAPITAL
from backtest.strategies import ALL_STRATEGIES


IS_END_MS = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp() * 1000)


def load_15m():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_pickle(os.path.join(here, 'btc_15m.pkl')).reset_index(drop=True)


def resample(df_15m, rule: str):
    """Resample 15m OHLCV into a higher timeframe while preserving engine layout."""
    d = df_15m.copy()
    d['dt'] = pd.to_datetime(d['open_time'], unit='ms', utc=True)
    d = d.set_index('dt')
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    r = d.resample(rule, label='left', closed='left').agg(agg).dropna()
    r = r.reset_index()
    r['open_time'] = (r['dt'].astype('int64') // 10**6).astype('int64')
    return r[['open_time','open','high','low','close','volume']]


# ---- timeframe-specific config (all values chosen a-priori) ----
TF_CONFIGS = {
    '15m': dict(rule=None,  bars_per_year=365*24*4,  max_hold_bars=96, cooldown_bars=32),  # 24h hold / 8h cd
    '4h':  dict(rule='4h',  bars_per_year=365*6,     max_hold_bars=30, cooldown_bars=2),   # 5d hold / 8h cd
    '1D':  dict(rule='1D',  bars_per_year=365,       max_hold_bars=20, cooldown_bars=1),   # 20d hold / 1d cd
}

RISK_PCT_CONS      = 0.01
MAX_LEVERAGE_CONS  = 2.0


def run_phase(df, label, strategies, cfg):
    rows = []
    for name, fn in strategies:
        res = run_backtest(
            df, fn, name,
            risk_pct     = RISK_PCT_CONS,
            max_leverage = MAX_LEVERAGE_CONS,
            cooldown_bars= cfg['cooldown_bars'],
            max_hold_bars= cfg['max_hold_bars'],
            bars_per_year= cfg['bars_per_year'],
        )
        s = res.summary()
        s['window'] = label
        rows.append(s)
    return rows


def fmt(r):
    return (f"  {r['strategy']:24s}  end=${r['end_equity']:9.2f}  "
            f"ret={r['total_return']*100:+8.2f}%  "
            f"sharpe={r['sharpe']:+7.2f}  "
            f"pf={r['profit_factor']:5.2f}  "
            f"dd={r['max_dd']*100:+7.2f}%  "
            f"trades={r['trades']:5d}")


def buy_hold(df):
    p0 = df['close'].iloc[0]; p1 = df['close'].iloc[-1]
    return STARTING_CAPITAL * p1 / p0, p1/p0 - 1


def main():
    df15 = load_15m()
    df15_is  = df15[df15['open_time'] <  IS_END_MS].reset_index(drop=True)
    df15_oos = df15[df15['open_time'] >= IS_END_MS].reset_index(drop=True)

    datasets = {
        '15m': (df15_is, df15_oos),
        '4h':  (resample(df15_is, '4h'), resample(df15_oos, '4h')),
        '1D':  (resample(df15_is, '1D'), resample(df15_oos, '1D')),
    }

    print('\n================================================================')
    print('MULTI-TIMEFRAME TOURNAMENT — CONS profile  ($200 start)')
    print('================================================================')
    for tf, (isd, oosd) in datasets.items():
        bh_is,  ret_is  = buy_hold(isd)
        bh_oos, ret_oos = buy_hold(oosd)
        print(f'\n{tf}: IS bars={len(isd):,}  OOS bars={len(oosd):,}   '
              f'B&H IS=${bh_is:.2f} ({ret_is:+.1%})  '
              f'B&H OOS=${bh_oos:.2f} ({ret_oos:+.1%})')

    full_results = {}
    for tf, (isd, oosd) in datasets.items():
        cfg = TF_CONFIGS[tf]
        print(f'\n------------------  TIMEFRAME {tf}  ------------------')
        print(f'  hold={cfg["max_hold_bars"]}b  cooldown={cfg["cooldown_bars"]}b  '
              f'bars/year={cfg["bars_per_year"]}')
        print(f'\n---- IS ----')
        is_rows = run_phase(isd, 'IS', ALL_STRATEGIES, cfg)
        for r in is_rows: print(fmt(r))
        print(f'\n---- OOS (unchanged strategies, unchanged params) ----')
        oos_rows = run_phase(oosd, 'OOS', ALL_STRATEGIES, cfg)
        for r in oos_rows: print(fmt(r))
        full_results[tf] = dict(is_=is_rows, oos=oos_rows)

    # ---- CROSS-TF LEADERBOARD (OOS end equity) ----
    print('\n================================================================')
    print('CROSS-TIMEFRAME OOS LEADERBOARD (ranked by OOS end equity)')
    print('================================================================')
    all_oos = []
    for tf, r in full_results.items():
        for row in r['oos']:
            row2 = dict(row); row2['tf'] = tf
            all_oos.append(row2)
    all_oos.sort(key=lambda x: x['end_equity'], reverse=True)
    print(f"\n{'TF':<5}{'Strategy':<26}{'End $':>10}{'Ret':>10}"
          f"{'PF':>7}{'Sharpe':>9}{'DD':>10}{'Trades':>8}")
    for r in all_oos[:25]:
        print(f"{r['tf']:<5}{r['strategy']:<26}${r['end_equity']:>8.2f}"
              f"{r['total_return']*100:>+9.1f}%{r['profit_factor']:>7.2f}"
              f"{r['sharpe']:>+9.2f}{r['max_dd']*100:>+9.1f}%{r['trades']:>8d}")

    # ---- top-5 PF on OOS (strategies with real edge) ----
    print('\n--- Strategies with OOS profit factor > 1.0 (genuine edge) ---')
    with_edge = [r for r in all_oos if r['profit_factor'] > 1.0]
    with_edge.sort(key=lambda x: (x['profit_factor'], x['end_equity']), reverse=True)
    if not with_edge:
        print('  NONE.')
    else:
        for r in with_edge:
            print(f"  {r['tf']:<4}{r['strategy']:<26}  "
                  f"end=${r['end_equity']:8.2f}  ret={r['total_return']*100:+7.2f}%  "
                  f"PF={r['profit_factor']:.3f}  sharpe={r['sharpe']:+.2f}  "
                  f"trades={r['trades']}")

    # ---- save ----
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'results', 'multi_tf_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print('\nresults/multi_tf_results.json saved.')


if __name__ == '__main__':
    main()
