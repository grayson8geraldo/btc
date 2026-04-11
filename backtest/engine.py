"""
Unified event-driven backtesting engine for BTCUSDT 15m bars.

Honest backtest rules:
  * Signals are generated on bar CLOSE (index i) and executed at bar OPEN (index i+1).
  * No peeking into future bars — all indicators use shift()/rolling operations only.
  * Costs: taker fee 0.04% per side (perpetual futures Binance) + 0.02% slippage per side
    => round-trip cost ~0.12% of notional.
  * SL / TP / trailing stops are evaluated intrabar using high/low of next bar(s);
    when SL and TP would both be hit in the same bar we conservatively assume SL first.
  * Fixed fractional risk sizing:
        qty = (equity * risk_pct) / (entry - stop_price)
    leverage cap applied (max_leverage=3) so that qty * price <= equity * max_leverage.
  * ONE position at a time per strategy (intraday focus).
  * Starting capital = $200.
  * No parameter optimization on the test set — parameters are textbook defaults
    fixed before seeing OOS data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional


# ----------  Fixed backtest configuration (same for every strategy)  ----------
FEE_PER_SIDE = 0.0004        # 0.04% Binance USDT-M taker
SLIPPAGE_PER_SIDE = 0.0002   # 0.02% half-spread slippage
COST_ROUND_TRIP = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)  # 0.12%
STARTING_CAPITAL = 200.0
RISK_PCT = 0.02              # risk 2% of equity per trade
MAX_LEVERAGE = 3.0           # notional / equity cap
BARS_PER_YEAR = 365 * 24 * 4 # 35,040  (15m bars)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time:  pd.Timestamp
    side:       int    # +1 long / -1 short
    entry:      float
    exit:       float
    qty:        float
    pnl_usd:    float
    reason:     str    # "signal" | "stop" | "target" | "time"


@dataclass
class BTResult:
    name:       str
    trades:     list = field(default_factory=list)
    equity:     pd.Series = None
    start_eq:   float = STARTING_CAPITAL
    end_eq:     float = STARTING_CAPITAL
    bars_per_year: float = BARS_PER_YEAR

    @property
    def n_trades(self):            return len(self.trades)
    @property
    def total_return(self):        return self.end_eq / self.start_eq - 1
    @property
    def wins(self):                return [t for t in self.trades if t.pnl_usd > 0]
    @property
    def losses(self):              return [t for t in self.trades if t.pnl_usd <= 0]
    @property
    def win_rate(self):
        return len(self.wins) / self.n_trades if self.n_trades else 0.0
    @property
    def profit_factor(self):
        g = sum(t.pnl_usd for t in self.wins)
        l = -sum(t.pnl_usd for t in self.losses)
        return g / l if l > 0 else float('inf') if g > 0 else 0.0
    @property
    def avg_pnl(self):
        return np.mean([t.pnl_usd for t in self.trades]) if self.n_trades else 0.0
    @property
    def expectancy_R(self):
        pnls = [t.pnl_usd for t in self.trades]
        return float(np.mean(pnls)) / (self.start_eq * RISK_PCT) if pnls else 0.0
    @property
    def max_dd(self):
        if self.equity is None or len(self.equity) == 0: return 0.0
        peak = self.equity.cummax()
        dd = self.equity / peak - 1
        return float(dd.min())
    @property
    def sharpe(self):
        if self.equity is None or len(self.equity) < 2: return 0.0
        rets = self.equity.pct_change().dropna()
        if rets.std() == 0: return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(self.bars_per_year))
    @property
    def cagr(self):
        if self.equity is None or len(self.equity) < 2: return 0.0
        years = len(self.equity) / self.bars_per_year
        if years <= 0: return 0.0
        if self.end_eq <= 0: return -1.0
        return (self.end_eq / self.start_eq) ** (1/years) - 1

    def summary(self) -> dict:
        return dict(
            strategy     = self.name,
            trades       = self.n_trades,
            win_rate     = round(self.win_rate, 4),
            profit_factor= round(self.profit_factor, 3) if self.profit_factor != float('inf') else 999,
            total_return = round(self.total_return, 4),
            cagr         = round(self.cagr, 4),
            sharpe       = round(self.sharpe, 3),
            max_dd       = round(self.max_dd, 4),
            end_equity   = round(self.end_eq, 2),
            expectancy_R = round(self.expectancy_R, 3),
        )


SignalFn = Callable[[pd.DataFrame], pd.DataFrame]
# A signal function must return a DataFrame with columns:
#   signal: -1, 0, +1
#   stop  : absolute stop price for *that bar's* hypothetical entry (NaN if no signal)
#   tp    : absolute take-profit price (NaN -> use R multiple default)
# extra columns allowed.


def run_backtest(df: pd.DataFrame,
                 signal_fn: SignalFn,
                 name: str,
                 r_multiple_tp: float = 2.0,
                 max_hold_bars: int = 96,    # 24h max hold = intraday focus
                 cooldown_bars: int = 0,     # bars to wait after a closed trade
                 risk_pct: float = RISK_PCT,
                 max_leverage: float = MAX_LEVERAGE,
                 bars_per_year: float = BARS_PER_YEAR,
                 allow_short: bool = True) -> BTResult:
    """
    df must have columns: open_time, open, high, low, close, volume
    Index is RangeIndex.
    """
    sig_df = signal_fn(df)
    sig = sig_df['signal'].values
    stop_arr = sig_df['stop'].values
    tp_arr   = sig_df['tp'].values if 'tp' in sig_df.columns else np.full(len(df), np.nan)

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    t = df['open_time'].values

    equity = STARTING_CAPITAL
    equity_curve = np.empty(len(df), dtype=float)
    equity_curve[0] = equity
    trades: list[Trade] = []

    i = 1  # start at 1 because we enter at next bar's open
    in_pos = False
    side = 0
    entry = 0.0
    stop = 0.0
    tp = 0.0
    qty = 0.0
    entry_idx = 0
    cooldown_until = 0

    while i < len(df):
        if not in_pos:
            s = sig[i-1]
            if s != 0 and (allow_short or s > 0) and i >= cooldown_until:
                # Execute at THIS bar's open
                raw_entry = o[i]
                # apply slippage on entry
                entry_px = raw_entry * (1 + SLIPPAGE_PER_SIDE * s)
                st = stop_arr[i-1]
                tgt = tp_arr[i-1]
                if np.isnan(st) or (s > 0 and st >= entry_px) or (s < 0 and st <= entry_px):
                    equity_curve[i] = equity; i += 1; continue
                risk_per_unit = abs(entry_px - st)
                usd_risk = equity * risk_pct
                q = usd_risk / risk_per_unit
                # leverage cap
                max_q = equity * max_leverage / entry_px
                if q > max_q: q = max_q
                if q <= 0:
                    equity_curve[i] = equity; i += 1; continue
                # pay entry fee
                equity -= q * entry_px * FEE_PER_SIDE
                side = s
                entry = entry_px
                stop = st
                tp = tgt if not np.isnan(tgt) else entry + side * r_multiple_tp * risk_per_unit
                qty = q
                entry_idx = i
                in_pos = True
            equity_curve[i] = equity
            i += 1
            continue

        # in position — evaluate this bar
        hi = h[i]; lo = l[i]
        exit_px = None; reason = ''
        # conservative: check stop first
        if side > 0:
            if lo <= stop:
                exit_px = stop * (1 - SLIPPAGE_PER_SIDE); reason = 'stop'
            elif hi >= tp:
                exit_px = tp * (1 - SLIPPAGE_PER_SIDE); reason = 'target'
        else:
            if hi >= stop:
                exit_px = stop * (1 + SLIPPAGE_PER_SIDE); reason = 'stop'
            elif lo <= tp:
                exit_px = tp * (1 + SLIPPAGE_PER_SIDE); reason = 'target'

        # time stop
        if exit_px is None and (i - entry_idx) >= max_hold_bars:
            exit_px = c[i] * (1 - SLIPPAGE_PER_SIDE * side)
            reason = 'time'

        # opposite signal while in position -> exit at next open
        if exit_px is None and sig[i-1] == -side:
            # Will be handled next iter as forced exit via time? Simplify: exit now at this bar open
            # (already in position so use close of prev bar -> use current bar open)
            exit_px = o[i] * (1 - SLIPPAGE_PER_SIDE * side)
            reason = 'signal'

        if exit_px is not None:
            pnl = (exit_px - entry) * qty * side
            equity += pnl
            equity -= qty * exit_px * FEE_PER_SIDE   # exit fee
            trades.append(Trade(
                entry_time=pd.to_datetime(t[entry_idx], unit='ms', utc=True),
                exit_time=pd.to_datetime(t[i], unit='ms', utc=True),
                side=side, entry=entry, exit=exit_px, qty=qty,
                pnl_usd=pnl - qty * exit_px * FEE_PER_SIDE,
                reason=reason,
            ))
            in_pos = False
            side = 0
            cooldown_until = i + cooldown_bars
            if equity <= 0:
                # account blown — zero out and stop
                equity_curve[i:] = 0
                break
        equity_curve[i] = equity
        i += 1

    # close any open position at last close
    if in_pos:
        exit_px = c[-1] * (1 - SLIPPAGE_PER_SIDE * side)
        pnl = (exit_px - entry) * qty * side
        equity += pnl - qty * exit_px * FEE_PER_SIDE
        trades.append(Trade(
            entry_time=pd.to_datetime(t[entry_idx], unit='ms', utc=True),
            exit_time=pd.to_datetime(t[-1], unit='ms', utc=True),
            side=side, entry=entry, exit=exit_px, qty=qty,
            pnl_usd=pnl - qty * exit_px * FEE_PER_SIDE,
            reason='eod',
        ))
        equity_curve[-1] = equity

    eq_series = pd.Series(equity_curve, index=pd.to_datetime(t, unit='ms', utc=True))
    return BTResult(name=name, trades=trades, equity=eq_series,
                    start_eq=STARTING_CAPITAL, end_eq=float(equity),
                    bars_per_year=bars_per_year)


# ----------------  Common indicator helpers  -----------------

def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()

def atr(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def stoch(df, k=14, d=3):
    ll = df['low'].rolling(k).min()
    hh = df['high'].rolling(k).max()
    kv = 100 * (df['close'] - ll) / (hh - ll)
    dv = kv.rolling(d).mean()
    return kv, dv

def bollinger(s, n=20, k=2):
    mid = s.rolling(n).mean()
    sd  = s.rolling(n).std()
    return mid + k*sd, mid, mid - k*sd

def donchian(df, n=20):
    return df['high'].rolling(n).max(), df['low'].rolling(n).min()

def adx(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, mdi

def macd(s, fast=12, slow=26, sig=9):
    f = ema(s, fast); sl = ema(s, slow)
    line = f - sl
    signal = ema(line, sig)
    return line, signal, line - signal

def supertrend(df, period=10, mult=3.0):
    a = atr(df, period)
    hl2 = (df['high']+df['low'])/2
    up = hl2 + mult*a
    dn = hl2 - mult*a
    st = pd.Series(np.nan, index=df.index)
    dir_ = pd.Series(1, index=df.index)
    c = df['close'].values
    up_v = up.values.copy(); dn_v = dn.values.copy()
    for i in range(1, len(df)):
        up_v[i] = min(up_v[i], up_v[i-1]) if c[i-1] <= up_v[i-1] else up_v[i]
        dn_v[i] = max(dn_v[i], dn_v[i-1]) if c[i-1] >= dn_v[i-1] else dn_v[i]
        if c[i] > up_v[i-1]:    dir_.iloc[i] = 1
        elif c[i] < dn_v[i-1]:  dir_.iloc[i] = -1
        else:                   dir_.iloc[i] = dir_.iloc[i-1]
        st.iloc[i] = dn_v[i] if dir_.iloc[i] == 1 else up_v[i]
    return st, dir_

def keltner(df, n=20, mult=2.0):
    ma = ema(df['close'], n)
    a  = atr(df, n)
    return ma + mult*a, ma, ma - mult*a

def vwap_session(df, session='1D'):
    """Anchored daily VWAP."""
    dt = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    day = dt.dt.floor('D')
    tp = (df['high']+df['low']+df['close'])/3
    cum_pv = (tp * df['volume']).groupby(day).cumsum()
    cum_v  = df['volume'].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)

def cmf(df, n=20):
    mf = ((df['close']-df['low']) - (df['high']-df['close'])) / (df['high']-df['low']).replace(0, np.nan)
    mfv = mf * df['volume']
    return mfv.rolling(n).sum() / df['volume'].rolling(n).sum()

def parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    h = df['high'].values; l = df['low'].values
    n = len(df)
    sar = np.zeros(n); trend = np.ones(n, dtype=int)
    af = af_start; ep = h[0]; sar[0] = l[0]
    for i in range(1, n):
        prior = sar[i-1]; tr = trend[i-1]
        if tr == 1:
            sar[i] = prior + af*(ep - prior)
            sar[i] = min(sar[i], l[i-1], l[i-2] if i>=2 else l[i-1])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep; ep = l[i]; af = af_start
            else:
                trend[i] = 1
                if h[i] > ep:
                    ep = h[i]; af = min(af + af_step, af_max)
        else:
            sar[i] = prior + af*(ep - prior)
            sar[i] = max(sar[i], h[i-1], h[i-2] if i>=2 else h[i-1])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep; ep = h[i]; af = af_start
            else:
                trend[i] = -1
                if l[i] < ep:
                    ep = l[i]; af = min(af + af_step, af_max)
    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)

def williams_r(df, n=14):
    hh = df['high'].rolling(n).max()
    ll = df['low'].rolling(n).min()
    return -100 * (hh - df['close']) / (hh - ll).replace(0, np.nan)

def heikin_ashi(df):
    ha_close = (df['open']+df['high']+df['low']+df['close'])/4
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0])/2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1])/2
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df['low'],  ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close
