"""
20 intraday BTCUSDT strategies.  Each function takes the OHLCV DataFrame and
returns a DataFrame with columns: signal (-1/0/+1), stop, tp.

All parameters are TEXTBOOK DEFAULTS fixed before seeing the OOS data — no
optimization is ever performed on any subset of the data.  This is required
for the backtest to be honest.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import (
    ema, sma, atr, rsi, stoch, bollinger, donchian, adx, macd,
    supertrend, keltner, vwap_session, cmf, parabolic_sar, williams_r,
    heikin_ashi,
)


# ---------- helper: assemble the standard output frame ----------

def _pack(df, sig, stop, tp=None):
    out = pd.DataFrame(index=df.index)
    out['signal'] = sig.fillna(0).astype(int) if hasattr(sig, 'fillna') else sig
    out['stop']   = stop
    out['tp']     = tp if tp is not None else np.nan
    return out


# ========================================================================
# 1. EMA 9/21 cross trend follower
# ========================================================================
def s01_ema_cross(df: pd.DataFrame) -> pd.DataFrame:
    fast = ema(df['close'], 9)
    slow = ema(df['close'], 21)
    a = atr(df, 14)
    up_cross = (fast > slow) & (fast.shift() <= slow.shift())
    dn_cross = (fast < slow) & (fast.shift() >= slow.shift())
    sig = pd.Series(0, index=df.index)
    sig[up_cross] = 1; sig[dn_cross] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 2. RSI(2) mean reversion  (Connors style)
# ========================================================================
def s02_rsi2_reversion(df: pd.DataFrame) -> pd.DataFrame:
    r = rsi(df['close'], 2)
    trend = ema(df['close'], 200)
    a = atr(df, 14)
    long_sig  = (r < 10) & (df['close'] > trend)
    short_sig = (r > 90) & (df['close'] < trend)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 2.0*a,
           np.where(sig < 0, df['close'] + 2.0*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 3. Bollinger band breakout
# ========================================================================
def s03_bb_breakout(df: pd.DataFrame) -> pd.DataFrame:
    up, mid, lo = bollinger(df['close'], 20, 2)
    a = atr(df, 14)
    long_sig = (df['close'] > up) & (df['close'].shift() <= up.shift())
    short_sig = (df['close'] < lo) & (df['close'].shift() >= lo.shift())
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 4. Bollinger band fade (mean reversion)
# ========================================================================
def s04_bb_fade(df: pd.DataFrame) -> pd.DataFrame:
    up, mid, lo = bollinger(df['close'], 20, 2)
    a = atr(df, 14)
    long_sig  = (df['close'] < lo)
    short_sig = (df['close'] > up)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    tp = np.where(sig != 0, mid, np.nan)
    return _pack(df, sig, stop, tp)


# ========================================================================
# 5. Donchian 20 breakout (turtle)
# ========================================================================
def s05_donchian_breakout(df: pd.DataFrame) -> pd.DataFrame:
    hh, ll = donchian(df, 20)
    a = atr(df, 14)
    long_sig  = (df['close'] > hh.shift())
    short_sig = (df['close'] < ll.shift())
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 2.0*a,
           np.where(sig < 0, df['close'] + 2.0*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 6. MACD signal cross
# ========================================================================
def s06_macd_cross(df: pd.DataFrame) -> pd.DataFrame:
    line, signal, hist = macd(df['close'], 12, 26, 9)
    a = atr(df, 14)
    up_cross = (line > signal) & (line.shift() <= signal.shift()) & (line < 0)
    dn_cross = (line < signal) & (line.shift() >= signal.shift()) & (line > 0)
    sig = pd.Series(0, index=df.index)
    sig[up_cross] = 1; sig[dn_cross] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 7. Stochastic 14/3/3 oversold/overbought cross
# ========================================================================
def s07_stoch_cross(df: pd.DataFrame) -> pd.DataFrame:
    k, d = stoch(df, 14, 3)
    a = atr(df, 14)
    long_sig  = (k > d) & (k.shift() <= d.shift()) & (d < 20)
    short_sig = (k < d) & (k.shift() >= d.shift()) & (d > 80)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 8. VWAP reversion (anchored daily)
# ========================================================================
def s08_vwap_reversion(df: pd.DataFrame) -> pd.DataFrame:
    v = vwap_session(df)
    a = atr(df, 14)
    dev = (df['close'] - v) / a
    long_sig  = dev < -2.0
    short_sig = dev >  2.0
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    tp = np.where(sig != 0, v, np.nan)
    return _pack(df, sig, stop, tp)


# ========================================================================
# 9. VWAP trend follow (breakouts from VWAP after consolidation)
# ========================================================================
def s09_vwap_breakout(df: pd.DataFrame) -> pd.DataFrame:
    v = vwap_session(df)
    a = atr(df, 14)
    above = df['close'] > v
    sig = pd.Series(0, index=df.index)
    sig[above & ~above.shift(fill_value=False)] = 1
    sig[~above & above.shift(fill_value=False)] = -1
    # filter weak breakouts: require distance > 0.25 ATR
    weak = (df['close']-v).abs() < 0.25*a
    sig[weak] = 0
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 10. Keltner channel breakout
# ========================================================================
def s10_keltner_breakout(df: pd.DataFrame) -> pd.DataFrame:
    up, mid, lo = keltner(df, 20, 2)
    a = atr(df, 14)
    long_sig  = (df['close'] > up) & (df['close'].shift() <= up.shift())
    short_sig = (df['close'] < lo) & (df['close'].shift() >= lo.shift())
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 11. SuperTrend (10, 3.0)
# ========================================================================
def s11_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    st, dir_ = supertrend(df, 10, 3.0)
    a = atr(df, 14)
    flip_up = (dir_ == 1)  & (dir_.shift() == -1)
    flip_dn = (dir_ == -1) & (dir_.shift() == 1)
    sig = pd.Series(0, index=df.index)
    sig[flip_up] = 1; sig[flip_dn] = -1
    stop = np.where(sig != 0, st, np.nan)  # supertrend line itself
    return _pack(df, sig, stop)


# ========================================================================
# 12. Heikin-Ashi trend flip (2-bar confirmation)
# ========================================================================
def s12_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ho, hh, hl, hc = heikin_ashi(df)
    green = (hc > ho).astype(int)
    a = atr(df, 14)
    # Two consecutive greens after a red => long; vice versa
    long_sig  = (green == 1) & (green.shift() == 1) & (green.shift(2) == 0)
    short_sig = (green == 0) & (green.shift() == 0) & (green.shift(2) == 1)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 13. Opening Range Breakout — first 4h of UTC day (timeframe-agnostic)
# ========================================================================
def s13_opening_range(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    hours_from_day = (dt - dt.dt.floor('D')).dt.total_seconds() / 3600.0
    # "Opening range" = first 4 hours of each UTC day
    in_or = hours_from_day < 4.0
    day = dt.dt.floor('D')
    h, l = df['high'], df['low']
    or_high = h.where(in_or).groupby(day).cummax().ffill()
    or_low  = l.where(in_or).groupby(day).cummin().ffill()
    a = atr(df, 14)
    in_window = hours_from_day >= 4.0
    long_sig  = in_window & (df['close'] > or_high) & (df['close'].shift() <= or_high.shift())
    short_sig = in_window & (df['close'] < or_low)  & (df['close'].shift() >= or_low.shift())
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, or_low,
           np.where(sig < 0, or_high, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 14. Parabolic SAR flip
# ========================================================================
def s14_parabolic_sar(df: pd.DataFrame) -> pd.DataFrame:
    sar, trend = parabolic_sar(df)
    a = atr(df, 14)
    flip_up = (trend == 1) & (trend.shift() == -1)
    flip_dn = (trend == -1) & (trend.shift() == 1)
    sig = pd.Series(0, index=df.index)
    sig[flip_up] = 1; sig[flip_dn] = -1
    stop = np.where(sig != 0, sar, np.nan)
    return _pack(df, sig, stop)


# ========================================================================
# 15. ADX trend filter + EMA direction
# ========================================================================
def s15_adx_trend(df: pd.DataFrame) -> pd.DataFrame:
    adx_, pdi, mdi = adx(df, 14)
    e50 = ema(df['close'], 50)
    a = atr(df, 14)
    strong = adx_ > 25
    long_sig  = strong & (df['close'] > e50) & (df['close'].shift() <= e50.shift())
    short_sig = strong & (df['close'] < e50) & (df['close'].shift() >= e50.shift())
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 16. Volume spike fade
# ========================================================================
def s16_volume_fade(df: pd.DataFrame) -> pd.DataFrame:
    v_ma = df['volume'].rolling(20).mean()
    v_sd = df['volume'].rolling(20).std()
    z = (df['volume'] - v_ma) / v_sd
    ret = df['close'].pct_change()
    a = atr(df, 14)
    long_sig  = (z > 3) & (ret < -0.005)
    short_sig = (z > 3) & (ret >  0.005)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 17. Z-score mean reversion of close-vs-EMA
# ========================================================================
def s17_zscore_reversion(df: pd.DataFrame) -> pd.DataFrame:
    m = df['close'].rolling(20).mean()
    s = df['close'].rolling(20).std()
    z = (df['close'] - m) / s
    a = atr(df, 14)
    long_sig  = z < -2
    short_sig = z >  2
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    tp = np.where(sig != 0, m, np.nan)
    return _pack(df, sig, stop, tp)


# ========================================================================
# 18. 3-bar momentum continuation
# ========================================================================
def s18_momentum3(df: pd.DataFrame) -> pd.DataFrame:
    mom = df['close'] / df['close'].shift(3) - 1
    a = atr(df, 14)
    long_sig  = (mom > 0.01) & (df['close'] > ema(df['close'],50))
    short_sig = (mom < -0.01) & (df['close'] < ema(df['close'],50))
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 19. CMF zero cross
# ========================================================================
def s19_cmf_cross(df: pd.DataFrame) -> pd.DataFrame:
    c = cmf(df, 20)
    a = atr(df, 14)
    long_sig  = (c > 0) & (c.shift() <= 0)
    short_sig = (c < 0) & (c.shift() >= 0)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


# ========================================================================
# 20. Williams %R reversal (-80/-20)
# ========================================================================
def s20_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    wr = williams_r(df, 14)
    a = atr(df, 14)
    long_sig  = (wr > -80) & (wr.shift() <= -80)
    short_sig = (wr < -20) & (wr.shift() >= -20)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1; sig[short_sig] = -1
    stop = np.where(sig > 0, df['close'] - 1.5*a,
           np.where(sig < 0, df['close'] + 1.5*a, np.nan))
    return _pack(df, sig, stop)


ALL_STRATEGIES = [
    ('S01_EMA_Cross',         s01_ema_cross),
    ('S02_RSI2_Reversion',    s02_rsi2_reversion),
    ('S03_BB_Breakout',       s03_bb_breakout),
    ('S04_BB_Fade',           s04_bb_fade),
    ('S05_Donchian20',        s05_donchian_breakout),
    ('S06_MACD_Cross',        s06_macd_cross),
    ('S07_Stoch_Cross',       s07_stoch_cross),
    ('S08_VWAP_Reversion',    s08_vwap_reversion),
    ('S09_VWAP_Breakout',     s09_vwap_breakout),
    ('S10_Keltner_Breakout',  s10_keltner_breakout),
    ('S11_SuperTrend',        s11_supertrend),
    ('S12_HeikinAshi',        s12_heikin_ashi),
    ('S13_OpeningRange',      s13_opening_range),
    ('S14_ParabolicSAR',      s14_parabolic_sar),
    ('S15_ADX_Trend',         s15_adx_trend),
    ('S16_Volume_Fade',       s16_volume_fade),
    ('S17_ZScore_Reversion',  s17_zscore_reversion),
    ('S18_Momentum3',         s18_momentum3),
    ('S19_CMF_Cross',         s19_cmf_cross),
    ('S20_Williams_R',        s20_williams_r),
]
