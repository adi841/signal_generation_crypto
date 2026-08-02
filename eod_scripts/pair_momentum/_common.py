"""Shared pieces for the pair_momentum EOD builders.

Everything here is a faithful transcription of
`/home/rocky/crypto_sims/PRODUCTION/engines/pairs_momentum.py::prep()` (lines 30-83) and
the constants it imports from `engines/core/vendored_b1_dmp.py`. PRODUCTION is the source
of truth; where a line looks odd it is because the reference is odd, and the comment says so.

We deliberately do NOT import PRODUCTION directly: it does `import config as cfg`, and
`signal_generation_crypto` has its own `config` PACKAGE that shadows PRODUCTION's
`config.py` on sys.path. That collision already bit the test-1 comparison harness.
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# Universe. PRODUCTION/config.py:88,92 —
#   _A8 = ['SOLUSDT', ...]
#   B1_PAIRS = [(l, a) for a in _A8 for l in ('BTCUSDT', 'ETHUSDT')] + [('BTCUSDT','ETHUSDT')]
# Order matters only for readability; every consumer here is order-invariant (a mean).
# BNB is in _A7/_A9 but NOT _A8, so it is not a B1 pair.
# ---------------------------------------------------------------------------------
B1_ALTS = ['SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
           'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT']
B1_MAJORS = ('BTCUSDT', 'ETHUSDT')
B1_PAIRS = [(l, a) for a in B1_ALTS for l in B1_MAJORS] + [('BTCUSDT', 'ETHUSDT')]
B1_SYMBOLS = sorted({s for pair in B1_PAIRS for s in pair})      # 10 distinct legs

# vendored_b1_dmp.py:18 — SS is the rv (sizing-vol) EWM span; the *100.0 is at
# pairs_momentum.py:55.
SS = 75
RV_SCALE = 100.0

# vendored_b1_dmp.py:26 — the IS window the clip quantiles are fit over. ISE is a
# TIMESTAMP, so `rv.loc[ST:ISE]` stops at 2025-03-31 00:00:00 and excludes the rest of
# that day. Exposed as CLI defaults rather than hardcoded, but these are the frozen values.
IS_START_DEFAULT = '2021'
IS_END_DEFAULT = '2025-03-31'
Q_LO_DEFAULT, Q_HI_DEFAULT = 0.0, 0.997

# Frozen R constants, PRODUCTION/config.py:120. Exposed as CLI defaults, not magic numbers.
# NOTE: PRODUCTION has no R_STATE_TF constant -- the 15 is a hardcoded literal in the
# `prep(l1, l2, 15)` call at pairs_momentum.py:94. params.json invents the name.
R_FLOOR = 0.25
R_SLOPE = 1.5
R_LAG_DAYS = 1
R_MED_WIN = 60
R_PCT_MINPERIODS = 120
R_STATE_TF = 15

# config.py:63-64
DATA_PERP = '/home/rocky/crypto_sims/crypto_data/'
DATA_COMBINED = '/home/rocky/binance/data/combined/'
_COLS = ['Open', 'High', 'Low', 'Close']                          # data_io.py:14

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def write_parquet_atomic(df, path, **kwargs):
    """Write a parquet the live path can poll safely.

    A plain `to_parquet` is not atomic: the live loader stats this file every minute and
    re-reads it whenever mtime changes, so it can catch a half-written file. Writing to a
    temp name in the SAME directory (so it is the same filesystem) and then os.replace()
    makes the swap atomic -- a reader sees either the old file or the new one, never a
    partial one. Also creates the directory on first run.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)


def spliced_ohlc(sym, perp_dir=DATA_PERP, combined_dir=DATA_COMBINED):
    """Minute OHLC for one asset: combined (frozen truth) + PERP tail.

    Transcribed from PRODUCTION/engines/core/data_io.py:17-32. Only PERP minutes STRICTLY
    NEWER than the combined tail are appended, which is why every series effectively
    starts 2019-12-31 even though the PERP files reach back to 2019-09.
    """
    p = pd.read_parquet(f'{perp_dir}{sym}PERP-1m-data.parquet', columns=_COLS)
    cf = f'{combined_dir}{sym}-1m-combined.parquet'
    if not os.path.exists(cf):
        return p
    c = pd.read_parquet(cf, columns=_COLS)
    tail = p[p.index > c.index[-1]]
    if len(tail):
        gap = tail.index[0] - c.index[-1]
        assert gap <= pd.Timedelta(minutes=90), f'splice gap {sym}: {gap}'
    return pd.concat([c, tail])


def ratio_minute_frame(d1, d2, a1, a2):
    """The minute-level ratio frame, i.e. pairs_momentum.py:35-46.

    Two details that are load-bearing and easy to get wrong:
      * the ratio is formed on the OUTER-merged frame, so it is NaN wherever EITHER leg
        is missing. The ffill/bfill happens later (in ratio_candles), which is what
        back-fills alt prices to before those alts existed.
      * minute high/low are max/min(open, close) -- NOT true intrabar extremes.
    """
    d1 = d1.rename(columns=str.lower)
    d2 = d2.rename(columns=str.lower)
    d1 = d1[['open', 'close']].add_prefix(a1)
    d2 = d2[['open', 'close']].add_prefix(a2)
    raw = d1.merge(d2, left_index=True, right_index=True, how='outer')

    raw['Open'] = raw[a1 + 'open'] / raw[a2 + 'open']
    raw['Close'] = raw[a1 + 'close'] / raw[a2 + 'close']
    raw['High'] = np.max(raw[['Open', 'Close']].values, axis=1)
    raw['Low'] = np.min(raw[['Open', 'Close']].values, axis=1)
    return raw[_COLS]


def ratio_candles(raw, tf):
    """Resample the minute ratio frame onto the TF grid.

    pairs_momentum.py:48-50 via vendored_pairs.py:70-80. The `.ffill().bfill()` is applied
    to the MINUTE frame before resampling, not after.
    """
    # 'min' not 'T': pandas deprecated the 'T' alias. Same offset, no behaviour change.
    df = raw.ffill().bfill().resample(f'{tf}min').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df = df.loc[~df.index.duplicated(), :]
    return df.rename(columns={'Open': 'open', 'High': 'high',
                              'Low': 'low', 'Close': 'close'})


def rv_raw(candles, span=SS, scale=RV_SCALE):
    """Unclipped sizing vol: pairs_momentum.py:53,55."""
    lr = np.log(candles['close']).diff()
    return lr.ewm(span=span).std() * scale


def is_bounds(rv, is_start, is_end, q_lo, q_hi):
    """The frozen clip bounds = the FINAL value of the expanding IS quantiles.

    pairs_momentum.py:57-58 computes an expanding quantile over the IS slice and then
    `.reindex(df.index).ffill()`, so every bar at or after the IS end sees the last IS
    value -- a constant. That constant is what we freeze.

    Returns (q_lo_value, q_hi_value, n_is). `is_end` is compared as a Timestamp, so a bare
    date means 00:00:00 that day; see IS_END_DEFAULT.
    """
    seg = rv.loc[is_start:is_end]
    n_is = int(len(seg))
    if n_is == 0:
        raise ValueError(f'empty IS window [{is_start}, {is_end}]')
    lo = float(seg.expanding().quantile(q_lo).iloc[-1])
    hi = float(seg.expanding().quantile(q_hi).iloc[-1])
    return lo, hi, n_is


def clip_frozen(rv, q_lo, q_hi):
    """Clip with the frozen constants from rv_alloc_bounds.parquet.

    Correct for any bar AT OR AFTER the IS window end, which is every bar production will
    ever process -- but NOT equivalent to PRODUCTION inside or before the IS window. Use
    clip_expanding when reproducing history.
    """
    return np.minimum(np.maximum(q_lo, rv), q_hi)


def clip_expanding(rv, is_start, is_end, q_lo_q, q_hi_q):
    """PRODUCTION's clip, pairs_momentum.py:57-59 verbatim.

    The behaviour that matters, and that clip_frozen does NOT reproduce:
      * BEFORE is_start the bounds are NaN (nothing to ffill from), so rv_alloc is NaN --
        and the later minute-level .bfill() then floods all of pre-history with the first
        valid value. That is what creates the ~367 flat days which sit in R's expanding
        percentile comparison set forever.
      * INSIDE the IS window the bounds GROW bar by bar.
      * AFTER is_end they are frozen at the final IS value == the constants clip_frozen uses.
    """
    rlo = rv.loc[is_start:is_end].expanding().quantile(q_lo_q).reindex(rv.index).ffill()
    rhi = rv.loc[is_start:is_end].expanding().quantile(q_hi_q).reindex(rv.index).ffill()
    return np.minimum(np.maximum(rlo, rv), rhi)


def rv_alloc_minutes(raw, tf, clip_fn):
    """Clipped rv_alloc broadcast onto the MINUTE timeline -- what R's state is built from.

    `clip_fn(rv) -> clipped rv` is either clip_frozen or clip_expanding, partially applied.

    This is the subtle one. `build_r_state` takes a daily mean of `d['rvalloc']`, and that
    array is the MINUTE-broadcast series (pairs_momentum.py:82), not the TF candle series.
    Reproducing it needs pairs_momentum.py:63-71 exactly:
        mc index  = OUTER union of the minute index and the candle index
        feats     = .shift(tf - 1)      <- step boundaries land at :29/:44/:59/:14 for tf=15
        then        .ffill().bfill()    <- the bfill is what floods pre-history (see
                                           clip_expanding)
    """
    candles = ratio_candles(raw, tf)
    candles = candles.assign(rv_alloc=clip_fn(rv_raw(candles)))

    mc = raw[['Close']].copy()
    mc = mc.merge(candles[['rv_alloc']], left_index=True, right_index=True, how='outer')
    mc['rv_alloc'] = mc['rv_alloc'].shift(tf - 1).ffill().bfill()
    return mc['rv_alloc']


def daily_state_from_pairs(per_pair_minute_rv):
    """V = cross-pair daily mean. pairs_momentum.py:95-96.

    Per pair: resample('D').mean() over MINUTES. Then concat across pairs and mean(axis=1),
    which is skipna -- a day is averaged over whatever pairs have data, and only dropped
    when ALL are NaN.
    """
    dailies = [s.resample('D').mean() for s in per_pair_minute_rv]
    return pd.concat(dailies, axis=1).mean(axis=1).dropna()


def r_from_state(V, r_floor, r_slope, med_win, pct_minp, lag_days):
    """V -> R, indexed by the day the value APPLIES TO. pairs_momentum.py:97-100.

    Three things a naive rewrite gets wrong:
      * rolling(med_win) uses the DEFAULT min_periods (= med_win), trailing.
      * the expanding percentile EXCLUDES the current observation from its own comparison
        set (`x[:-1]`), so the denominator is len(x)-1.
      * the lag: PRODUCTION writes `.shift(lag_days)`.
    No clip: [0.25, 1.75] is emergent because pct is in [0, 1].

    WHY NOT `.shift()`: it moves VALUES into index positions that already exist, so the
    value computed from the newest day is silently dropped -- there is no row for
    `newest + 1` because that day is not complete yet and so never entered V. Live trading
    on day D then finds no row for D, ffills to D-1, and sizes off a state computed through
    D-2: a TWO-day lag where the backtest uses one. Re-running the builder cannot fix it;
    it recomputes and drops the same value again.

    Shifting the INDEX instead is identical on a contiguous daily index (idx[i] == idx[i-1]
    + 1 day, so `R[idx[i]] = f(idx[i-1])` either way) -- historical values, and the bitwise
    match against PRODUCTION, are unchanged -- but it also emits that final row. It is
    additionally the correct reading when a day is missing (the completeness guard can skip
    one): `.shift()` would carry a value across the gap, whereas "computed from d, applies
    on d + lag" stays true.
    """
    ewma_norm = (V / V.rolling(med_win).median()).dropna()
    pct = ewma_norm.expanding(min_periods=pct_minp).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    r = (r_floor + r_slope * (1 - pct)).dropna()
    r.index = r.index + pd.Timedelta(days=lag_days)      # stamp with the day it applies to
    return r
