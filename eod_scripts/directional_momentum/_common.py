"""Shared pieces for the directional_momentum EOD builders.

Everything here is a faithful transcription of
`/home/rocky/crypto_sims/PRODUCTION/engines/directional_momentum.py` (`run_cell` lines
48-99 and `build_r_state` lines 28-45) and the constants it imports from
`engines/core/vendored_b1_dmp.py`. PRODUCTION is the source of truth; where a line looks
odd it is because the reference is odd, and the comment says so.

We deliberately do NOT import PRODUCTION directly: it does `import config as cfg`, and
`signal_generation_crypto` has its own `config` PACKAGE that shadows PRODUCTION's
`config.py` on sys.path.

=============================================================================
THE ONE THING TO GET RIGHT: THE IS WINDOW IS SLICED WITH **STRING** LABELS
=============================================================================
DMP:  vendored_b1_dmp.py:15   START, IS_END = '2021-01-01', '2025-03-31'
      directional_momentum.py:25   same, and both are used as `rv.loc[START:IS_END]`.
B1:   vendored_b1_dmp.py:26   ISE = pd.Timestamp('2025-03-31')

A STRING end label makes `.loc` include the WHOLE of 2025-03-31; a Timestamp stops at
00:00:00 that day. Measured on BTCUSDT the two conventions differ by 287 IS candles at
TF=5 (23 at TF=60) and move q_hi by 1.0e-04 — small, and fatal to bitwise parity.

So `is_bounds` below takes the end as a STRING and never converts it, and the CLI refuses
a value carrying a time component. This is the exact opposite of the sibling
eod_scripts/pair_momentum/_common.py, which parses --is-end as a Timestamp because B1
does. Do not share bound-building code between the two sleeves.
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# Universe. PRODUCTION/config.py:93-94 — DMP_ASSETS, 11 single assets (no pairs).
# v3.1 removed AAVE and added DOT at frozen parameters. This is NOT the B1/QR1/QR31
# universe: BNB is a DMP asset but not a B1 pair leg.
# Order matters only for readability; every consumer here is order-invariant (a mean).
# ---------------------------------------------------------------------------------
DMP_ASSETS = ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'BNBUSDT', 'UNIUSDT', 'DOTUSDT',
              'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']

TFS = [5, 15, 30, 60]                                   # config.py:128

# vendored_b1_dmp.py:85 — MED_LB/SPAN_TRAIL/SPAN_ALLOC/ZMED_EMA. NOTE the module also
# binds SPAN_BAND = 14, which prep() uses and run_cell DISCARDS; the live band span is
# config.DMP['SPAN_BAND'] = 10. Not needed here (bounds depend only on the rv span).
SS = 75                                                 # SPAN_ALLOC, the sizing-vol span
RV_SCALE = 100.0                                        # directional_momentum.py:73

# vendored_b1_dmp.py:15 / directional_momentum.py:25. STRINGS — see the module docstring.
IS_START_DEFAULT = '2021-01-01'
IS_END_DEFAULT = '2025-03-31'
Q_LO_DEFAULT, Q_HI_DEFAULT = 0.0, 0.997                 # directional_momentum.py:39-40, 74-75

# Frozen R constants, PRODUCTION/config.py:133. Exposed as CLI defaults, not magic numbers.
# R_CAP is the DMP-only one: `np.minimum(R_CAP, ...)` at directional_momentum.py:58, which
# is why R lives in [0.25, 1.00] here and in [0.25, 1.75] for B1.
# NOTE PRODUCTION has no R_STATE_TF constant — the 15 is a hardcoded literal in the
# `resample('15min')` call at directional_momentum.py:38. params.json invents the name.
R_FLOOR = 0.25
R_SLOPE = 1.5
R_CAP = 1.0
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

    A plain `to_parquet` is not atomic: the live loader stats r_state_daily every minute and
    re-reads it whenever mtime changes, so it can catch a half-written file. Writing to a
    temp name in the SAME directory (so it is the same filesystem) and then os.replace()
    makes the swap atomic — a reader sees either the old file or the new one, never a
    partial one. Also creates the directory on first run.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)


def spliced_ohlc(sym, perp_dir=DATA_PERP, combined_dir=DATA_COMBINED):
    """Minute OHLC for one asset: combined (frozen truth) + PERP tail.

    Transcribed from PRODUCTION/engines/core/data_io.py:17-32. Only PERP minutes STRICTLY
    NEWER than the combined tail are appended. B1 and DMP read this splice; QR1/QR31 read
    the PERP files directly.

    Note what this does NOT do: it does not reindex onto a complete minute grid. Gaps are
    ABSENT LABELS, not NaN rows — which is why `.ffill()` is a no-op on this frame (see
    cell_candles).
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


def spliced_close(sym, perp_dir=DATA_PERP, combined_dir=DATA_COMBINED):
    """data_io.py:31-32."""
    return spliced_ohlc(sym, perp_dir, combined_dir)['Close']


# ---------------------------------------------------------------------------------
# TWO 15-MINUTE SERIES, DELIBERATELY KEPT APART
#
# A TF=15 CELL and the R STATE both look at 15-minute closes, but the reference builds
# them differently:
#
#   cell   run_cell:61 -> resample_ohlc(raw[OHLC].ffill().bfill(), '15T')['close']
#   state  build_r_state:37-38 -> px['Close'].loc[~dup].sort_index().resample('15min').last()
#
# The state path has NO ffill/bfill and never touches OHLC. On the spliced ARCHIVE the two
# happen to coincide bitwise (verified for BTC/ADA/DOGE: identical index, zero NaN, zero
# differing closes, and identical daily state) precisely because spliced_ohlc leaves gaps as
# absent labels, so the fill is a no-op. They would NOT coincide on a frame that was
# reindexed onto a complete minute grid — which is what the live path produces.
#
# So both are transcribed here, from their own reference lines, and
# build_rv_alloc_bounds.py ASSERTS the coincidence rather than assuming it. If a future
# data change breaks it, that assert fires instead of the R state silently drifting.
# ---------------------------------------------------------------------------------
def cell_candles(raw, tf):
    """The (asset, TF) candle frame. run_cell:61-63 via vendored_b1_dmp.resample_ohlc:97-100."""
    df = raw[_COLS].ffill().bfill()
    df = df.loc[~df.index.duplicated(), :].sort_index()
    # 'min' not 'T': pandas deprecated the 'T' alias. Same offset, no behaviour change.
    df = df.resample(f'{tf}min').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df = df.loc[~df.index.duplicated(), :]
    return df.rename(columns={'Open': 'open', 'High': 'high',
                              'Low': 'low', 'Close': 'close'})


def state_close_15m(close, tf=R_STATE_TF):
    """The R state's 15-minute close series. build_r_state:37 VERBATIM:
        c = px.loc[~px.index.duplicated()].sort_index().resample('15min').last()
    No ffill, no bfill, no OHLC.
    """
    return close.loc[~close.index.duplicated()].sort_index().resample(f'{tf}min').last()


def rv_raw(close, span=SS, scale=RV_SCALE):
    """Unclipped sizing vol from a candle CLOSE series.
    run_cell:73 / build_r_state:38 — the same expression on both paths.
    """
    return np.log(close).diff().ewm(span=span).std() * scale


def is_bounds(rv, is_start, is_end, q_lo, q_hi):
    """The frozen clip bounds = the FINAL value of the expanding IS quantiles.

    run_cell:74-75 computes an expanding quantile over the IS slice and then
    `.reindex(df.index).ffill()`, so every bar at or after the IS end sees the last IS
    value — a constant. That constant is what we freeze.

    `is_start` / `is_end` are pandas .loc LABELS and are passed through as given. Keep them
    STRINGS: see the module docstring. Returns (q_lo_value, q_hi_value, n_is).
    """
    _check_label(is_end)
    seg = rv.loc[is_start:is_end]
    n_is = int(len(seg))
    if n_is == 0:
        raise ValueError(f'empty IS window [{is_start}, {is_end}]')
    lo = float(seg.expanding().quantile(q_lo).iloc[-1])
    hi = float(seg.expanding().quantile(q_hi).iloc[-1])
    return lo, hi, n_is


def _check_label(is_end):
    """Refuse an IS end that has been turned into a timestamp or given a time component.

    Both mistakes silently shrink the window to 00:00:00 of that day — the B1 convention —
    and move q_hi by ~1e-4. Cheap to catch here; very expensive to notice downstream.
    """
    if isinstance(is_end, pd.Timestamp):
        raise TypeError(
            f'is_end must be a STRING label for DMP, got a Timestamp ({is_end!r}). '
            f'`rv.loc[start:Timestamp("2025-03-31")]` stops at 00:00:00 and drops the rest '
            f'of the day — that is B1_v8_v10\'s convention, not DMP_v3_2\'s.')
    if not isinstance(is_end, str):
        raise TypeError(f'is_end must be a string label, got {type(is_end).__name__}')
    if any(ch in is_end for ch in (' ', 'T', ':')):
        raise ValueError(
            f'is_end {is_end!r} carries a time component. DMP freezes on the bare date '
            f'string {IS_END_DEFAULT!r}, which includes the whole day.')


def clip_frozen(rv, q_lo, q_hi):
    """Clip with the frozen constants from rv_alloc_bounds.parquet.

    Correct for any bar AT OR AFTER the IS window end, which is every bar production will
    ever process — but NOT equivalent to PRODUCTION inside or before the IS window. Use
    clip_expanding when reproducing history.
    """
    return np.minimum(np.maximum(q_lo, rv), q_hi)


def clip_expanding(rv, is_start, is_end, q_lo_q, q_hi_q):
    """PRODUCTION's clip, run_cell:74-76 / build_r_state:39-41 verbatim.

    The behaviour that matters, and that clip_frozen does NOT reproduce:
      * BEFORE is_start the bounds are NaN (nothing to ffill from), so rv_alloc is NaN.
        On the R-state path that NaN then drops the day out of the daily mean entirely
        (`.dropna()` at build_r_state:42), so pre-2021 days never enter the expanding
        percentile's comparison set at all — the OPPOSITE of B1, where a minute-level
        bfill floods ~367 flat days into it forever. DMP takes its daily mean of the
        CANDLE series, with no minute broadcast and therefore no bfill.
      * INSIDE the IS window the bounds GROW bar by bar.
      * AFTER is_end they are frozen at the final IS value == the constants clip_frozen uses.
    """
    _check_label(is_end)
    rlo = rv.loc[is_start:is_end].expanding().quantile(q_lo_q).reindex(rv.index).ffill()
    rhi = rv.loc[is_start:is_end].expanding().quantile(q_hi_q).reindex(rv.index).ffill()
    return np.minimum(np.maximum(rlo, rv), rhi)


def state_daily_from_assets(per_asset_clipped_rv):
    """V = cross-asset daily mean. build_r_state:41-42.

        vols.append(<clipped rv>.resample('D').mean())
        V = pd.concat(vols, axis=1).mean(axis=1).dropna()

    NOTE the daily mean is taken over the 15-MINUTE CANDLE series (96 obs/day), NOT over a
    minute-broadcast series. That is the sharpest difference from B1's R state, which means
    over minutes because its `rvalloc` array is the projected minute series. Reusing
    pair_momentum's `rv_alloc_minutes` here would be wrong.

    `mean(axis=1)` is skipna — a day is averaged over whatever assets have data — and the
    trailing `.dropna()` only removes days where ALL 11 are NaN.
    """
    dailies = [s.resample('D').mean() for s in per_asset_clipped_rv]
    return pd.concat(dailies, axis=1).mean(axis=1).dropna()


def r_from_state(V, r_floor, r_slope, r_cap, med_win, pct_minp, lag_days):
    """V -> R, indexed by the day the value APPLIES TO. build_r_state:43-45 + run_cell:58.

    Four things a naive rewrite gets wrong:
      * rolling(med_win) uses the DEFAULT min_periods (= med_win), trailing.
      * the expanding percentile EXCLUDES the current observation from its own comparison
        set (`x[:-1]`), so the denominator is len(x)-1.
      * the CAP: `np.minimum(R_CAP, floor + slope*(1-pct))`. DMP-only. Without it R would
        reach 1.75 and every calm-regime SHORT entry would be sized ~75% too large.
      * the lag: PRODUCTION writes `.shift(lag_days)`.

    WHY NOT `.shift()`: it moves VALUES into index positions that already exist, so the
    value computed from the newest day is silently dropped — there is no row for
    `newest + 1` because that day is not complete yet and so never entered V. Live trading
    on day D then finds no row for D, ffills to D-1, and sizes off a state computed through
    D-2: a TWO-day lag where the backtest uses one. Re-running the builder cannot fix it;
    it recomputes and drops the same value again.

    Shifting the INDEX instead is identical on a contiguous daily index (idx[i] == idx[i-1]
    + 1 day, so `R[idx[i]] = f(idx[i-1])` either way) — historical values, and the bitwise
    match against PRODUCTION, are unchanged — but it also emits that final row. It is
    additionally the correct reading when a day is missing (the completeness guard can skip
    one): `.shift()` would carry a value across the gap, whereas "computed from d, applies
    on d + lag" stays true. build_r_state.py --compare-to checks this against a
    literal-`.shift()` reference so a non-contiguous V cannot slip through unnoticed.
    """
    ewma_norm = (V / V.rolling(med_win).median()).dropna()
    pct = ewma_norm.expanding(min_periods=pct_minp).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    r = np.minimum(r_cap, r_floor + r_slope * (1 - pct)).dropna()
    r.index = r.index + pd.Timedelta(days=lag_days)      # stamp with the day it applies to
    return r
