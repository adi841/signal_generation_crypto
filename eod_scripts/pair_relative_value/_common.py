"""Shared pieces for the pair_relative_value (QR1_v4) EOD builders.

Everything here is a faithful transcription of
`/home/rocky/crypto_sims/PRODUCTION/engines/pairs_relvalue.py` (run_cell/build_expo),
`engines/core/vendored_qr1.py` (load_raw/prep_cell constants) and
`engines/pairs_leadlag.py::build_ewma_norm`. PRODUCTION is the source of truth; where a
line looks odd it is because the reference is odd, and the comment says so.

We deliberately do NOT import PRODUCTION directly: it does `import config as cfg`, and
`signal_generation_crypto` has its own `config` PACKAGE that shadows PRODUCTION's
`config.py` on sys.path (the collision that bit the earlier comparison harnesses).

DATA SOURCE — the key delta vs the pair_momentum builders: QR1 reads the PERP files
DIRECTLY (`vendored_qr1.load_raw`; data_io.py's docstring: "the pairs strategies load
through their own vendored load_raw functions, which read the PERP files directly").
The spliced combined archive is a B1/DMP convention and must NOT be used here.
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# Universe. PRODUCTION/config.py:86,90 —
#   _A7 = ['SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'BNBUSDT']
#   QR1_PAIRS = [('BTCUSDT', a) for a in _A7] + [('ETHUSDT', a) for a in _A7]
#               + [('BTCUSDT', 'ETHUSDT')]
# BNB is IN (unlike B1); DOT/UNI are OUT. Order matches config (consumers here are
# order-invariant, but the artifact rows keep it for readability).
# ---------------------------------------------------------------------------------
QR1_ALTS = ['SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'BNBUSDT']
QR1_PAIRS = ([('BTCUSDT', a) for a in QR1_ALTS] + [('ETHUSDT', a) for a in QR1_ALTS]
             + [('BTCUSDT', 'ETHUSDT')])
QR1_SYMBOLS = sorted({s for pair in QR1_PAIRS for s in pair})     # 9 distinct legs

TFS = [15, 30, 60, 120, 240]                                      # config.py:160

# Per-TF band width, vendored_qr1.py:58-64 (QR1_CFG.nbdev) — needed for f_bandwp/wmed.
NBDEV_PER_TF = {15: 3.5, 30: 3.0, 60: 2.0, 120: 3.5, 240: 2.0}

# prep_cell/run_cell spans. CENTRE_SPAN = d_timeperiod['8'][0] (vendored_qr1.py:39,141);
# ATR lens = d_atr_list['2']/['10'] + the atr3 loop's [75] (vendored_qr1.py:132-139);
# EV_SPAN = config QR1.EV_SPAN (config.py:164); EV75 is run_cell's legacy scale anchor
# (pairs_relvalue.py:79).
CENTRE_SPAN = 10
ATR_FAST, ATR_SLOW, ATR_ALLOC = 14, 50, 75
EV_SPAN, EV_LEGACY_SPAN = 10, 75

# The fixed CALIBRATION WINDOW (vendored_qr1.py:16). IS_END is a TIMESTAMP: `.loc['2021':
# '2025-03-31']` stops at 2025-03-31 00:00:00. Frozen; CLI defaults, not magic numbers.
IS_START_DEFAULT = '2021'
IS_END_DEFAULT = '2025-03-31'
CALIBRATION_END = '2025-04-01'                                    # config.py:175

# EXPO overlay constants, config.py:169 + pairs_relvalue.py:42-52 + pairs_leadlag.py:29-42.
EXPO_FLOOR, EXPO_SLOPE, EXPO_WIN = 0.5, 1.0, 20
EXPO_STATE_SPAN = 10          # daily-vol EWMA span (build_ewma_norm)
EXPO_MED_WIN = 60             # rolling median normalizer (build_ewma_norm)
# The frozen 10-symbol basket (pairs_leadlag.py STATE_SYMS): DOT IN, UNI OUT —
# deliberately NOT the traded universe.
STATE_SYMS = ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'BNBUSDT', 'DOTUSDT',
              'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------
# DATA SOURCE: tsdb `ohlcv_data`, NOT the PERP parquet archive.
#
# The reference book was built on `{SYM}PERP-1m-data.parquet` under a crypto_sims path
# that no longer exists on the live host. The database is now the source of truth. That
# substitution was verified, not assumed: over the compared windows the two agree
# BITWISE on every OHLC cell they share --
#
#   BTCUSDT  2021-01-01..2021-06-30   259,201 common minutes   bitwise equal
#   AVAXUSDT 2024-01-01..2024-03-31   129,601 common minutes   bitwise equal
#   DOGEUSDT 2025-01-01..2025-03-31   128,161 common minutes   bitwise equal
#
# and ohlcv_data reaches back further than the 2021 IS window for every basket leg
# (earliest: BTC 2019-09-08, latest first-bar: AVAX 2020-09-23), so no frozen constant
# moves as a result of the swap.
#
# One trap worth keeping in mind when comparing the two by hand: pandas PARTIAL-STRING
# slicing (`df.loc['2021':'2025-03-31']`) includes the WHOLE end day, whereas
# `start_time <= '2025-03-31'` stops at 00:00. That is a 1,439-minute difference, and it
# is a slicing artifact, not missing data.
# ---------------------------------------------------------------------------------
def tsdb_symbol(sym):
    """'BTCUSDT' -> 'BTC-USDT.PERP' (ohlcv_data stores the internal name)."""
    if not sym.endswith('USDT'):
        raise ValueError(f'unexpected symbol form: {sym}')
    return f'{sym[:-4]}-USDT.PERP'


def connect_tsdb(host=None, port=None, user=None, database=None):
    """Direct postgres on 5432 (not pgbouncer's transaction-pooled 6432); the
    config_utils import is guarded (it git-fetches at import time) with PG* env
    fallbacks."""
    import psycopg2
    pw = os.environ.get('POSTGRES_PASSWORD')
    try:
        ## These scripts put their OWN directory on sys.path (for `_common`), not the
        ## repo root, so `shared_codes` was never importable and this always fell through
        ## to the PG* branch — which defaults to 127.0.0.1. That is invisible while the
        ## database is local and wrong the moment it is not. Put the repo root on the
        ## path so get_db_params actually gets a chance to answer.
        import sys as _sys
        _repo = os.path.dirname(os.path.dirname(ARTIFACT_DIR))
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from shared_codes.utils.config_utils import get_db_params
        p = get_db_params()
        host = host or p.host
        user = user or p.user
        database = database or p.database
        pw = pw or p.password
    except Exception as e:
        print(f'  [info] get_db_params unavailable ({type(e).__name__}), using PG* env vars')
        host = host or os.environ.get('PGHOST', '127.0.0.1')
        user = user or os.environ.get('PGUSER', 'tsdbadmin')
        database = database or os.environ.get('PGDATABASE', 'tsdb')
    if pw is None:
        raise RuntimeError('POSTGRES_PASSWORD is not set and get_db_params did not supply one')
    port = port or 5432
    return psycopg2.connect(user=user, password=pw, host=host, port=port, database=database)


def load_minute_ohlc(sym, conn=None, start=None, end=None):
    """Minute OHLC for one asset from ohlcv_data — the archive-shaped replacement for
    the old `load_perp` (vendored_qr1.load_raw:70-71).

    Returns EXACTLY what the parquet read returned, so nothing downstream changes:
    columns ['Open','High','Low','Close'] (capitalised), float64, tz-NAIVE UTC index,
    sorted and de-duplicated. Naive on purpose — every consumer here slices with plain
    date strings ('2021', '2025-03-31'), which a tz-aware index would still accept but
    with different boundary semantics.

    `conn=None` opens (and closes) its own connection, so this is safe to call from
    multiprocessing workers, which cannot inherit a live psycopg2 socket.
    """
    own = conn is None
    if own:
        conn = connect_tsdb()
    try:
        cur = conn.cursor()          # plain tuple cursor: ~3.6M rows per symbol
        q = ('SELECT start_time, open, high, low, close FROM ohlcv_data '
             'WHERE symbol = %s')
        params = [tsdb_symbol(sym)]
        if start is not None:
            q += ' AND start_time >= %s'; params.append(str(start))
        if end is not None:
            q += ' AND start_time <= %s'; params.append(str(end))
        q += ' ORDER BY start_time'
        cur.execute(q, params)
        rows = cur.fetchall()
        cur.close()
    finally:
        if own:
            conn.close()
    if not rows:
        raise ValueError(f'ohlcv_data returned no rows for {sym} ({tsdb_symbol(sym)})')
    df = pd.DataFrame(rows, columns=['ts', 'Open', 'High', 'Low', 'Close'])
    idx = pd.to_datetime(df.pop('ts'), utc=True).dt.tz_localize(None)
    df = df.astype('float64')
    df.index = idx
    return df[~df.index.duplicated(keep='last')].sort_index()


def resolve_bundle_dir():
    """Where the built artifacts get published, per the S3 client config.

    The old hardcoded '/home/rocky/crypto_sims/pair_relative_value_production/data' is
    gone along with the rest of that tree. sleeve_config is the single source of truth
    for where LIVE reads its artifacts, so derive the publish target from the same place
    rather than keeping a second copy of the path here that can silently disagree.
    Returns None (caller warns and skips the copy) if the config is unreachable — the
    primary artifacts are already written by then, so a missing publish target must not
    fail the whole build.
    """
    try:
        import sys as _sys
        _repo = os.path.dirname(os.path.dirname(ARTIFACT_DIR))
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from config.config_read import config_object
        return os.path.dirname(config_object.sleeve_config['pair_relative_value']
                               ['expo_daily_path'])
    except Exception as e:
        print(f'  [warn] could not resolve bundle dir from sleeve_config '
              f'({type(e).__name__}: {e})')
        return None


def ratio_minute_frame(d1, d2, a1, a2):
    """The minute-level ratio frame — vendored_qr1.load_raw:69-87 (fix=True).

    Load-bearing details:
      * legs are OUTER-merged and the frame is trimmed to the COMMON start
        (`fix=True`: max of the two first-valid closes);
      * the ratio is formed on the raw merge, so it is NaN wherever either leg is
        missing; the ffill/bfill happens later (in ratio_candles / prep);
      * minute high/low are max/min(Open, Close) — NOT true intrabar extremes.
    """
    x1 = d1[['Open', 'Close']].add_prefix(a1)
    x2 = d2[['Open', 'Close']].add_prefix(a2)
    raw = x1.merge(x2, left_index=True, right_index=True, how='outer')
    start = max(raw[a1 + 'Close'].first_valid_index(), raw[a2 + 'Close'].first_valid_index())
    raw = raw.loc[start:]
    raw['Open'] = raw[a1 + 'Open'] / raw[a2 + 'Open']
    raw['Close'] = raw[a1 + 'Close'] / raw[a2 + 'Close']
    raw['High'] = np.max(raw[['Open', 'Close']].values, axis=1)
    raw['Low'] = np.min(raw[['Open', 'Close']].values, axis=1)
    return raw[['Open', 'High', 'Low', 'Close']]


def ratio_candles(raw, tf):
    """prep_cell:93-100 — ffill/bfill the MINUTE frame, then close-based resample."""
    df = raw.ffill().bfill().resample(f'{tf}min').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df = df.loc[~df.index.duplicated(), :]
    return df.rename(columns={'Open': 'open', 'High': 'high',
                              'Low': 'low', 'Close': 'close'})


def wilder_atr_pct(candles, n):
    """Wilder ATR as % of close — vendored_pairs.ATR (skipna TR max; wwma alpha=1/n,
    adjust=False) with prep_cell's /close*100 scaling."""
    prev = candles['close'].shift(1)
    tr = pd.concat([(candles['high'] - candles['low']).abs(),
                    (candles['high'] - prev).abs(),
                    (candles['low'] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean() / candles['close'] * 100.0


def is_minmax_bounds(series_pct, is_start, is_end):
    """The frozen ATR %-clip bounds = the FINAL values of clip_atr_pct's expanding
    IS quantile(0)/quantile(1), i.e. the IS-window min/max (vendored_qr1.py:40-43).
    Constant for every bar at or after the IS end. Returns (lo, hi, n_is)."""
    seg = series_pct.loc[is_start:is_end]
    n_is = int(len(seg))
    if n_is == 0:
        raise ValueError(f'empty IS window [{is_start}, {is_end}]')
    return float(seg.min()), float(seg.max()), n_is


def clip_frozen(series, lo, hi):
    """pandas-clip semantics with the frozen constants: NaN passes through.

    Correct for any bar AT OR AFTER the IS end — every bar production will ever
    process — but NOT inside the IS window. Use clip_expanding_is when reproducing
    IS-window statistics (kv / wmed)."""
    return series.clip(lower=lo, upper=hi)


def clip_expanding_is(series_pct, is_start, is_end):
    """PRODUCTION's clip_atr_pct VERBATIM (vendored_qr1.py:40-43): expanding IS
    quantile(0.0)/quantile(1.0) (= running min/max), reindexed + ffilled.

    The behaviour that matters, and that clip_frozen does NOT reproduce:
      * BEFORE is_start the bounds are NaN (nothing to ffill from) -> clipped NaN;
      * INSIDE the IS window the bounds GROW bar by bar — kv and wmed are IS-window
        statistics, so this is load-bearing for them;
      * AFTER is_end they are frozen at the final IS value == clip_frozen's constants.
    """
    lo = series_pct.loc[is_start:is_end].expanding().quantile(0.0).reindex(series_pct.index).ffill()
    hi = series_pct.loc[is_start:is_end].expanding().quantile(1.0).reindex(series_pct.index).ffill()
    return np.minimum(np.maximum(lo, series_pct), hi)


def project_to_minutes(minute_index, candle_series, tf):
    """Candle series -> minute grid, prep_cell:188-201 semantics: OUTER union of the
    minute index and the candle labels (the merge injects bucket-start minutes absent
    from the minute frame), POSITIONAL shift(tf-1), then ffill/bfill. A candle's value
    lands on its own closing minute — the candle-fresh convention."""
    union = minute_index.union(candle_series.index)
    return candle_series.reindex(union).shift(tf - 1).ffill().bfill()


def kv_wmed_for_cell(raw, tf, nbdev, is_start, is_end):
    """The two per-cell IS statistics + the three ATR bounds, for one (pair, TF).

    kv: the EV10 sizing scale, run_cell:77-83 VERBATIM chain (k_ through EV75 with the
        max(.., 1e-9) guards — algebraically IS-mean(clipped ATR75%)/IS-mean(EV10), but
        the two-step form is what the reference computes, so keep it bitwise).
    wmed: the wide-band gate median — the IS median of the MINUTE-PROJECTED band-width %
        (run_cell:96-97 takes nanmedian over the minute grid, not the candle grid).
    """
    candles = ratio_candles(raw, tf)
    minute_index = raw.index

    a14_raw = wilder_atr_pct(candles, ATR_FAST)
    a50_raw = wilder_atr_pct(candles, ATR_SLOW)
    a75_raw = wilder_atr_pct(candles, ATR_ALLOC)
    a14_lo, a14_hi, n_is = is_minmax_bounds(a14_raw, is_start, is_end)
    a50_lo, a50_hi, _ = is_minmax_bounds(a50_raw, is_start, is_end)
    a75_lo, a75_hi, _ = is_minmax_bounds(a75_raw, is_start, is_end)

    # The IS statistics below must see PRODUCTION's EXPANDING clip (bounds grow
    # through the IS window), not the frozen constants — that difference moves kv by
    # ~1e-4 and wmed by ~5e-4, which the --compare-to check against the harness-built
    # cells catches. The frozen bounds in the artifact are for LIVE bars (post-IS).
    a14 = clip_expanding_is(a14_raw, is_start, is_end)
    a50 = clip_expanding_is(a50_raw, is_start, is_end)
    # f_atr75p (prep_cell:134-139) round-trips through PRICE units (pct*close/100,
    # then /close*100) — 1-2 ulp, kept for bitwise parity.
    a75 = (clip_expanding_is(a75_raw, is_start, is_end)
           * candles['close'] / 100.0) / candles['close'] * 100.0

    # bands in PRICE units (prep_cell:141-153, constr='min'), then band-width %
    mid = candles['close'].ewm(span=CENTRE_SPAN).mean()            # adjust=True
    p14 = a14 * candles['close'] / 100.0
    p50 = a50 * candles['close'] / 100.0
    s_upper = pd.concat([mid + nbdev * p14, mid + nbdev * p50], axis=1).min(axis=1)
    f_bandwp = (s_upper - mid) / candles['close'] * 100.0          # prep_cell:172

    lr = np.log(candles['close']).diff()
    ev75 = lr.ewm(span=EV_LEGACY_SPAN).std() * 100.0               # run_cell:77
    ev10 = lr.ewm(span=EV_SPAN).std() * 100.0                      # run_cell:81

    atr75_m = project_to_minutes(minute_index, a75, tf)
    ev75_m = project_to_minutes(minute_index, ev75, tf)
    ev10_m = project_to_minutes(minute_index, ev10, tf)
    bandwp_m = project_to_minutes(minute_index, f_bandwp, tf)

    k_ = atr75_m.loc[is_start:is_end].mean() / max(ev75_m.loc[is_start:is_end].mean(), 1e-9)
    kv = k_ * (ev75_m.loc[is_start:is_end].mean() / max(ev10_m.loc[is_start:is_end].mean(), 1e-9))
    wmed = float(np.nanmedian(bandwp_m.loc[is_start:is_end].values))

    return dict(kv=float(kv), wmed=wmed, n_is=n_is,
                atr14_lo=a14_lo, atr14_hi=a14_hi,
                atr50_lo=a50_lo, atr50_hi=a50_hi,
                atr75_lo=a75_lo, atr75_hi=a75_hi)


# ---------------------------------------------------------------------------------
# EXPO chain (pairs_leadlag.build_ewma_norm + pairs_relvalue.build_expo, verbatim).
# ---------------------------------------------------------------------------------
def basket_state(daily_closes):
    """`en`: per-symbol sqrt(EWMA(span=10, adjust=False) of squared daily log-returns)
    -> basket mean -> / rolling-60d median -> shift(1). The row stamped on day D uses
    closes through D-1 (the causality lag). daily_closes: DataFrame, one column per
    STATE_SYMS symbol, daily-last closes."""
    ew = {}
    for s in daily_closes.columns:
        d = daily_closes[s]
        ew[s] = np.sqrt((np.log(d).diff() ** 2).ewm(span=EXPO_STATE_SPAN, adjust=False).mean())
    m = pd.DataFrame(ew).mean(axis=1)
    return (m / m.rolling(EXPO_MED_WIN).median()).shift(1)


def expo_signal(en, win=EXPO_WIN):
    """The 20-day CHANGE of the state, shifted ONE MORE day (build_expo:47): the value
    applied on day D uses closes through D-2. BOTH shifts are frozen."""
    return (en - en.shift(win)).shift(1)


def fit_expo_frozen(sig, calib_end=CALIBRATION_END):
    """The IS-frozen 101-point quantile grid + the IS-mean normalizer (build_expo:48-52).
    fillna(1.0) BEFORE the mean-normalize is load-bearing: early pre-grid days enter the
    IS mean as 1.0. Run ONCE (--rebuild); frozen thereafter."""
    ref = sig[sig.index < pd.Timestamp(calib_end)].dropna()
    grid = np.quantile(ref, np.linspace(0, 1, 101))
    pct = pd.Series(np.interp(sig.values, grid, np.linspace(0, 1, 101)), index=sig.index)
    e = (EXPO_FLOOR + EXPO_SLOPE * (1 - pct)).fillna(1.0)
    is_mean = float(e[e.index < pd.Timestamp(calib_end)].mean())
    return grid, is_mean


def expo_from_signal(sig, grid, is_mean):
    """signal -> expo series with the FROZEN grid/mean. NaN signal -> pct NaN ->
    e = 1.0 (fillna) -> 1.0/is_mean, exactly the reference's early-history value."""
    grid = np.asarray(grid, dtype='float64')
    pct = pd.Series(np.interp(sig.values, grid, np.linspace(0, 1, 101)), index=sig.index)
    e = (EXPO_FLOOR + EXPO_SLOPE * (1 - pct)).fillna(1.0)
    return e / is_mean
