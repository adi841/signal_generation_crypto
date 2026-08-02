"""Shared pieces for the pair_lead_lag (QR31_v2) EOD builders.

Everything here is a faithful transcription of
`/home/rocky/crypto_sims/PRODUCTION/engines/core/vendored_pairs.py` (load_raw,
resample_pairs_data_closebased_perc, ATR/wwma_old, vol_pct_series, prep_cell_c's
role_vol) — PRODUCTION is the source of truth; where a line looks odd it is because
the reference is odd, and the comment says so.

We deliberately do NOT import PRODUCTION directly: it does `import config as cfg`
in the engine layer, and `signal_generation_crypto` has its own `config` PACKAGE
that shadows PRODUCTION's `config.py` on sys.path (the collision that bit
pair_momentum's test-1 harness). vendored_pairs itself is config-free, so the
VERIFICATION script imports it for cross-checks — but the builder stands alone.

DATA SOURCE: the PERP files directly (`{SYM}PERP-1m-data.parquet`) — QR31's frozen
books were built on those, NOT on the spliced archive (data_io.py docstring: "The
pairs strategies (QR1, QR3.1) load through their own vendored load_raw functions,
which read the PERP files directly"). No splicing here, unlike pair_momentum.

IS-WINDOW SUBTLETY (string vs Timestamp end): vendored_pairs has TWO
identically-quantiled clip laws whose windows differ at the margin —
  * clip_atr_pct (prep_cell_c: kernel atr/atr2, sizing vol): `.loc['2021':'2025-03-31']`
    with STRING labels -> pandas partial-string slicing INCLUDES the whole 2025-03-31 day.
  * clip_is (qr31_cert role_vol: the BAND ATRs): `.loc['2021':pd.Timestamp('2025-03-31')]`
    -> cuts at 2025-03-31 00:00:00, excluding that day's candles.
This module freezes the clip_atr_pct (string-end) values; the builder VERIFIES both
laws agree per cell (a 4+-year min/max almost never moves on the final day) and
refuses to write silently if they don't.
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# Universe. PRODUCTION/config.py:87,91 —
#   _A9 = _A7 + ['DOTUSDT', 'UNIUSDT']   (_A7 = SOL,XRP,DOGE,ADA,AVAX,LINK,BNB)
#   QR31_PAIRS = [('BTCUSDT', a) for a in _A9] + [('ETHUSDT', a) for a in _A9]
#                + [('BTCUSDT', 'ETHUSDT')]
# The WIDEST pairs universe: BNB and DOT/UNI all in (not B1's, not QR1's).
# ---------------------------------------------------------------------------------
QR31_ALTS = ['SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT',
             'BNBUSDT', 'DOTUSDT', 'UNIUSDT']
QR31_PAIRS = ([('BTCUSDT', a) for a in QR31_ALTS] +
              [('ETHUSDT', a) for a in QR31_ALTS] +
              [('BTCUSDT', 'ETHUSDT')])
QR31_SYMBOLS = sorted({s for pair in QR31_PAIRS for s in pair})    # 11 distinct legs

# vendored_pairs.py:15 — the frozen IS window, STRING labels (see module docstring).
IS_START_DEFAULT = '2021'
IS_END_DEFAULT = '2025-03-31'

# prep_cell_c: atr=ATR14, atr2=ATR50, atr3=period-75 sizing chain; band spans idem.
ATR_FAST = 14
ATR_SLOW = 50
SIZING_PERIOD = 75

## The retired parquet archive location (crypto_sims is gone; the surviving copy of
## the {SYM}PERP-1m-data.parquet files moved to /home/rocky/crypto_data, but its
## refresher went with crypto_sims, so it only goes STALE from 2026-08-05 19:06 on).
## Kept solely for the frozen one-shot build_qr31_cell_constants.py (--perp-dir);
## the recurring builders read tsdb `ohlcv_data` instead — see load_minute_ohlc.
DATA_PERP = '/home/rocky/crypto_data/'

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# ---------------------------------------------------------------------------------
# DATA SOURCE: tsdb `ohlcv_data`, NOT the PERP parquet archive (mirrors the
# pair_relative_value builders, where the substitution was verified BITWISE on every
# shared OHLC cell over multi-year sample windows — see that _common.py's note).
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
        ## repo root, so `shared_codes` was never importable without help. Put the
        ## repo root on the path so get_db_params gets a chance to answer.
        import sys as _sys
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    load_perp. Returns EXACTLY what the parquet read returned, so nothing downstream
    changes: columns ['Open','High','Low','Close'] (capitalised), float64, tz-NAIVE
    UTC index, sorted and de-duplicated. Naive on purpose — consumers slice with
    plain date strings, which a tz-aware index would accept with different boundary
    semantics.

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


def write_parquet_atomic(df, path, **kwargs):
    """Write a parquet the live path can poll safely (temp name + os.replace so a
    reader sees either the old file or the new one, never a partial one)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)


def load_perp(sym, perp_dir=DATA_PERP):
    """Raw PERP minute frame for one leg (Open/Close are all QR31 consumes)."""
    return pd.read_parquet(f'{perp_dir}{sym}PERP-1m-data.parquet',
                           columns=['Open', 'High', 'Low', 'Close'])


def ratio_minute_frame(d1, d2):
    """The minute-level ratio frame — vendored_pairs.load_raw(fix=True) verbatim:
    outer merge of the two legs, trim to the LATER leg's first valid close (the
    fix-start), ratio Open/Close, minute high/low = max/min(Open, Close)."""
    r1 = d1[['Open', 'Close']].rename(columns={'Open': 'o1', 'Close': 'c1'})
    r2 = d2[['Open', 'Close']].rename(columns={'Open': 'o2', 'Close': 'c2'})
    raw = r1.merge(r2, left_index=True, right_index=True, how='outer')
    start = max(raw['c1'].first_valid_index(), raw['c2'].first_valid_index())
    raw = raw.loc[start:]
    out = pd.DataFrame(index=raw.index)
    out['Open'] = raw['o1'] / raw['o2']
    out['Close'] = raw['c1'] / raw['c2']
    out['High'] = np.max(out[['Open', 'Close']].values, axis=1)
    out['Low'] = np.min(out[['Open', 'Close']].values, axis=1)
    return out[['Open', 'High', 'Low', 'Close']]


def ratio_candles(raw, tf):
    """prep_cell_c's candle frame: `.ffill().bfill()` on the MINUTE frame, then a
    plain close-based resample (first/max/min/last), dedup. NaN bins are KEPT."""
    df = raw.ffill().bfill().resample(f'{tf}min').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df = df.loc[~df.index.duplicated(), :]
    return df.rename(columns={'Open': 'open', 'High': 'high',
                              'Low': 'low', 'Close': 'close'})


def wilder_atr_pct(candles, n):
    """RAW Wilder ATR as % of close — vendored ATR (tr0/tr1/tr2 max, skipna; first
    row = high-low) + wwma_old (ewm alpha=1/n, adjust=False), then /close*100."""
    tr = pd.concat([(candles['high'] - candles['low']).abs(),
                    (candles['high'] - candles['close'].shift()).abs(),
                    (candles['low'] - candles['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr / candles['close'] * 100.0


def ewma_vol_pct(candles, span):
    """RAW EWMA sizing vol % — vol_pct_series(df, n, 'ewma'):
    sqrt((log-return²).ewm(span=n, adjust=False).mean()) * 100."""
    r = np.log(candles['close']).diff()
    return np.sqrt((r ** 2).ewm(span=span, adjust=False).mean()) * 100.0


def is_final_bounds(series_pct, is_start, is_end):
    """The frozen clip bounds = the FINAL values of clip_atr_pct's expanding
    quantiles 0.0 / 1.0 (== running min / max) over the IS slice. STRING labels:
    a bare '2025-03-31' end INCLUDES the whole day (see module docstring).
    Returns (lo, hi, n_is)."""
    seg = series_pct.loc[is_start:is_end]
    n_is = int(len(seg))
    if n_is == 0:
        raise ValueError(f'empty IS window [{is_start}, {is_end}]')
    lo = float(seg.expanding().quantile(0.0).iloc[-1])
    hi = float(seg.expanding().quantile(1.0).iloc[-1])
    return lo, hi, n_is


def sizing_scale_and_bounds(candles, is_start, is_end,
                            period=SIZING_PERIOD):
    """prep_cell_c role_vol(75, 'ewma') frozen constants:
      scale      = IS-mean(RAW ATR75 %) / max(IS-mean(raw EWMA75 %), 1e-12)
                   (RAW ATR75 — the reference scales BEFORE any clipping)
      lo/hi      = clip_atr_pct bounds of the SCALED series over IS.
    Returns (scale, lo, hi, n_is)."""
    a75 = wilder_atr_pct(candles, period)
    v75 = ewma_vol_pct(candles, period)
    scale = float(a75.loc[is_start:is_end].mean()) / max(float(v75.loc[is_start:is_end].mean()), 1e-12)
    scaled = v75 * scale
    lo, hi, n_is = is_final_bounds(scaled, is_start, is_end)
    return scale, lo, hi, n_is
