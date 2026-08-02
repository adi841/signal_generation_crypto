"""Build the daily entry-sizing multiplier R for directional_momentum (DMP_v3_2).

R scales the allocation of NEW **SHORT** ENTRIES ONLY:
    LONG   tal = clip(VT / rv_alloc / annf, 0, 1)     * clip(K/z, 0, 1)
    SHORT  tal = clip(VT / rv_alloc / annf, 0, 1) * R * clip(K/z, 0, 1)
It is read once, at the entry bar, and held for the life of the trade. `allocation` appears
in no kernel CONDITION, so R changes position SIZE and never entry/exit timing.

WHY SHORTS-ONLY AND CAPPED (both are the same decision, writeup section 4): hot-state short
entries are late crash-chasing, so throttling them helps; up-sizing calm LONG entries
de-hedges the book — tested and rejected, it cost -0.24 OOS and deepened MaxDD by 1.8pp.
The cap is `np.minimum(R_CAP=1.0, ...)`, so R lives in [0.25, 1.00] and can only ever
throttle. (B1's R is the same family but uncapped, reaching 1.75.)

R is a CROSS-ASSET daily state -- a single-asset live process cannot derive it -- so it has
to arrive as a shared artifact, which is what this builds.

    state V   = daily mean, across the 11 traded assets, of each asset's clipped 15-minute
                rv. The mean is over that asset's 15-MINUTE CANDLES (96/day), NOT over a
                minute-broadcast series -- that is the sharpest difference from B1's R.
    ewma_norm = V / V.rolling(60).median()
    pct       = expanding strict-less percentile of ewma_norm, min 120 obs,
                CURRENT OBSERVATION EXCLUDED from its own comparison set
    R         = min(1.0, 0.25 + 1.5 * (1 - pct))   -> [0.25, 1.00],  stamped at d + 1 day

TWO MODES

  --rebuild   Seed the daily state from the spliced minute archive, reproducing
              PRODUCTION/engines/directional_momentum.py::build_r_state() over all of
              history. Run once. It must use the EXPANDING IS clip, not the frozen bounds:
              before 2021 the expanding bounds are NaN, so those days drop out of the daily
              mean entirely (`.dropna()`) and never enter the expanding percentile's
              comparison set. With frozen bounds they would be real numbers, would survive,
              and every subsequent percentile -- hence every R -- would differ.
              (Contrast B1, where a minute-level bfill FLOODS pre-history in rather than
              dropping it out. Opposite mechanics, same lesson: use the expanding clip.)

  (default)   Nightly. Read the cached daily state, compute only the new completed day(s)
              from tsdb, append, then recompute the rolling median / percentile / cap / lag
              over the (small) daily series and rewrite the artifact.

  Output: eod_scripts/directional_momentum/output/r_state_cache.parquet  (date, V, source)
          eod_scripts/directional_momentum/output/r_state_daily.parquet  (date, R)
          -- already capped and already stamped with the day it APPLIES TO

  Run:    python eod_scripts/directional_momentum/build_r_state.py --rebuild
          python eod_scripts/directional_momentum/build_r_state.py
          python eod_scripts/directional_momentum/build_r_state.py --date 2026-08-02

A day is only written if EVERY one of the 11 assets covers it end to end. Otherwise the day
is skipped with a critical log and live holds the previous R, rather than consuming an R
computed from a partial universe -- which, because R only throttles, would systematically
OVER-size short entries.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (DMP_ASSETS, ARTIFACT_DIR, DATA_PERP, DATA_COMBINED,
                     R_FLOOR, R_SLOPE, R_CAP, R_LAG_DAYS, R_MED_WIN, R_PCT_MINPERIODS,
                     R_STATE_TF, IS_START_DEFAULT, IS_END_DEFAULT, Q_LO_DEFAULT, Q_HI_DEFAULT,
                     spliced_ohlc, state_close_15m, rv_raw, clip_frozen, clip_expanding,
                     state_daily_from_assets, r_from_state, write_parquet_atomic)

CACHE_NAME = 'r_state_cache.parquet'
ARTIFACT_NAME = 'r_state_daily.parquet'
BOUNDS_NAME = 'rv_alloc_bounds.parquet'


# ---------------------------------------------------------------------------------
# tsdb
# ---------------------------------------------------------------------------------
def tsdb_symbol(sym):
    """'BTCUSDT' -> 'BTC-USDT.PERP'.

    ohlcv_data stores the INTERNAL name (hist_data.py writes `coin`, the value side of
    binance_config's symbol_mapping), not the exchange symbol. Derived rather than read from
    a checked-in config copy so this works on a fresh host; every symbol is then verified to
    exist during the completeness check, so a wrong guess fails loudly instead of silently
    producing an empty day.
    """
    if not sym.endswith('USDT'):
        raise ValueError(f'unexpected symbol form: {sym}')
    return f'{sym[:-4]}-USDT.PERP'


def connect_tsdb(host=None, port=None, user=None, database=None):
    """Direct postgres connection.

    Port 5432 on purpose, NOT pgbouncer's 6432: that runs pool_mode=transaction, which a
    long batch read should not sit on.

    config_utils runs check_sync_with_origin() -> `git fetch origin` at IMPORT time, which
    fails outside a properly-remoted checkout, so the import is guarded and we fall back to
    the standard PG* env vars.
    """
    import psycopg2
    pw = os.environ.get('POSTGRES_PASSWORD')
    try:
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
    print(f'  connecting {user}@{host}:{port}/{database}')
    return psycopg2.connect(user=user, password=pw, host=host, port=port, database=database)


def load_tsdb_minutes(cur, sym, start, end):
    """1-minute bars for [start, end], UTC. Shape copied from
    utils/base.py::load_intraday_data -- parameterised, ORDER BY start_time.

    Returns OHLC even though the R state only reads Close: the completeness check wants the
    same frame shape as the archive path, and the cost is one column.
    """
    cur.execute(
        'SELECT start_time, open, high, low, close FROM ohlcv_data '
        'WHERE symbol = %s AND start_time >= %s AND start_time <= %s ORDER BY start_time',
        (tsdb_symbol(sym), start, end))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'],
                            index=pd.DatetimeIndex([]))
    df = pd.DataFrame(rows, columns=['start_time', 'Open', 'High', 'Low', 'Close'])
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    df = df.set_index('start_time')
    # The archive path is tz-naive UTC; match it so both sources share one index dtype.
    df.index = df.index.tz_localize(None)
    return df.astype('float64')


def day_is_complete(px, day, assets):
    """Every asset must cover the target day end to end.

    Deliberately tested against the TARGET DAY, never against 'latest bar in the table' --
    live ingest can trail by a few minutes on the current incomplete minute, and that must
    not reject a complete previous day.
    """
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    problems = []
    for s in assets:
        d = px.get(s)
        if d is None or len(d) == 0:
            problems.append(f'{s}: no rows at all (tsdb symbol {tsdb_symbol(s)})')
            continue
        day_rows = d.loc[d0:d1]
        if len(day_rows) == 0:
            problems.append(f'{s}: no rows on {day}')
        elif day_rows.index[0] > d0 or day_rows.index[-1] < d1:
            problems.append(f'{s}: covers only {day_rows.index[0]} .. {day_rows.index[-1]}')
        elif len(day_rows) < 1440:
            # Gaps inside the day leave empty 15-minute buckets -> NaN rv for those buckets,
            # which the daily mean then skips (as PRODUCTION does). Report, do not reject.
            print(f'    [warn] {s}: {1440 - len(day_rows)} missing minute(s) on {day}')
    return problems


# ---------------------------------------------------------------------------------
# state construction
# ---------------------------------------------------------------------------------
def load_bounds(path, assets):
    """The R state's frozen 15-minute clip bounds.

    Read from the tf=15 rows of rv_alloc_bounds.parquet. That is legitimate ONLY because the
    state's 15-minute construction (Close only, no ffill/bfill) coincides bitwise with the
    TF=15 cell construction on the spliced archive -- build_rv_alloc_bounds.py asserts this
    every time it runs and refuses to write if it ever stops holding.
    """
    df = pd.read_parquet(path)
    df = df[df['tf'] == R_STATE_TF]
    out = {r.coin1: (float(r.q_lo), float(r.q_hi)) for r in df.itertuples()}
    missing = [s for s in assets if s not in out]
    if missing:
        raise KeyError(f'{path} has no tf={R_STATE_TF} row for: {missing}. '
                       f'Run build_rv_alloc_bounds.py first.')
    return out


def asset_state_rv(px, assets, bounds=None, is_window=None):
    """Per-asset CLIPPED 15-minute rv, for every DMP asset. build_r_state:34-41.

    Exactly one of `bounds` / `is_window` must be given:

      bounds     {asset: (q_lo, q_hi)} -- the FROZEN constants. Correct for any bar at or
                 after the IS end, i.e. the nightly path.
      is_window  (is_start, is_end, q_lo_q, q_hi_q) -- reproduce PRODUCTION's EXPANDING
                 clip, which is NaN before is_start and grows through the IS window. The
                 seed MUST use this: with frozen bounds, pre-2021 rv is a real number
                 instead of NaN, those days survive `.dropna()` and enter the expanding
                 percentile's permanent comparison set, and every R thereafter differs.
    """
    if (bounds is None) == (is_window is None):
        raise ValueError('pass exactly one of bounds= / is_window=')
    out = []
    for sym in assets:
        c = state_close_15m(px[sym]['Close'])
        rv = rv_raw(c)
        if bounds is not None:
            q_lo, q_hi = bounds[sym]
            out.append(clip_frozen(rv, q_lo, q_hi))
        else:
            s, e, ql, qh = is_window
            out.append(clip_expanding(rv, s, e, ql, qh))
    return out


def write_outputs(cache, args):
    """Recompute R over the whole daily state and write both files."""
    cache = cache.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    V = pd.Series(cache['V'].to_numpy(), index=pd.DatetimeIndex(cache['date']), name='V')
    R = r_from_state(V, args.r_floor, args.r_slope, args.r_cap,
                     args.med_win, args.pct_minp, args.lag_days)

    if args.compare_to_reference:
        _compare_reference(V, R, args)

    write_parquet_atomic(cache, os.path.join(args.out_dir, CACHE_NAME), index=False)
    art = pd.DataFrame({'date': R.index, 'R': R.to_numpy()}).dropna().reset_index(drop=True)
    write_parquet_atomic(art, os.path.join(args.out_dir, ARTIFACT_NAME), index=False)

    print(f'\ncache     {len(cache):,} days  {cache["date"].iloc[0].date()} .. '
          f'{cache["date"].iloc[-1].date()}  ({cache["source"].value_counts().to_dict()})')
    print(f'artifact  {len(art):,} days  {art["date"].iloc[0].date()} .. '
          f'{art["date"].iloc[-1].date()}')
    print(f'R         min={art["R"].min():.6f}  median={art["R"].median():.6f}  '
          f'max={art["R"].max():.6f}')
    if not art['R'].between(args.r_floor, args.r_cap).all():
        print(f'  *** R outside [{args.r_floor}, {args.r_cap}] -- investigate')
    print(f'wrote {os.path.join(args.out_dir, CACHE_NAME)}')
    print(f'wrote {os.path.join(args.out_dir, ARTIFACT_NAME)}')
    return art


def _compare_reference(V, R, args):
    """Check the index-shift against PRODUCTION's literal `.shift(lag_days)`.

    They agree exactly on a contiguous daily index; the index shift additionally emits the
    newest row (see _common.r_from_state). If V is NOT contiguous -- a skipped day -- they
    diverge, and that is precisely when we want to know.
    """
    ewma_norm = (V / V.rolling(args.med_win).median()).dropna()
    pct = ewma_norm.expanding(min_periods=args.pct_minp).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    ref = np.minimum(args.r_cap, args.r_floor + args.r_slope * (1 - pct)).shift(args.lag_days)
    ref = ref.dropna()
    j = pd.concat({'ours': R, 'ref': ref}, axis=1).dropna()
    d = (j['ours'] - j['ref']).abs()
    extra = R.index.difference(ref.index)
    gaps = int((V.index.to_series().diff().dt.days.dropna() != 1).sum())
    print(f'\n--compare-to-reference: {len(j):,} overlapping days, max|diff| = {d.max():.3e}, '
          f'{int((d > 0).sum())} differing')
    print(f'  V index gaps (non-consecutive days): {gaps}')
    print(f'  rows we emit that PRODUCTION\'s .shift() drops: {len(extra)} '
          f'{[str(x.date()) for x in extra]}')
    if d.max() > 0:
        print('  *** index-shift and .shift() DISAGREE -- V is not contiguous; investigate '
              'before trusting this artifact')


# ---------------------------------------------------------------------------------
def do_rebuild(args):
    t0 = time.time()
    print(f'loading {len(args.assets)} assets from the spliced archive ...')
    px = {s: spliced_ohlc(s, args.perp_dir, args.combined_dir) for s in args.assets}
    for s in args.assets:
        print(f'  {s:<10} {px[s].index[0]} .. {px[s].index[-1]}  n={len(px[s]):,}')

    is_window = (args.is_start, args.is_end, args.q_lo, args.q_hi)
    print(f'\nbuilding {len(args.assets)} asset rv series at tf={args.state_tf} '
          f'with the EXPANDING IS clip {is_window[0]!r} .. {is_window[1]!r} '
          f'[{args.q_lo}, {args.q_hi}] ...')
    V = state_daily_from_assets(asset_state_rv(px, args.assets, is_window=is_window))
    print(f'V: {len(V):,} days  {V.index[0].date()} .. {V.index[-1].date()}  '
          f'({time.time() - t0:.0f}s)')

    cache = pd.DataFrame({'date': V.index, 'V': V.to_numpy(), 'source': 'archive'})
    return write_outputs(cache, args)


def do_incremental(args):
    cache_path = os.path.join(args.out_dir, CACHE_NAME)
    if not os.path.exists(cache_path):
        raise SystemExit(f'{cache_path} not found -- run with --rebuild once to seed it.')
    cache = pd.read_parquet(cache_path)
    cache['date'] = pd.to_datetime(cache['date'])
    last = cache['date'].max()
    print(f'cache has {len(cache):,} days through {last.date()}')

    if args.date:
        targets = [pd.Timestamp(args.date).normalize()]
    else:
        # Only fully-completed UTC days.
        yesterday = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
        targets = list(pd.date_range(last + pd.Timedelta(days=1), yesterday, freq='D'))
    if not targets:
        print('nothing to do -- cache is already current')
        return None
    print(f'target day(s): {targets[0].date()} .. {targets[-1].date()}  ({len(targets)})')

    bounds = load_bounds(args.bounds, args.assets)
    conn = connect_tsdb(args.pg_host, args.pg_port, args.pg_user, args.pg_database)
    cur = conn.cursor()
    added, skipped = [], []
    try:
        for day in targets:
            lo = day - pd.Timedelta(days=args.warmup_days)
            hi = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            print(f'\n{day.date()}: pulling {args.warmup_days}d warm-up '
                  f'({lo.date()} .. {hi.date()}) for {len(args.assets)} assets')
            px = {s: load_tsdb_minutes(cur, s, lo, hi) for s in args.assets}

            problems = day_is_complete(px, day, args.assets)
            if problems:
                print(f'  [CRITICAL] incomplete cross-section on {day.date()} -- NOT written:')
                for p in problems:
                    print(f'      {p}')
                skipped.append(day)
                continue

            series = asset_state_rv(px, args.assets, bounds=bounds)
            d1 = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            per_asset = [float(s.loc[day:d1].mean()) for s in series]
            v = float(np.mean(per_asset))
            print(f'  V = {v:.9f}   (per-asset min {min(per_asset):.6f} max {max(per_asset):.6f})')
            added.append({'date': day, 'V': v, 'source': 'tsdb'})
    finally:
        cur.close(); conn.close()

    if skipped:
        print(f'\n[CRITICAL] {len(skipped)} day(s) skipped: '
              f'{", ".join(str(d.date()) for d in skipped)}')
    if not added:
        print('\nno new days written; existing artifact left untouched')
        return None

    cache = pd.concat([cache, pd.DataFrame(added)], ignore_index=True)
    return write_outputs(cache, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rebuild', action='store_true',
                    help='seed the full daily state from the spliced archive (run once)')
    ap.add_argument('--date', default=None,
                    help='compute this single UTC day instead of everything since the cache')
    ap.add_argument('--warmup-days', type=int, default=60,
                    help='history pulled per target day so the span-75 EWM is converged '
                         '(default 60 = 5760 candles at tf=15, 77x the span)')
    ap.add_argument('--state-tf', type=int, default=R_STATE_TF)
    ap.add_argument('--assets', nargs='+', default=DMP_ASSETS)
    ap.add_argument('--is-start', default=IS_START_DEFAULT,
                    help='--rebuild only: IS window start for the expanding clip (STRING label)')
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help='--rebuild only: IS window end, a bare date STRING -- the whole day '
                         'is INCLUDED. Not a Timestamp; see _common.py')
    ap.add_argument('--q-lo', type=float, default=Q_LO_DEFAULT)
    ap.add_argument('--q-hi', type=float, default=Q_HI_DEFAULT)
    ap.add_argument('--out-dir', default=ARTIFACT_DIR)
    ap.add_argument('--bounds', default=os.path.join(ARTIFACT_DIR, BOUNDS_NAME))
    ap.add_argument('--perp-dir', default=DATA_PERP)
    ap.add_argument('--combined-dir', default=DATA_COMBINED)
    ap.add_argument('--r-floor', type=float, default=R_FLOOR)
    ap.add_argument('--r-slope', type=float, default=R_SLOPE)
    ap.add_argument('--r-cap', type=float, default=R_CAP)
    ap.add_argument('--med-win', type=int, default=R_MED_WIN)
    ap.add_argument('--pct-minp', type=int, default=R_PCT_MINPERIODS)
    ap.add_argument('--lag-days', type=int, default=R_LAG_DAYS)
    ap.add_argument('--compare-to-reference', action='store_true',
                    help='also compute R with PRODUCTION\'s literal .shift(lag) and report '
                         'the diff plus the rows the shift drops')
    ap.add_argument('--pg-host', default=None)
    ap.add_argument('--pg-port', type=int, default=None)
    ap.add_argument('--pg-user', default=None)
    ap.add_argument('--pg-database', default=None)
    a = ap.parse_args()

    print(f'R = min({a.r_cap}, {a.r_floor} + {a.r_slope}*(1-pct))   med_win={a.med_win}  '
          f'pct_minp={a.pct_minp}  lag={a.lag_days}d  state_tf={a.state_tf}  '
          f'assets={len(a.assets)}   [SHORT ENTRIES ONLY]\n')
    do_rebuild(a) if a.rebuild else do_incremental(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
