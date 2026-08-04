"""Build the daily entry-sizing multiplier R for pair_momentum (B1_v8_v10).

R scales the allocation of NEW ENTRIES only:
    tal = clip(VT / rv_alloc / annf, 0, 1) * R * clip(K/z, 0, 1)
It is read once, at the entry bar, and held for the life of the trade. `alloc` appears in
no kernel CONDITION, so R changes position SIZE and never entry/exit timing.

R is a CROSS-PAIR daily state -- a single-pair live process cannot derive it -- so it has to
arrive as a shared artifact, which is what this builds.

    state V   = daily mean, across the 17 pairs, of the 15-minute rv_alloc
                (mean over MINUTES of the minute-broadcast series, not over candles)
    ewma_norm = V / V.rolling(60).median()
    pct       = expanding strict-less percentile of ewma_norm, min 120 obs,
                CURRENT OBSERVATION EXCLUDED from its own comparison set
    R         = (0.25 + 1.5 * (1 - pct)).shift(1)          -> [0.25, 1.75]

TWO MODES

  --rebuild   Seed the daily state from the spliced minute archive, reproducing
              PRODUCTION/engines/pairs_momentum.py::build_r_state() over all of history.
              Run once. This is what makes our R match the reference book: PRODUCTION's
              ratios start 2019-12-31 because the legs are outer-merged then bfilled, which
              floods ~367 pre-history days into V -- and those days sit in the expanding
              percentile's comparison set FOREVER. A tsdb-only rebuild cannot reproduce
              them (tsdb starts 2019-09 for BTC but 2020-07..09 for the alts), so R would
              differ from the book at every date.

  (default)   Nightly. Read the cached daily state, compute only the new completed day(s)
              from tsdb, append, then recompute the rolling median / percentile / lag over
              the (small) daily series and rewrite the artifact.

  Output: eod_scripts/pair_momentum/r_state_cache.parquet   (date, V, source)
          eod_scripts/pair_momentum/r_state_daily.parquet   (date, R)  -- already lagged

  Run:    python eod_scripts/pair_momentum/build_r_state.py --rebuild
          python eod_scripts/pair_momentum/build_r_state.py
          python eod_scripts/pair_momentum/build_r_state.py --date 2026-07-29

A day is only written if EVERY one of the 10 legs covers it end to end. Otherwise the day is
skipped with a critical log and live holds the previous R, rather than consuming an R
computed from a partial universe.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B1_PAIRS, B1_SYMBOLS, ARTIFACT_DIR, DATA_PERP, DATA_COMBINED,
                     R_FLOOR, R_SLOPE, R_LAG_DAYS, R_MED_WIN, R_PCT_MINPERIODS, R_STATE_TF,
                     IS_START_DEFAULT, IS_END_DEFAULT, Q_LO_DEFAULT, Q_HI_DEFAULT,
                     spliced_ohlc, ratio_minute_frame, rv_alloc_minutes,
                     clip_frozen, clip_expanding,
                     daily_state_from_pairs, r_from_state, write_parquet_atomic)

CACHE_NAME = 'r_state_cache.parquet'
ARTIFACT_NAME = 'r_state_daily.parquet'
BOUNDS_NAME = 'rv_alloc_bounds.parquet'


# ---------------------------------------------------------------------------------
# tsdb
# ---------------------------------------------------------------------------------
def tsdb_symbol(sym):
    """'BTCUSDT' -> 'BTC-USDT.PERP'.

    ohlcv_data stores the INTERNAL name (hist_data.py:239 writes `coin`, the value side of
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
    long batch read should not sit on (parquet_loader.py:17-23).

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
    utils/base.py:119-138::load_intraday_data -- parameterised, ORDER BY start_time."""
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


def day_is_complete(px, day):
    """Every leg must cover the target day end to end.

    Deliberately tested against the TARGET DAY, never against 'latest bar in the table' --
    live ingest can trail by a few minutes on the current incomplete minute, and that must
    not reject a complete previous day.
    """
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    problems = []
    for s in B1_SYMBOLS:
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
            # Gaps inside the day are ffilled (as PRODUCTION does); report, do not reject.
            print(f'    [warn] {s}: {1440 - len(day_rows)} missing minute(s) on {day}, ffilled')
    return problems


# ---------------------------------------------------------------------------------
# state construction
# ---------------------------------------------------------------------------------
def load_bounds(path, tf):
    df = pd.read_parquet(path)
    df = df[df['tf'] == int(tf)]
    out = {(r.coin1, r.coin2): (float(r.q_lo), float(r.q_hi)) for r in df.itertuples()}
    missing = [p for p in B1_PAIRS if p not in out]
    if missing:
        raise KeyError(f'{path} has no tf={tf} row for: {missing}. '
                       f'Run build_rv_alloc_bounds.py first.')
    return out


def pair_minute_rv(px, tf, bounds=None, is_window=None):
    """Per-pair clipped rv_alloc on the minute timeline, for every B1 pair.

    Exactly one of `bounds` / `is_window` must be given:

      bounds     {(c1,c2): (q_lo, q_hi)} -- the FROZEN constants. Correct for any bar at or
                 after the IS end, i.e. the nightly path.
      is_window  (is_start, is_end, q_lo_q, q_hi_q) -- reproduce PRODUCTION's EXPANDING
                 clip, which is NaN before is_start and grows through the IS window. The
                 seed must use this: with frozen bounds, pre-2021 rv_alloc is a real number
                 instead of NaN, the minute-level bfill never fires, and the ~367 flat
                 pre-history days that live in R's permanent comparison set are missing.
    """
    if (bounds is None) == (is_window is None):
        raise ValueError('pass exactly one of bounds= / is_window=')
    out = []
    for c1, c2 in B1_PAIRS:
        raw = ratio_minute_frame(px[c1], px[c2], c1, c2)
        if bounds is not None:
            q_lo, q_hi = bounds[(c1, c2)]
            clip_fn = lambda rv, lo=q_lo, hi=q_hi: clip_frozen(rv, lo, hi)
        else:
            s, e, ql, qh = is_window
            clip_fn = lambda rv: clip_expanding(rv, s, e, ql, qh)
        out.append(rv_alloc_minutes(raw, tf, clip_fn))
    return out


def write_outputs(cache, args):
    """Recompute R over the whole daily state and write both files."""
    cache = cache.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    V = pd.Series(cache['V'].to_numpy(), index=pd.DatetimeIndex(cache['date']), name='V')
    R = r_from_state(V, args.r_floor, args.r_slope, args.med_win, args.pct_minp, args.lag_days)

    write_parquet_atomic(cache, os.path.join(args.out_dir, CACHE_NAME), index=False)
    art = pd.DataFrame({'date': R.index, 'R': R.to_numpy()}).dropna().reset_index(drop=True)
    write_parquet_atomic(art, os.path.join(args.out_dir, ARTIFACT_NAME), index=False)

    print(f'\ncache     {len(cache):,} days  {cache["date"].iloc[0].date()} .. '
          f'{cache["date"].iloc[-1].date()}  ({cache["source"].value_counts().to_dict()})')
    print(f'artifact  {len(art):,} days  {art["date"].iloc[0].date()} .. '
          f'{art["date"].iloc[-1].date()}')
    print(f'R         min={art["R"].min():.6f}  median={art["R"].median():.6f}  '
          f'max={art["R"].max():.6f}')
    if not art['R'].between(args.r_floor, args.r_floor + args.r_slope).all():
        print(f'  *** R outside [{args.r_floor}, {args.r_floor + args.r_slope}] -- investigate')
    print(f'wrote {os.path.join(args.out_dir, CACHE_NAME)}')
    print(f'wrote {os.path.join(args.out_dir, ARTIFACT_NAME)}')
    return art


# ---------------------------------------------------------------------------------
def do_rebuild(args):
    t0 = time.time()
    print(f'loading {len(B1_SYMBOLS)} symbols from the spliced archive ...')
    px = {s: spliced_ohlc(s, args.perp_dir, args.combined_dir) for s in B1_SYMBOLS}
    for s in B1_SYMBOLS:
        print(f'  {s:<10} {px[s].index[0]} .. {px[s].index[-1]}  n={len(px[s]):,}')

    is_window = (args.is_start, pd.Timestamp(args.is_end), args.q_lo, args.q_hi)
    print(f'\nbuilding {len(B1_PAIRS)} pair rv_alloc series at tf={args.state_tf} '
          f'with the EXPANDING IS clip {is_window[0]} .. {is_window[1]} '
          f'[{args.q_lo}, {args.q_hi}] ...')
    V = daily_state_from_pairs(pair_minute_rv(px, args.state_tf, is_window=is_window))
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

    bounds = load_bounds(args.bounds, args.state_tf)
    conn = connect_tsdb(args.pg_host, args.pg_port, args.pg_user, args.pg_database)
    cur = conn.cursor()
    added, skipped = [], []
    try:
        for day in targets:
            lo = day - pd.Timedelta(days=args.warmup_days)
            hi = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            print(f'\n{day.date()}: pulling {args.warmup_days}d warm-up '
                  f'({lo.date()} .. {hi.date()}) for {len(B1_SYMBOLS)} symbols')
            px = {s: load_tsdb_minutes(cur, s, lo, hi) for s in B1_SYMBOLS}

            problems = day_is_complete(px, day)
            if problems:
                print(f'  [CRITICAL] incomplete cross-section on {day.date()} -- NOT written:')
                for p in problems:
                    print(f'      {p}')
                skipped.append(day)
                continue

            series = pair_minute_rv(px, args.state_tf, bounds=bounds)
            d1 = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            per_pair = [float(s.loc[day:d1].mean()) for s in series]
            v = float(np.mean(per_pair))
            print(f'  V = {v:.9f}   (per-pair min {min(per_pair):.6f} max {max(per_pair):.6f})')
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
                         '(default 60 = ~5760 candles at tf=15, 77x the span)')
    ap.add_argument('--state-tf', type=int, default=R_STATE_TF)
    ap.add_argument('--is-start', default=IS_START_DEFAULT,
                    help='--rebuild only: IS window start for the expanding clip')
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help='--rebuild only: IS window end, parsed as a Timestamp')
    ap.add_argument('--q-lo', type=float, default=Q_LO_DEFAULT)
    ap.add_argument('--q-hi', type=float, default=Q_HI_DEFAULT)
    ap.add_argument('--out-dir', default=ARTIFACT_DIR)
    ap.add_argument('--bounds', default=os.path.join(ARTIFACT_DIR, BOUNDS_NAME))
    ap.add_argument('--perp-dir', default=DATA_PERP)
    ap.add_argument('--combined-dir', default=DATA_COMBINED)
    ap.add_argument('--r-floor', type=float, default=R_FLOOR)
    ap.add_argument('--r-slope', type=float, default=R_SLOPE)
    ap.add_argument('--med-win', type=int, default=R_MED_WIN)
    ap.add_argument('--pct-minp', type=int, default=R_PCT_MINPERIODS)
    ap.add_argument('--lag-days', type=int, default=R_LAG_DAYS)
    ap.add_argument('--pg-host', default=None)
    ap.add_argument('--pg-port', type=int, default=None)
    ap.add_argument('--pg-user', default=None)
    ap.add_argument('--pg-database', default=None)
    a = ap.parse_args()

    print(f'R = {a.r_floor} + {a.r_slope}*(1-pct)   med_win={a.med_win}  '
          f'pct_minp={a.pct_minp}  lag={a.lag_days}d  state_tf={a.state_tf}\n')
    do_rebuild(a) if a.rebuild else do_incremental(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
