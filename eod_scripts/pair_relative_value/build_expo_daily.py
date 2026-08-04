"""Build the daily EXPO overlay for pair_relative_value (QR1_v4).

EXPO scales the allocation of NEW ENTRIES only:
    alloc = clip(VT / (EV10*kv) / annf, 0, 1) * clip(score/med, 0.25, 2.0) * EXPO
It appears in no kernel CONDITION, so EXPO changes position SIZE and never entry/exit
timing. It is a CROSS-MARKET daily state (10-symbol basket — DOT in, UNI out, NOT the
traded universe) that a single-pair live process cannot derive, so it arrives as a
shared artifact, which is what this builds.

    en   = per-sym sqrt(EWMA(span=10, adjust=False) of squared DAILY log-returns)
           -> basket mean -> / rolling-60d median -> shift(1 day)
    sig  = (en - en.shift(20)).shift(1)     <- day D uses closes through D-2 (BOTH
                                               shifts are frozen; do not collapse them)
    pct  = np.interp(sig, FROZEN 101-pt IS quantile grid, linspace(0,1,101))
    EXPO = (0.5 + 1.0*(1 - pct)).fillna(1.0) / FROZEN IS-mean

TWO MODES

  --rebuild   Seed the per-symbol daily closes from tsdb ohlcv_data (QR1's frozen book
              reads the raw PERP series — no splicing), fit + WRITE the frozen grid
              and IS-mean (expo_frozen.json — written ONLY here), and emit the full
              artifact. Run once.

  (default)   Nightly. Read the cached daily closes, pull only the new completed day(s)
              from tsdb, append, recompute the (small, daily) chain with the FROZEN
              grid/mean and rewrite the artifact + the bundle copy.

  Output: eod_scripts/pair_relative_value/expo_cache.parquet   (date, 10 close cols, source)
          eod_scripts/pair_relative_value/expo_frozen.json     (grid, is_mean, basket, consts)
          eod_scripts/pair_relative_value/expo_daily.parquet   (date, expo) -- already lagged
          + expo_daily.parquet copied to the production bundle data/ dir

  Run:    python eod_scripts/pair_relative_value/build_expo_daily.py --rebuild
          python eod_scripts/pair_relative_value/build_expo_daily.py
          python eod_scripts/pair_relative_value/build_expo_daily.py --date 2026-07-29

A day is only written if EVERY one of the 10 basket legs covers it end to end; otherwise
it is skipped with a critical log and live keeps serving the previous artifact (a missing
date resolves to the reference's fillna(1.0) at the consumer, with a CRITICAL log there).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (STATE_SYMS, ARTIFACT_DIR,
                     EXPO_FLOOR, EXPO_SLOPE, EXPO_WIN, EXPO_STATE_SPAN, EXPO_MED_WIN,
                     CALIBRATION_END,
                     basket_state, expo_signal, fit_expo_frozen, expo_from_signal,
                     tsdb_symbol, connect_tsdb, load_minute_ohlc, resolve_bundle_dir)

CACHE_NAME = 'expo_cache.parquet'
FROZEN_NAME = 'expo_frozen.json'
ARTIFACT_NAME = 'expo_daily.parquet'


# ---------------------------------------------------------------------------------
# tsdb. tsdb_symbol/connect_tsdb now live in _common.py — the whole sleeve reads from
# the database, not just this incremental path, so they are shared rather than local.
# ---------------------------------------------------------------------------------
def load_tsdb_day_closes(cur, sym, day):
    """The day's minute closes for one symbol (UTC, tz-naive to match the archive)."""
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    cur.execute(
        'SELECT start_time, close FROM ohlcv_data '
        'WHERE symbol = %s AND start_time >= %s AND start_time <= %s ORDER BY start_time',
        (tsdb_symbol(sym), d0, d1))
    rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype='float64')
    s = pd.Series([float(r[1]) for r in rows],
                  index=pd.to_datetime([r[0] for r in rows], utc=True))
    s.index = s.index.tz_localize(None)
    return s


def day_is_complete(closes_by_sym, day):
    """Every basket leg must cover the target day end to end (the daily close is the
    day's LAST minute — an early-truncated day would freeze the wrong close)."""
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    problems = []
    for s in STATE_SYMS:
        ser = closes_by_sym.get(s)
        if ser is None or len(ser) == 0:
            problems.append(f'{s}: no rows at all (tsdb symbol {tsdb_symbol(s)})')
        elif ser.index[0] > d0 or ser.index[-1] < d1:
            problems.append(f'{s}: covers only {ser.index[0]} .. {ser.index[-1]}')
    return problems


# ---------------------------------------------------------------------------------
def chain_and_write(cache, frozen, args):
    """closes cache -> en -> signal -> expo (frozen grid/mean) -> artifacts."""
    cache = cache.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    closes = cache.set_index('date')[STATE_SYMS]

    en = basket_state(closes)
    sig = expo_signal(en, args.expo_win)
    expo = expo_from_signal(sig, frozen['grid'], frozen['is_mean'])

    cache.to_parquet(os.path.join(args.out_dir, CACHE_NAME), index=False)
    art = pd.DataFrame({'date': expo.index, 'expo': expo.to_numpy()}).reset_index(drop=True)
    art_path = os.path.join(args.out_dir, ARTIFACT_NAME)
    art.to_parquet(art_path, index=False)

    print(f'\ncache     {len(cache):,} days  {cache["date"].iloc[0].date()} .. '
          f'{cache["date"].iloc[-1].date()}  ({cache["source"].value_counts().to_dict()})')
    print(f'artifact  {len(art):,} days  {art["date"].iloc[0].date()} .. '
          f'{art["date"].iloc[-1].date()}')
    lo_th = EXPO_FLOOR / frozen['is_mean']
    hi_th = (EXPO_FLOOR + EXPO_SLOPE) / frozen['is_mean']
    print(f'expo      min={art["expo"].min():.6f}  median={art["expo"].median():.6f}  '
          f'max={art["expo"].max():.6f}   (theoretical range [{lo_th:.6f}, {hi_th:.6f}])')
    if not art['expo'].between(lo_th - 1e-12, hi_th + 1e-12).all():
        print('  *** expo outside the theoretical range -- investigate')
    print(f'wrote {os.path.join(args.out_dir, CACHE_NAME)}')
    print(f'wrote {art_path}')

    ## Publishing is the LAST step and must not be able to fail a build whose real
    ## output is already on disk. The old hardcoded bundle path vanished with the
    ## crypto_sims tree and took the whole run down with it (FileNotFoundError) AFTER
    ## both artifacts had been written correctly. Warn loudly and carry on instead: a
    ## stale published copy is a visible problem, a build that reports failure while
    ## having actually succeeded is a confusing one.
    if not args.bundle_dir:
        print('\n*** NOT PUBLISHED: no bundle dir resolved from sleeve_config. '
              'Live still reads the OLD artifact.')
    elif not os.path.isdir(args.bundle_dir):
        print(f'\n*** NOT PUBLISHED: bundle dir does not exist: {args.bundle_dir}\n'
              f'    Live still reads the OLD artifact. Point sleeve_config at a real '
              f'directory, or pass --bundle-dir.')
    else:
        dst = os.path.join(args.bundle_dir, ARTIFACT_NAME)
        shutil.copyfile(art_path, dst)
        print(f'copied to {dst}')
    return art


def do_rebuild(args):
    t0 = time.time()
    print(f'loading {len(STATE_SYMS)} basket symbols from ohlcv_data ...')
    conn = connect_tsdb(args.pg_host, args.pg_port, args.pg_user, args.pg_database)
    closes, last_min = {}, None
    try:
        for s in STATE_SYMS:
            px = load_minute_ohlc(s, conn=conn)['Close']
            closes[s] = px.resample('D').last()
            if s == STATE_SYMS[0]:
                last_min = px.index[-1]
            print(f'  {s:<10} {closes[s].index[0].date()} .. {closes[s].index[-1].date()}  '
                  f'n={len(closes[s]):,}')
    finally:
        conn.close()
    daily = pd.DataFrame(closes)
    # Drop the trailing PARTIAL day: its "daily close" is whatever minute the feed stops
    # at, not a real 23:59 close. The last day is complete only if the last minute is
    # 23:59 — which, reading live from the database, is almost never true for today.
    if not (last_min.hour == 23 and last_min.minute == 59):
        print(f'  dropping trailing partial day {daily.index[-1].date()} '
              f'(feed ends {last_min})')
        daily = daily.iloc[:-1]

    en = basket_state(daily)
    sig = expo_signal(en, args.expo_win)
    grid, is_mean = fit_expo_frozen(sig, args.calib_end)
    frozen = dict(grid=[float(x) for x in grid], is_mean=is_mean,
                  basket=STATE_SYMS, expo_floor=EXPO_FLOOR, expo_slope=EXPO_SLOPE,
                  expo_win=args.expo_win, state_span=EXPO_STATE_SPAN,
                  med_win=EXPO_MED_WIN, calib_end=str(args.calib_end))
    with open(os.path.join(args.out_dir, FROZEN_NAME), 'w') as f:
        json.dump(frozen, f, indent=1)
    print(f'\nfroze grid (101 pts, [{grid[0]:.6f} .. {grid[-1]:.6f}]) + '
          f'is_mean={is_mean:.9f}  -> {FROZEN_NAME}   ({time.time() - t0:.0f}s)')

    cache = daily.reset_index().rename(columns={daily.index.name or 'index': 'date'})
    cache['source'] = 'archive'
    return chain_and_write(cache, frozen, args)


def do_incremental(args):
    cache_path = os.path.join(args.out_dir, CACHE_NAME)
    frozen_path = os.path.join(args.out_dir, FROZEN_NAME)
    for p in (cache_path, frozen_path):
        if not os.path.exists(p):
            raise SystemExit(f'{p} not found -- run with --rebuild once to seed it.')
    with open(frozen_path) as f:
        frozen = json.load(f)
    cache = pd.read_parquet(cache_path)
    cache['date'] = pd.to_datetime(cache['date'])
    last = cache['date'].max()
    print(f'cache has {len(cache):,} days through {last.date()}')

    if args.date:
        targets = [pd.Timestamp(args.date).normalize()]
    else:
        yesterday = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
        targets = list(pd.date_range(last + pd.Timedelta(days=1), yesterday, freq='D'))
    if not targets:
        print('nothing to do -- cache is already current')
        return None
    print(f'target day(s): {targets[0].date()} .. {targets[-1].date()}  ({len(targets)})')

    conn = connect_tsdb(args.pg_host, args.pg_port, args.pg_user, args.pg_database)
    cur = conn.cursor()
    added, skipped = [], []
    try:
        for day in targets:
            print(f'\n{day.date()}: pulling minute closes for {len(STATE_SYMS)} symbols')
            closes_by_sym = {s: load_tsdb_day_closes(cur, s, day) for s in STATE_SYMS}
            problems = day_is_complete(closes_by_sym, day)
            if problems:
                print(f'  [CRITICAL] incomplete basket on {day.date()} -- NOT written:')
                for p in problems:
                    print(f'      {p}')
                skipped.append(day)
                continue
            row = {'date': day, 'source': 'tsdb'}
            for s in STATE_SYMS:
                row[s] = float(closes_by_sym[s].iloc[-1])       # the day's LAST close
            added.append(row)
            print('  ' + '  '.join(f'{s}={row[s]:.6g}' for s in STATE_SYMS[:4]) + ' ...')
    finally:
        cur.close(); conn.close()

    if skipped:
        print(f'\n[CRITICAL] {len(skipped)} day(s) skipped: '
              f'{", ".join(str(d.date()) for d in skipped)}')
    if not added:
        print('\nno new days written; existing artifact left untouched')
        return None

    cache = pd.concat([cache, pd.DataFrame(added)], ignore_index=True)
    return chain_and_write(cache, frozen, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rebuild', action='store_true',
                    help='seed the full daily chain from ohlcv_data + freeze the '
                         'grid/mean (run once)')
    ap.add_argument('--date', default=None,
                    help='compute this single UTC day instead of everything since the cache')
    ap.add_argument('--expo-win', type=int, default=EXPO_WIN)
    ap.add_argument('--calib-end', default=CALIBRATION_END,
                    help='--rebuild only: the frozen-grid fit boundary')
    ap.add_argument('--out-dir', default=ARTIFACT_DIR)
    ap.add_argument('--bundle-dir', default=None,
                    help='dir the artifact is published to. Default: resolved from the '
                         "S3 client config's sleeve_config (the same place LIVE reads "
                         "it from). Pass '' to skip publishing.")
    ap.add_argument('--pg-host', default=None)
    ap.add_argument('--pg-port', type=int, default=None)
    ap.add_argument('--pg-user', default=None)
    ap.add_argument('--pg-database', default=None)
    a = ap.parse_args()

    ## None = "not specified, go ask the config"; '' = "explicitly skip publishing".
    if a.bundle_dir is None:
        a.bundle_dir = resolve_bundle_dir()

    print(f'EXPO = ({EXPO_FLOOR} + {EXPO_SLOPE}*(1-pct)) / is_mean   '
          f'win={a.expo_win}d  state_span={EXPO_STATE_SPAN}  med_win={EXPO_MED_WIN}  '
          f'calib_end={a.calib_end}\n')
    do_rebuild(a) if a.rebuild else do_incremental(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
