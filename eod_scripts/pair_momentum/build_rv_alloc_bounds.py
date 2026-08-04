"""Build the frozen per-(pair, TF) rv_alloc clip bounds for pair_momentum (B1_v8_v10).

`rv_alloc = clip(ewm(span=75).std(log-returns) * 100, q_lo, q_hi)`, where q_lo/q_hi are the
FINAL values of an expanding quantile fit over the in-sample window. PRODUCTION recomputes
that expanding quantile from 2021 on every run; live init data will never reach back that
far, so the live path reads the frozen constants from this artifact instead
(`generate_signal_pair_momentum.py::load_rv_alloc_bounds`).

Source is the spliced minute archive, not tsdb: these are frozen IS constants derived from
the definitional data, not live data.

  Output: eod_scripts/pair_momentum/rv_alloc_bounds.parquet
          (long: coin1, coin2, tf, q_lo, q_hi, n_is -- 85 rows)

  Run:    python eod_scripts/pair_momentum/build_rv_alloc_bounds.py
          python eod_scripts/pair_momentum/build_rv_alloc_bounds.py \
                 --is-start 2021 --is-end 2025-03-31 --quantiles 0.0 0.997

NOTE ON --is-end: it is parsed as a TIMESTAMP, matching PRODUCTION's
`rv.loc[ST:pd.Timestamp('2025-03-31')]`. A bare date therefore means 00:00:00 THAT DAY, and
the remainder of the day is excluded from the fit (95 candles at TF15). Pass an explicit
'2025-03-31 23:59' to include the whole day. The resolved window and the per-cell candle
count are logged so the fit is auditable rather than implicit.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B1_PAIRS, B1_SYMBOLS, ARTIFACT_DIR, IS_START_DEFAULT, IS_END_DEFAULT,
                     Q_LO_DEFAULT, Q_HI_DEFAULT, DATA_PERP, DATA_COMBINED,
                     spliced_ohlc, ratio_minute_frame, ratio_candles, rv_raw, is_bounds,
                     write_parquet_atomic)

TFS = [5, 15, 30, 60, 120]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--is-start', default=IS_START_DEFAULT,
                    help=f'IS window start, pandas .loc label (default {IS_START_DEFAULT!r})')
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help=f'IS window end, parsed as a Timestamp -- a bare date means '
                         f'00:00:00 that day (default {IS_END_DEFAULT!r})')
    ap.add_argument('--quantiles', nargs=2, type=float, metavar=('Q_LO', 'Q_HI'),
                    default=[Q_LO_DEFAULT, Q_HI_DEFAULT],
                    help=f'clip quantiles (default {Q_LO_DEFAULT} {Q_HI_DEFAULT})')
    ap.add_argument('--tfs', nargs='+', type=int, default=TFS)
    ap.add_argument('--perp-dir', default=DATA_PERP,
                    help='directory holding {SYM}PERP-1m-data.parquet')
    ap.add_argument('--combined-dir', default=DATA_COMBINED,
                    help='directory holding {SYM}-1m-combined.parquet')
    ap.add_argument('--out', default=os.path.join(ARTIFACT_DIR, 'rv_alloc_bounds.parquet'))
    ap.add_argument('--compare-to', default=None,
                    help='optional existing bounds parquet to diff against before writing; '
                         'reports per-row exact/differs and does not block the write')
    ap.add_argument('--dry-run', action='store_true', help='compute and report, write nothing')
    a = ap.parse_args()

    q_lo_q, q_hi_q = a.quantiles
    is_end = pd.Timestamp(a.is_end)
    print(f'IS window   {a.is_start} .. {is_end}   quantiles [{q_lo_q}, {q_hi_q}]')
    print(f'{len(B1_PAIRS)} pairs x {len(a.tfs)} TFs = {len(B1_PAIRS) * len(a.tfs)} cells\n')

    t0 = time.time()
    print(f'loading {len(B1_SYMBOLS)} symbols from the spliced archive ...')
    px = {}
    for s in B1_SYMBOLS:
        px[s] = spliced_ohlc(s, a.perp_dir, a.combined_dir)
        print(f'  {s:<10} {px[s].index[0]} .. {px[s].index[-1]}  n={len(px[s]):,}')
    print(f'loaded in {time.time() - t0:.0f}s\n')

    rows = []
    for i, (c1, c2) in enumerate(B1_PAIRS, 1):
        raw = ratio_minute_frame(px[c1], px[c2], c1, c2)
        for j, tf in enumerate(a.tfs):
            rv = rv_raw(ratio_candles(raw, tf))
            lo, hi, n_is = is_bounds(rv, a.is_start, is_end, q_lo_q, q_hi_q)
            rows.append(dict(coin1=c1, coin2=c2, tf=int(tf),
                             q_lo=lo, q_hi=hi, n_is=n_is))
            if j == 0:
                seg = rv.loc[a.is_start:is_end]
                print(f'[{i:>2}/{len(B1_PAIRS)}] {c1}/{c2}   IS candles span '
                      f'{seg.index[0]} .. {seg.index[-1]}')
            print(f'         tf={tf:>3}  q_lo={lo:.6f}  q_hi={hi:.6f}  n_is={n_is:,}')

    out = pd.DataFrame(rows, columns=['coin1', 'coin2', 'tf', 'q_lo', 'q_hi', 'n_is'])
    print(f'\nbuilt {len(out)} rows in {time.time() - t0:.0f}s')

    if a.compare_to:
        _compare(out, a.compare_to)

    if a.dry_run:
        print('\n--dry-run: nothing written')
    else:
        write_parquet_atomic(out, a.out, index=False)
        print(f'\nwrote {a.out}')
    return 0


def _compare(out, path):
    """Diff against an existing bounds file. Advisory only -- never blocks the write."""
    if not os.path.exists(path):
        print(f'\n--compare-to: {path} does not exist, skipping')
        return
    old = pd.read_parquet(path)
    j = old.merge(out, on=['coin1', 'coin2', 'tf'], suffixes=('_old', '_new'), how='inner')
    print(f'\ncomparing against {path}  ({len(old)} rows there, {len(j)} overlap)')
    if len(j) != len(old):
        print(f'  note: {len(old) - len(j)} row(s) in that file have no counterpart here')
    bad = 0
    for _, r in j.iterrows():
        same = (r.q_lo_old == r.q_lo_new and r.q_hi_old == r.q_hi_new
                and r.n_is_old == r.n_is_new)
        bad += (not same)
        print(f'  {r.coin1}/{r.coin2} tf={r.tf:>3}  {"exact" if same else "*** DIFFERS"}')
        if not same:
            print(f'      q_lo {r.q_lo_old!r} -> {r.q_lo_new!r}')
            print(f'      q_hi {r.q_hi_old!r} -> {r.q_hi_new!r}')
            print(f'      n_is {r.n_is_old} -> {r.n_is_new}')
    print(f'  {len(j) - bad}/{len(j)} reproduced bitwise')


if __name__ == '__main__':
    sys.exit(main())
