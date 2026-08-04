"""Build the frozen per-(pair, TF) IS constants for pair_relative_value (QR1_v4).

Per cell:
    kv          EV10 sizing scale — run_cell's k_/kv chain over the IS window
                (algebraically IS-mean(clipped ATR75%) / IS-mean(EV10))
    wmed        wide-band gate median — IS median of the minute-projected band-width %
    atrN_lo/hi  frozen ATR %-clip bounds (final expanding IS min/max) for N=14/50/75

PRODUCTION recomputes all of this from 2021 on every run; live init data never reaches
the calibration window, so the live path reads these frozen constants instead
(`generate_signal_pair_relative_value.py::load_qr1_cell_constants`).

Source is tsdb `ohlcv_data` DIRECTLY — no splicing (QR1's frozen book was built on the
raw PERP series, NOT the spliced combined archive the B1/DMP builders use). The old PERP
parquet archive is no longer the source; see the DATA SOURCE note in _common.py for the
bitwise archive-vs-database verification behind that swap.

  Output: eod_scripts/pair_relative_value/qr1_cell_constants.parquet
          (long: coin1, coin2, tf, kv, wmed, atr14_lo/hi, atr50_lo/hi, atr75_lo/hi,
           n_is -- 75 rows)
          then copy to /home/rocky/crypto_sims/pair_relative_value_production/data/

  Run:    python eod_scripts/pair_relative_value/build_qr1_cell_constants.py
          python eod_scripts/pair_relative_value/build_qr1_cell_constants.py \
                 --pairs BTCUSDT_SOLUSDT --tfs 15 240 --dry-run

NOTE ON --is-end: passed through as a STRING label, because QR1's reference slices with
plain strings (`vendored_qr1.py:16: st, end_insample = '2021', '2025-03-31'`) and pandas
PARTIAL-STRING slicing includes the WHOLE end day (through 23:59). This differs from the
pair_momentum builder, whose reference compares against a Timestamp (day excluded at
00:00) — converting here to a Timestamp shifts wmed by ~5e-4 and kv by ~5e-5, which the
--compare-to check against the harness-built cells catches.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (QR1_PAIRS, ARTIFACT_DIR, TFS, NBDEV_PER_TF,
                     IS_START_DEFAULT, IS_END_DEFAULT,
                     load_minute_ohlc, ratio_minute_frame, kv_wmed_for_cell)


def _build_pair(args):
    """All TF cells for one pair (the ratio frame is the expensive shared piece).

    Each worker opens its OWN db connection (load_minute_ohlc with conn=None): a
    psycopg2 socket cannot be inherited across a fork/spawn boundary.
    """
    c1, c2, tfs, is_start, is_end = args
    d1, d2 = load_minute_ohlc(c1), load_minute_ohlc(c2)
    raw = ratio_minute_frame(d1, d2, c1, c2)
    rows = []
    for tf in tfs:
        r = kv_wmed_for_cell(raw, tf, NBDEV_PER_TF[tf], is_start, is_end)
        rows.append(dict(coin1=c1, coin2=c2, tf=int(tf), **r))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='+', default=None,
                    help='subset as COIN1_COIN2 (default: all 15 QR1 pairs)')
    ap.add_argument('--tfs', nargs='+', type=int, default=TFS)
    ap.add_argument('--is-start', default=IS_START_DEFAULT)
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help='IS window end, used as a STRING label — partial-string '
                         'slicing INCLUDES the whole end day (QR1 reference semantics)')
    ap.add_argument('--workers', type=int, default=8,
                    help='each worker pulls both legs of its pair in full from '
                         'ohlcv_data (~3.6M rows/leg), so this is the memory knob')
    ap.add_argument('--out', default=os.path.join(ARTIFACT_DIR, 'qr1_cell_constants.parquet'))
    ap.add_argument('--compare-to', default=None,
                    help='optional existing constants parquet to diff against before '
                         'writing; reports per-row exact/differs, never blocks the write')
    ap.add_argument('--dry-run', action='store_true', help='compute and report, write nothing')
    a = ap.parse_args()

    pairs = QR1_PAIRS
    if a.pairs:
        want = {tuple(p.split('_')) for p in a.pairs}
        pairs = [p for p in QR1_PAIRS if p in want]
        missing = want - set(pairs)
        if missing:
            raise SystemExit(f'not in the QR1 universe: {sorted(missing)}')
    for tf in a.tfs:
        if tf not in NBDEV_PER_TF:
            raise SystemExit(f'tf={tf} has no frozen nbdev (QR1 TFs: {TFS})')

    is_end = a.is_end          # STRING label on purpose — see the module docstring
    print(f'IS window   {a.is_start} .. {is_end} (end day INCLUDED — string-label slice)')
    print(f'{len(pairs)} pairs x {len(a.tfs)} TFs = {len(pairs) * len(a.tfs)} cells, '
          f'{a.workers} workers\n', flush=True)

    t0 = time.time()
    tasks = [(c1, c2, a.tfs, a.is_start, is_end) for c1, c2 in pairs]
    rows = []
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for i, pair_rows in enumerate(pool.imap(_build_pair, tasks), 1):
            rows.extend(pair_rows)
            r0 = pair_rows[0]
            print(f'[{i:>2}/{len(pairs)}] {r0["coin1"]}/{r0["coin2"]}  '
                  + '  '.join(f'tf={r["tf"]}: kv={r["kv"]:.4f} wmed={r["wmed"]:.4f}'
                              for r in pair_rows), flush=True)

    out = pd.DataFrame(rows, columns=['coin1', 'coin2', 'tf', 'kv', 'wmed',
                                      'atr14_lo', 'atr14_hi', 'atr50_lo', 'atr50_hi',
                                      'atr75_lo', 'atr75_hi', 'n_is'])
    print(f'\nbuilt {len(out)} rows in {time.time() - t0:.0f}s')

    bad = out[(out.kv <= 0) | (out.wmed <= 0) | (out.atr14_lo >= out.atr14_hi)
              | (out.atr50_lo >= out.atr50_hi) | (out.atr75_lo >= out.atr75_hi)]
    if len(bad):
        print(f'*** {len(bad)} row(s) fail sanity (kv>0, wmed>0, lo<hi):\n{bad}')

    if a.compare_to:
        _compare(out, a.compare_to)

    if a.dry_run:
        print('\n--dry-run: nothing written')
    else:
        out.to_parquet(a.out, index=False)
        print(f'\nwrote {a.out}')
    return 0


def _compare(out, path):
    """Diff against an existing constants file. Advisory only — never blocks the write."""
    if not os.path.exists(path):
        print(f'\n--compare-to: {path} does not exist, skipping')
        return
    old = pd.read_parquet(path)
    cols = [c for c in ['kv', 'wmed', 'atr14_lo', 'atr14_hi', 'atr50_lo', 'atr50_hi',
                        'atr75_lo', 'atr75_hi'] if c in old.columns]
    j = old.merge(out, on=['coin1', 'coin2', 'tf'], suffixes=('_old', '_new'), how='inner')
    print(f'\ncomparing against {path}  ({len(old)} rows there, {len(j)} overlap)')
    bad = 0
    for _, r in j.iterrows():
        diffs = [c for c in cols if getattr(r, f'{c}_old') != getattr(r, f'{c}_new')]
        bad += bool(diffs)
        print(f'  {r.coin1}/{r.coin2} tf={r.tf:>3}  '
              f'{"exact" if not diffs else "*** DIFFERS: " + ", ".join(diffs)}')
        for c in diffs:
            print(f'      {c} {getattr(r, f"{c}_old")!r} -> {getattr(r, f"{c}_new")!r}')
    print(f'  {len(j) - bad}/{len(j)} reproduced bitwise')


if __name__ == '__main__':
    sys.exit(main())
