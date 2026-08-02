"""Build the frozen per-(pair, TF) constants for pair_lead_lag (QR31_v2).

Per cell, all fitted once over the frozen IS window and constant for every bar
production will ever process (`generate_signal_pair_lead_lag.py::load_qr31_cell_constants`
and the streaming calcs read this artifact instead of recomputing expanding
quantiles into live data — the pre-2021 chain history is load-bearing, so the
chains here run from DATA START and only the SLICE is IS):

  atr14_lo/hi, atr50_lo/hi   final expanding min/max of the RAW Wilder ATR-% series
                             over IS (clip_atr_pct's frozen values; band AND kernel
                             trigger/stop distances)
  sizing_scale               IS-mean(RAW ATR75 %) / max(IS-mean(raw EWMA75 %), 1e-12)
  sizing_lo/hi               clip bounds of the SCALED sizing-vol series over IS

  Output: eod_scripts/pair_lead_lag/output/qr31_cell_constants.parquet
          (long: coin1, coin2, tf, atr14_lo, atr14_hi, atr50_lo, atr50_hi,
           sizing_scale, sizing_lo, sizing_hi, n_is -- 95 rows for the full universe)

  Run:    python eod_scripts/pair_lead_lag/build_qr31_cell_constants.py \
                 --pairs BTCUSDT_AVAXUSDT            # test cell only
          python eod_scripts/pair_lead_lag/build_qr31_cell_constants.py   # all 19 pairs

NOTE ON --is-end: vendored_pairs' clip_atr_pct slices with STRING labels, so the
default bare date '2025-03-31' INCLUDES that whole day (pandas partial-string
slicing) — the opposite of pair_momentum's Timestamp convention. The sibling
clip_is law (band ATRs in the v2 rebuild) cuts at 00:00 that day; --check-clip-is
recomputes the bounds under that window too and FAILS the build if any cell's
bounds differ between the two laws (they should not — a 4+-year min/max almost
never moves on the final day — but silence would be wrong).
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (QR31_PAIRS, ARTIFACT_DIR, IS_START_DEFAULT, IS_END_DEFAULT,
                     ATR_FAST, ATR_SLOW, DATA_PERP,
                     load_perp, ratio_minute_frame, ratio_candles,
                     wilder_atr_pct, is_final_bounds, sizing_scale_and_bounds,
                     write_parquet_atomic)

TFS = [15, 30, 60, 120, 240]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='+', default=None,
                    help='subset, e.g. BTCUSDT_AVAXUSDT (default: all 19 QR31 pairs)')
    ap.add_argument('--tfs', nargs='+', type=int, default=TFS)
    ap.add_argument('--is-start', default=IS_START_DEFAULT)
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help='STRING label — a bare date includes the whole day (see header)')
    ap.add_argument('--perp-dir', default=DATA_PERP)
    ap.add_argument('--out', default=os.path.join(ARTIFACT_DIR, 'qr31_cell_constants.parquet'))
    ap.add_argument('--check-clip-is', action='store_true', default=True)
    ap.add_argument('--no-check-clip-is', dest='check_clip_is', action='store_false')
    ap.add_argument('--compare-to', default=None,
                    help='optional existing parquet to diff against before writing')
    ap.add_argument('--merge-into-existing', action='store_true',
                    help='merge these rows into an existing artifact instead of replacing it')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pairs = QR31_PAIRS
    if a.pairs:
        want = {tuple(p.split('_')) for p in a.pairs}
        pairs = [p for p in QR31_PAIRS if p in want]
        missing = want - set(pairs)
        if missing:
            raise SystemExit(f'not in the QR31 universe: {sorted(missing)}')

    print(f'IS window   {a.is_start} .. {a.is_end} (string labels; end day INCLUDED)')
    print(f'{len(pairs)} pairs x {len(a.tfs)} TFs = {len(pairs) * len(a.tfs)} cells\n')

    t0 = time.time()
    syms = sorted({s for pr in pairs for s in pr})
    px = {}
    for s in syms:
        px[s] = load_perp(s, a.perp_dir)
        print(f'  {s:<10} {px[s].index[0]} .. {px[s].index[-1]}  n={len(px[s]):,}')
    print(f'loaded {len(syms)} PERP symbols in {time.time() - t0:.0f}s\n')

    is_end_ts = pd.Timestamp(a.is_end)      # the clip_is (band-law) window end, 00:00 cut
    rows = []
    n_law_diff = 0
    for i, (c1, c2) in enumerate(pairs, 1):
        raw = ratio_minute_frame(px[c1], px[c2])
        for tf in a.tfs:
            cd = ratio_candles(raw, tf)
            a14 = wilder_atr_pct(cd, ATR_FAST)
            a50 = wilder_atr_pct(cd, ATR_SLOW)
            a14_lo, a14_hi, n_is = is_final_bounds(a14, a.is_start, a.is_end)
            a50_lo, a50_hi, _ = is_final_bounds(a50, a.is_start, a.is_end)
            scale, s_lo, s_hi, _ = sizing_scale_and_bounds(cd, a.is_start, a.is_end)

            if a.check_clip_is:
                # clip_is law: same quantiles, window cut at Timestamp 00:00.
                for nm, ser, lo, hi in (('atr14', a14, a14_lo, a14_hi),
                                        ('atr50', a50, a50_lo, a50_hi)):
                    lo2, hi2, _ = is_final_bounds(ser, a.is_start, is_end_ts)
                    if lo2 != lo or hi2 != hi:
                        n_law_diff += 1
                        print(f'  *** {c1}/{c2} tf={tf} {nm}: clip_atr_pct vs clip_is '
                              f'bounds DIFFER: [{lo!r},{hi!r}] vs [{lo2!r},{hi2!r}]')

            rows.append(dict(coin1=c1, coin2=c2, tf=int(tf),
                             atr14_lo=a14_lo, atr14_hi=a14_hi,
                             atr50_lo=a50_lo, atr50_hi=a50_hi,
                             sizing_scale=scale, sizing_lo=s_lo, sizing_hi=s_hi,
                             n_is=n_is))
            print(f'[{i:>2}/{len(pairs)}] {c1}/{c2} tf={tf:>3}  '
                  f'atr14 [{a14_lo:.6f},{a14_hi:.6f}]  atr50 [{a50_lo:.6f},{a50_hi:.6f}]  '
                  f'scale {scale:.6f}  sizing [{s_lo:.6f},{s_hi:.6f}]  n_is={n_is:,}')

    if n_law_diff:
        raise SystemExit(f'\n{n_law_diff} cell(s) where the two clip laws disagree — '
                         f'the single-bounds artifact cannot represent them; escalate.')

    out = pd.DataFrame(rows, columns=['coin1', 'coin2', 'tf', 'atr14_lo', 'atr14_hi',
                                      'atr50_lo', 'atr50_hi', 'sizing_scale',
                                      'sizing_lo', 'sizing_hi', 'n_is'])
    print(f'\nbuilt {len(out)} rows in {time.time() - t0:.0f}s')

    if a.compare_to and os.path.exists(a.compare_to):
        old = pd.read_parquet(a.compare_to)
        j = old.merge(out, on=['coin1', 'coin2', 'tf'], suffixes=('_old', '_new'))
        exact = sum(all(r[f'{c}_old'] == r[f'{c}_new'] for c in
                        ['atr14_lo', 'atr14_hi', 'atr50_lo', 'atr50_hi',
                         'sizing_scale', 'sizing_lo', 'sizing_hi'])
                    for _, r in j.iterrows())
        print(f'compare-to {a.compare_to}: {exact}/{len(j)} overlapping rows bitwise')

    if a.merge_into_existing and os.path.exists(a.out):
        old = pd.read_parquet(a.out)
        keep = old.merge(out[['coin1', 'coin2', 'tf']], on=['coin1', 'coin2', 'tf'],
                         how='left', indicator=True)
        old = old[keep['_merge'] == 'left_only']
        out = pd.concat([old, out], ignore_index=True).sort_values(
            ['coin1', 'coin2', 'tf']).reset_index(drop=True)
        print(f'merged into existing artifact -> {len(out)} total rows')

    if a.dry_run:
        print('--dry-run: nothing written')
    else:
        write_parquet_atomic(out, a.out, index=False)
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
