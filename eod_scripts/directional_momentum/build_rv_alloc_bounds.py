"""Build the frozen per-(asset, TF) rv_alloc clip bounds for directional_momentum (DMP_v3_2).

`rv_alloc = clip(ewm(span=75).std(log-returns) * 100, q_lo, q_hi)`, where q_lo/q_hi are the
FINAL values of an expanding quantile fit over the in-sample window. PRODUCTION recomputes
that expanding quantile from 2021 on every run; live init data will never reach back that
far, so the live path reads the frozen constants from this artifact instead
(`generate_signal_directional_momentum.py::load_rv_alloc_bounds`).

ONE row per (asset, TF) — SHARED BY BOTH SIDES. The reference clips `rv` once in run_cell,
before either kernel is called; there is no per-side calibration.

Source is the spliced minute archive, not tsdb: these are frozen IS constants derived from
the definitional data, not live data.

  Output: eod_scripts/directional_momentum/output/rv_alloc_bounds.parquet
          (long: coin1, tf, q_lo, q_hi, n_is -- 44 rows)

  Run:    python eod_scripts/directional_momentum/build_rv_alloc_bounds.py
          python eod_scripts/directional_momentum/build_rv_alloc_bounds.py \
                 --is-start 2021-01-01 --is-end 2025-03-31 --quantiles 0.0 0.997

=============================================================================
NOTE ON --is-end: IT IS A **STRING** LABEL, NOT A TIMESTAMP.
=============================================================================
DMP slices `rv.loc['2021-01-01':'2025-03-31']` with strings, so the WHOLE of 2025-03-31 is
inside the fit. The sibling pair_momentum builder parses --is-end as a Timestamp, because
B1 uses `pd.Timestamp('2025-03-31')` and therefore STOPS at 00:00:00 that day. Measured on
BTCUSDT the two conventions differ by 287 IS candles at TF=5 (23 at TF=60) and move q_hi by
1.0e-04. `_common._check_label` rejects a --is-end carrying a time component so the B1
convention cannot be applied here by accident.

The script also cross-checks the R STATE's own 15-minute construction (which has no
ffill/bfill and reads Close only) against the TF=15 cell rows. They coincide on the archive;
the assert exists so that if a future data change breaks the coincidence, this fails loudly
instead of build_r_state silently drifting. Use --no-state-check to downgrade it to a warning.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (DMP_ASSETS, TFS, ARTIFACT_DIR, IS_START_DEFAULT, IS_END_DEFAULT,
                     Q_LO_DEFAULT, Q_HI_DEFAULT, DATA_PERP, DATA_COMBINED, R_STATE_TF,
                     spliced_ohlc, cell_candles, state_close_15m, rv_raw, is_bounds,
                     write_parquet_atomic)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--is-start', default=IS_START_DEFAULT,
                    help=f'IS window start, pandas .loc STRING label (default {IS_START_DEFAULT!r})')
    ap.add_argument('--is-end', default=IS_END_DEFAULT,
                    help=f'IS window end, a bare date STRING -- the whole day is INCLUDED '
                         f'(default {IS_END_DEFAULT!r}). Not a Timestamp; see the header.')
    ap.add_argument('--quantiles', nargs=2, type=float, metavar=('Q_LO', 'Q_HI'),
                    default=[Q_LO_DEFAULT, Q_HI_DEFAULT],
                    help=f'clip quantiles (default {Q_LO_DEFAULT} {Q_HI_DEFAULT})')
    ap.add_argument('--tfs', nargs='+', type=int, default=TFS)
    ap.add_argument('--assets', nargs='+', default=DMP_ASSETS)
    ap.add_argument('--perp-dir', default=DATA_PERP,
                    help='directory holding {SYM}PERP-1m-data.parquet')
    ap.add_argument('--combined-dir', default=DATA_COMBINED,
                    help='directory holding {SYM}-1m-combined.parquet')
    ap.add_argument('--out', default=os.path.join(ARTIFACT_DIR, 'rv_alloc_bounds.parquet'))
    ap.add_argument('--compare-to', default=None,
                    help='optional existing bounds parquet to diff against before writing; '
                         'reports per-row exact/differs and does not block the write')
    ap.add_argument('--no-state-check', action='store_true',
                    help='downgrade the R-state 15m cross-check from an error to a warning')
    ap.add_argument('--dry-run', action='store_true', help='compute and report, write nothing')
    a = ap.parse_args()

    q_lo_q, q_hi_q = a.quantiles
    print(f'IS window   {a.is_start!r} .. {a.is_end!r}  (STRING labels: the whole of '
          f'{a.is_end} is INCLUDED)   quantiles [{q_lo_q}, {q_hi_q}]')
    print(f'{len(a.assets)} assets x {len(a.tfs)} TFs = {len(a.assets) * len(a.tfs)} cells '
          f'(shared by both sides -> {len(a.assets) * len(a.tfs) * 2} streams)\n')

    t0 = time.time()
    rows, state_rows = [], []
    for i, sym in enumerate(a.assets, 1):
        raw = spliced_ohlc(sym, a.perp_dir, a.combined_dir)
        print(f'[{i:>2}/{len(a.assets)}] {sym:<10} {raw.index[0]} .. {raw.index[-1]}  n={len(raw):,}')
        for tf in a.tfs:
            rv = rv_raw(cell_candles(raw, tf)['close'])
            lo, hi, n_is = is_bounds(rv, a.is_start, a.is_end, q_lo_q, q_hi_q)
            rows.append(dict(coin1=sym, tf=int(tf), q_lo=lo, q_hi=hi, n_is=n_is))
            if tf == a.tfs[0]:
                seg = rv.loc[a.is_start:a.is_end]
                print(f'           IS candles span {seg.index[0]} .. {seg.index[-1]}')
            print(f'           tf={tf:>3}  q_lo={lo:.10f}  q_hi={hi:.10f}  n_is={n_is:,}')

        # The R state's own 15-minute series -- Close only, no ffill/bfill (build_r_state:37).
        srv = rv_raw(state_close_15m(raw['Close']))
        slo, shi, sn = is_bounds(srv, a.is_start, a.is_end, q_lo_q, q_hi_q)
        state_rows.append(dict(coin1=sym, q_lo=slo, q_hi=shi, n_is=sn))

    out = pd.DataFrame(rows, columns=['coin1', 'tf', 'q_lo', 'q_hi', 'n_is'])
    print(f'\nbuilt {len(out)} rows in {time.time() - t0:.0f}s')

    _state_check(out, pd.DataFrame(state_rows), a)

    if a.compare_to:
        _compare(out, a.compare_to)

    if a.dry_run:
        print('\n--dry-run: nothing written')
    else:
        write_parquet_atomic(out, a.out, index=False)
        print(f'\nwrote {a.out}')
    return 0


def _state_check(cells, state, a):
    """The R state builds its own 15-minute series (Close only, no ffill/bfill). On the
    spliced archive that coincides bitwise with the TF=15 cell series, which is what lets
    build_r_state read the tf=15 rows of this artifact instead of carrying a second one.
    Verify rather than assume -- the two constructions genuinely differ on any frame that
    HAS been reindexed onto a complete minute grid."""
    tf15 = cells[cells['tf'] == R_STATE_TF].set_index('coin1')
    j = state.set_index('coin1').join(tf15, lsuffix='_state', rsuffix='_cell', how='inner')
    bad = j[(j['q_lo_state'] != j['q_lo_cell']) | (j['q_hi_state'] != j['q_hi_cell'])
            | (j['n_is_state'] != j['n_is_cell'])]
    print(f'\nR-state 15m cross-check: {len(j) - len(bad)}/{len(j)} assets reproduce the '
          f'tf={R_STATE_TF} cell bounds bitwise')
    if len(bad) == 0:
        return
    for sym, r in bad.iterrows():
        print(f'  *** {sym}: state ({r.q_lo_state!r}, {r.q_hi_state!r}, n={r.n_is_state}) '
              f'vs cell ({r.q_lo_cell!r}, {r.q_hi_cell!r}, n={r.n_is_cell})')
    msg = ('the R state\'s 15-minute series no longer matches the TF=15 cell series. '
           'build_r_state reads the tf=15 rows of rv_alloc_bounds on the assumption that '
           'they do; give it its own bounds artifact before trusting the next R.')
    if a.no_state_check:
        print(f'  [warn] {msg}')
    else:
        raise SystemExit(f'  [FATAL] {msg}  (--no-state-check to override)')


def _compare(out, path):
    """Diff against an existing bounds file. Advisory only -- never blocks the write."""
    if not os.path.exists(path):
        print(f'\n--compare-to: {path} does not exist, skipping')
        return
    old = pd.read_parquet(path)
    j = old.merge(out, on=['coin1', 'tf'], suffixes=('_old', '_new'), how='inner')
    print(f'\ncomparing against {path}  ({len(old)} rows there, {len(j)} overlap)')
    if len(j) != len(old):
        print(f'  note: {len(old) - len(j)} row(s) in that file have no counterpart here')
    bad = 0
    for _, r in j.iterrows():
        same = (r.q_lo_old == r.q_lo_new and r.q_hi_old == r.q_hi_new
                and r.n_is_old == r.n_is_new)
        bad += (not same)
        print(f'  {r.coin1:<10} tf={r.tf:>3}  {"exact" if same else "*** DIFFERS"}')
        if not same:
            print(f'      q_lo {r.q_lo_old!r} -> {r.q_lo_new!r}')
            print(f'      q_hi {r.q_hi_old!r} -> {r.q_hi_new!r}')
            print(f'      n_is {r.n_is_old} -> {r.n_is_new}')
    print(f'  {len(j) - bad}/{len(j)} reproduced bitwise')


if __name__ == '__main__':
    sys.exit(main())
