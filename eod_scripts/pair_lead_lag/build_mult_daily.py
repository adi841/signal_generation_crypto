"""Build the daily R x I sizing multiplier for pair_lead_lag (QR31_v2).

    mult = min( (0.4 + 0.8*(1-pR)) * (0.75 + 1.0*(1-pI)), 2.0 ), capped at MULT_CAP=1.5
    pR   = expanding(120).rank(pct=True) of ewma_norm — the 10-symbol daily market-vol
           state (EWMA-10 of squared daily-close log-returns, basket mean, / rolling-60d
           median, shift 1 day)
    pI   = expanding(120).rank(pct=True) of int_entryrate7 — the 7-day mean (shift 1) of
           the strategy's internal ENTRY CLOCK: entries/day of the 17-pair x 6-TF EMA10
           base reference machine (prep_cell_c's own lines, sec_mode=2)

Faithful transcription of PRODUCTION/engines/pairs_leadlag.py::build_ewma_norm/build_mult
and the verified extension recipe (crypto_sims/work/2026-07-20_portfolio_ledger/
ext202607_run.py PASS 1+2, whose output IS the frozen bundle's activity tail).

ACTIVITY-COUNTS SPLICE DISCIPLINE (frozen, do not "fix"):
  * the FROZEN history (pair_lead_lag_production/data/leadlag_activity_counts.parquet,
    2021-01-01 .. 2026-07-22) is kept VERBATIM — its head is ledger-based (master_ledger
    entry times), which the transition counter cannot reproduce exactly (same-bar
    exit+re-entry is invisible to position transitions; measured ~0.7%/day, documented
    in ext202607_run.py). Extension is APPEND-ONLY beyond the frozen end.
  * the counter regenerates the full series each run; the segment where the frozen tail
    is itself transition-based (>= 2026-04-15) must reproduce EXACTLY — asserted.
  * entry TIMES are mult-INDEPENDENT (allocation appears in no kernel condition), so
    the counter runs mult-free — no circularity. Slippage DOES shift entry times (it
    is baked into the pivot -> exits -> arm cycles), so the counter uses the same
    slippage_ohlc4_dict values the frozen generator used.

MULT LAW (PRODUCTION run_cell): applied via reindex(idx.normalize()).fillna(1.0) —
NO ffill. Days missing from the artifact size at 1.0x; the live reader mirrors that
exactly (loudly when it means the EOD job is stale). The series is indexed by the day
each value APPLIES TO (both state legs carry their own shift(1)), and the newest data
day gets a row — there is no dropped-final-row trap here.

  Output: eod_scripts/pair_lead_lag/output/mult_daily.parquet        (date, mult)
          eod_scripts/pair_lead_lag/output/leadlag_activity_counts.parquet (spliced)

  Run:    python eod_scripts/pair_lead_lag/build_mult_daily.py                # full
          python eod_scripts/pair_lead_lag/build_mult_daily.py --skip-counts # reuse counts
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
## The crypto_sims tree (PRODUCTION + the frozen bundle) was retired at go-live.
## What each dependency became:
##   * minute closes (ewma_norm leg)  -> tsdb ohlcv_data (_common.load_minute_ohlc,
##     the substitution verified bitwise by the pair_relative_value builders);
##   * MULT_CAP                       -> submodel_parameters (the live source of truth);
##   * the frozen activity-counts head -> output/leadlag_activity_counts.parquet, which
##     already carries the bundle history VERBATIM plus the verified transition-based
##     extension — the artifact is now its own append-only base (and the ONLY on-disk
##     copy of the frozen history: keep it backed up);
##   * the fresh entry-clock counter (counts_cell) imported PRODUCTION's
##     vendored_pairs (load_raw / QR31_CFG / prep_cell_c) — that module is GONE, and
##     its per-TF base-machine constants are recorded nowhere else, so the counter is
##     UNAVAILABLE until the file is restored. Run with --skip-counts: mult stays
##     exact through counts_end + 1 day, and beyond that days go MISSING from the
##     artifact, which live sizes at 1.0 LOUDLY (the frozen no-ffill law) rather
##     than silently wrong.
PROD = '/home/rocky/crypto_sims/PRODUCTION'          # retired; see note above

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)          # our verified kernel (bitwise == frozen v16_flex_c)
from _common import ARTIFACT_DIR, connect_tsdb, load_minute_ohlc, write_parquet_atomic

# The ENTRY-CLOCK universe: the 17-pair (UNI-free) x 6-TF EMA10 base machine — the
# strategy's internal reference machine, NOT the traded 19-pair x 5-TF EMA8 book.
# Transcribed from work/2026-07-12_v16_review/ledger_ablation_run.py::PAIRS/TFS.
CLOCK_PAIRS = [('BTCUSDT', 'AVAXUSDT'), ('BTCUSDT', 'BNBUSDT'), ('BTCUSDT', 'DOTUSDT'),
               ('BTCUSDT', 'SOLUSDT'), ('BTCUSDT', 'ETHUSDT'), ('BTCUSDT', 'LINKUSDT'),
               ('BTCUSDT', 'XRPUSDT'), ('BTCUSDT', 'ADAUSDT'), ('BTCUSDT', 'DOGEUSDT'),
               ('ETHUSDT', 'BNBUSDT'), ('ETHUSDT', 'DOTUSDT'), ('ETHUSDT', 'SOLUSDT'),
               ('ETHUSDT', 'AVAXUSDT'), ('ETHUSDT', 'LINKUSDT'), ('ETHUSDT', 'XRPUSDT'),
               ('ETHUSDT', 'ADAUSDT'), ('ETHUSDT', 'DOGEUSDT')]
CLOCK_TFS = [5, 15, 30, 60, 120, 240]
CLOCK_FIXED_COST = 0.000035           # ledger_ablation_run.fixed_cost (decision-inert)
# main_util_mainmachine.slippage_ohlc4_dict — the frozen generator's slippage (values
# happen to equal the binance preset; kept as its own constant because THIS machine's
# provenance is the ledger build, not the venue preset).
CLOCK_SLIPPAGE = {'BTCUSDT': 0.000055, 'ETHUSDT': 0.000055, 'SOLUSDT': 0.0001,
                  'DOGEUSDT': 0.0001, 'XRPUSDT': 0.0001, 'ADAUSDT': 0.0001,
                  'BNBUSDT': 0.0001, 'LINKUSDT': 0.0001, 'AVAXUSDT': 0.0001,
                  'DOTUSDT': 0.0001}

# The 10-symbol market-vol basket (PRODUCTION pairs_leadlag.STATE_SYMS; DOT in, UNI out).
STATE_SYMS = ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'BNBUSDT', 'DOTUSDT',
              'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']
DATA_PERP = '/home/rocky/crypto_data/'   # surviving (stale) archive; counter-only
TRANSITION_TAIL_START = '2026-04-15'  # frozen tail is transition-based from here on


def counts_cell(args_):
    """Entry clock for one (pair, TF) cell: daily entry counts from 2021-01-01.
    runQ2-verbatim kernel call (mult-free — entry times are mult-independent)."""
    a1, a2, t = args_
    try:
        import engines.core.vendored_pairs as VP
        from utils.pair_lead_lag_utils import cryptopairs_qr31v2_long_iact
        VP.INPUT = DATA_PERP
        raw = VP.load_raw(a1, a2, fix=True)
        cfgq = dict(VP.QR31_CFG[t])
        P = VP.prep_cell_c(raw, a1, a2, t, cfgq, alloc_est='ewma')
        idx = pd.DatetimeIndex(P['index'] if 'index' in P else P['idx'])
        slip = (CLOCK_SLIPPAGE[a1] + CLOCK_SLIPPAGE[a2]) / 2
        up = P['upper'].copy()
        n = len(P['nc1'])
        K, _ = cryptopairs_qr31v2_long_iact(
            P['nc1'], P['sc1'], P['tp'], P['mh'], P['ml'], P['p1'],
            P['nc2'], P['sc2'], CLOCK_FIXED_COST, P['msig'], 1, up, P['middle'],
            P['atr'], P['atr2'], float(cfgq['nbdev']), slip, P['alloc'],
            P['z'], float(cfgq['z_thr']), float(cfgq['x_atr']), P['corr'],
            float(cfgq['c_thr']), VP.NEG, 0, 0, 0, 0, np.zeros(n, dtype=np.int8),
            0, 2, 0, 0, 0, up)
        d = np.diff(K[0], prepend=0.0)
        ent = idx[np.where(d > 0.5)[0]]
        ent = ent[ent >= pd.Timestamp('2021-01-01')]
        return f'{a1}_{a2}', t, pd.Series(1, index=ent).resample('D').sum()
    except Exception:
        import traceback
        return f'{a1}_{a2}', t, traceback.format_exc()


def build_counts(nproc):
    import multiprocessing as mp
    try:
        sys.path.insert(0, PROD)
        import engines.core.vendored_pairs  # noqa: F401 — availability probe only
    except ImportError:
        raise SystemExit(
            'entry-clock counter UNAVAILABLE: PRODUCTION/engines/core/vendored_pairs.py '
            'was retired with the crypto_sims tree, and its QR31_CFG base-machine '
            'constants are recorded nowhere else. Restore that file (any backup of '
            'crypto_sims/PRODUCTION) to extend activity counts, or run with '
            '--skip-counts: mult stays exact through counts_end + 1 day, after which '
            'missing days size at 1.0 loudly in live (frozen no-ffill law).')
    tasks = [(a1, a2, t) for t in CLOCK_TFS for a1, a2 in CLOCK_PAIRS]
    print(f'entry clock: {len(tasks)} cells on {nproc} workers ...', flush=True)
    t0 = time.time()
    with mp.Pool(nproc, maxtasksperchild=1) as pool:
        res = pool.map(counts_cell, tasks)
    bad = [(p, t, x) for p, t, x in res if isinstance(x, str)]
    if bad:
        print(bad[0][2][:2000])
        raise SystemExit(f'entry clock FAILED for {bad[0][:2]}')
    fresh = pd.concat([x for _, _, x in res], axis=1).sum(axis=1).astype(float)
    fresh = fresh.asfreq('D').fillna(0.0)
    print(f'entry clock done in {time.time()-t0:.0f}s: {fresh.index[0].date()} .. '
          f'{fresh.index[-1].date()}, mean {fresh.mean():.1f}/day', flush=True)
    return fresh


def splice_counts(fresh):
    """Frozen history verbatim + fresh tail beyond it, with the two overlap checks.

    The base is our own output artifact — the bundle copy retired with crypto_sims,
    and the artifact carries that history VERBATIM (splice discipline) plus the
    verified transition-based extension, so it is its own append-only base. Check 1
    below therefore hardens over time: every previously-appended day must reproduce."""
    base_path = os.path.join(ARTIFACT_DIR, 'leadlag_activity_counts.parquet')
    if not os.path.exists(base_path):
        raise SystemExit(f'{base_path} missing — it is the only copy of the frozen '
                         f'activity-counts history; restore it from backup before '
                         f'splicing (never rebuild the ledger-based head).')
    frozen = pd.read_parquet(base_path)['entries']
    frozen.index = pd.DatetimeIndex(frozen.index)

    # 1. The transition-based frozen tail must reproduce EXACTLY.
    tail = frozen.loc[TRANSITION_TAIL_START:]
    ovl = fresh.reindex(tail.index)
    tail_max = float((ovl - tail).abs().max())
    print(f'overlap vs frozen TRANSITION tail ({TRANSITION_TAIL_START}..'
          f'{tail.index[-1].date()}): max |diff| = {tail_max}', flush=True)
    if tail_max != 0.0:
        raise SystemExit('entry clock does not reproduce the frozen transition tail — '
                         'the generator recipe has drifted; do not splice.')

    # 2. The ledger-based head differs by construction (~0.7%/day) — informational.
    head = frozen.loc[:TRANSITION_TAIL_START].iloc[:-1]
    ovh = fresh.reindex(head.index)
    print(f'overlap vs frozen LEDGER head: mean |diff| = '
          f'{float((ovh - head).abs().mean()):.2f}/day '
          f'(same-bar exit+re-entry is invisible to transitions; frozen head kept)',
          flush=True)

    ext = fresh[fresh.index > frozen.index[-1]]
    out = pd.concat([frozen, ext]).asfreq('D').fillna(0.0)
    out.name = 'entries'
    print(f'spliced counts: {out.index[0].date()} .. {out.index[-1].date()} '
          f'({len(ext)} appended day(s))', flush=True)
    return out


def build_mult(counts, mult_cap):
    """PRODUCTION build_ewma_norm + build_mult, verbatim — closes now from tsdb
    ohlcv_data (archive-shaped: naive-UTC index, float64; the daily resample lands on
    the same UTC day boundaries the parquet read produced)."""
    ew = {}
    conn = connect_tsdb()
    try:
        for s in STATE_SYMS:
            px = load_minute_ohlc(s, conn=conn)['Close']
            d = px.resample('D').last()
            ew[s] = np.sqrt((np.log(d).diff() ** 2).ewm(span=10, adjust=False).mean())
    finally:
        conn.close()
    en = pd.DataFrame(ew).mean(axis=1).pipe(lambda m: (m / m.rolling(60).median()).shift(1))

    ## fillna(0.0) runs FIRST, over the counter's true range only (interior gaps are
    ## genuine zero-entry days). The grid is then EXTENDED to the close series'
    ## horizon so shift(1) can land the final rolling value on counts_end+1 —
    ## PRODUCTION's nightly counter always provided that label, and day D's value
    ## uses counts <= D-1 only, so this fabricates nothing: padded days are NaN,
    ## every day beyond counts_end+1 rolls a NaN window and drops in dropna().
    i7_src = counts.asfreq('D').fillna(0.0)
    i7_src = i7_src.reindex(pd.date_range(i7_src.index[0], en.index[-1], freq='D'))
    i7 = i7_src.rolling(7).mean().shift(1)
    X = pd.DataFrame(index=en.index)
    X['pR'] = en.expanding(120).rank(pct=True)
    X['pI'] = i7.reindex(en.index).expanding(120).rank(pct=True)
    mult_raw = np.clip((0.4 + 0.8 * (1 - X['pR'])) * (0.75 + 1.0 * (1 - X['pI'])), None, 2.0)
    return mult_raw.clip(upper=mult_cap)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-counts', action='store_true',
                    help='reuse output/leadlag_activity_counts.parquet instead of '
                         're-running the 102-cell entry clock')
    ap.add_argument('--nproc', type=int, default=10)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    ## MULT_CAP from submodel_parameters — the live source of truth (the bundle
    ## params.json retired with crypto_sims). All 95 live rows carry the one frozen
    ## value; DISTINCT + the length assert make any drift loud.
    conn = connect_tsdb()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT DISTINCT (model_parameters->>'mult_cap')::float8
                       FROM submodel_parameters
                       WHERE model_parameters->>'strategy_name' = 'pair_lead_lag'
                         AND (model_parameters->>'is_live')::int = 1""")
        caps = [r[0] for r in cur.fetchall()]
        cur.close()
    finally:
        conn.close()
    assert len(caps) == 1 and caps[0] is not None, \
        f'expected exactly one live mult_cap in submodel_parameters, got {caps}'
    mult_cap = float(caps[0])

    counts_path = os.path.join(ARTIFACT_DIR, 'leadlag_activity_counts.parquet')
    if a.skip_counts and os.path.exists(counts_path):
        counts = pd.read_parquet(counts_path)['entries']
        counts.index = pd.DatetimeIndex(counts.index)
        print(f'--skip-counts: reusing {counts_path} '
              f'({counts.index[0].date()} .. {counts.index[-1].date()})', flush=True)
    else:
        counts = splice_counts(build_counts(a.nproc))

    mult = build_mult(counts, mult_cap).dropna()
    print(f'mult: {mult.index[0].date()} .. {mult.index[-1].date()}  '
          f'last={mult.iloc[-1]:.6f}  median={mult.median():.4f}  '
          f'range=[{mult.min():.4f}, {mult.max():.4f}]', flush=True)

    if a.dry_run:
        print('--dry-run: nothing written')
        return 0
    write_parquet_atomic(counts.to_frame('entries'), counts_path)
    write_parquet_atomic(mult.rename('mult').rename_axis('date').reset_index(),
                         os.path.join(ARTIFACT_DIR, 'mult_daily.parquet'), index=False)
    print(f'wrote {counts_path}')
    print(f'wrote {os.path.join(ARTIFACT_DIR, "mult_daily.parquet")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
