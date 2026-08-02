"""Build the QR31_v2 BOOK OVERLAY: daily per-TF weights + book leverage.

The two book-level sizing layers PRODUCTION emits in golden_overlay_QR31_v2.parquet
and that live book construction must apply on top of the cell positions
(cell weight = 1/19 within TF x tf_weight[TF] x book_leverage):

    TSq[tf] = equal-weight mean of that TF's 19 cell DAILY NET RETURN streams
    rw      = (TSq.rolling(365).mean() / TSq.rolling(365).std() * sqrt(365)).clip(lower=0.2)
              -> resample('ME').first() -> reindex(daily).ffill() -> renormalise rows to 1
    s3      = (TSq * rw).sum(axis=1).replace(0, nan).dropna()
    lev     = (TV / s3.rolling(60).std().shift(1)).clip(0.5, 2.0).fillna(1.0)

Faithful transcription of PRODUCTION/engines/pairs_leadlag.py::run_cell (daily net
accounting, binance costs, R x I mult applied) + ::assemble (incl. its own
`overlay mirror broke` self-assert against the frozen VP.assemble book). TV is the
frozen constant vendored_pairs.TV. The mult series comes from OUR verified artifact
(bitwise == PRODUCTION build_mult on the whole frozen overlap), so the overlay
reproduces the goldens on their window and extends beyond the frozen LIVE_ASOF.

Costs are the frozen binance preset from the bundle (params.json) — this job never
imports PRODUCTION's config (the repo's `config` package would shadow it).

  Output: eod_scripts/pair_lead_lag/output/overlay_daily.parquet
          (index Timestamp; tf_weight_15/30/60/120/240, book_leverage —
           same schema as golden_overlay_QR31_v2.parquet)

  Run:    python eod_scripts/pair_lead_lag/build_overlay_daily.py
"""
import argparse
import json
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
PROD = '/home/rocky/crypto_sims/PRODUCTION'
BUNDLE = '/home/rocky/crypto_sims/pair_lead_lag_production'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, PROD)
from _common import ARTIFACT_DIR, DATA_PERP, write_parquet_atomic

_PARAMS = json.load(open(f'{BUNDLE}/params.json'))
PAIRS = [tuple(p) for p in _PARAMS['universe']['pairs_leg1_leg2']]
TFS = [int(t) for t in _PARAMS['timeframes_minutes']]
TXN_COST = float(_PARAMS['global_constants']['FIXED_COST_per_trade'])
SLIP = {k: float(v) for k, v in _PARAMS['slippage_per_symbol_fraction'].items()
        if k != 'note'}
MULT_PATH = os.path.join(ARTIFACT_DIR, 'mult_daily.parquet')


def cell_daily(args_):
    """One (pair, TF) cell -> daily net return Series. run_cell verbatim (with mult)."""
    a1, a2, t = args_
    try:
        import engines.core.vendored_pairs as VP
        from utils.pair_lead_lag_utils import cryptopairs_qr31v2_long_iact
        VP.INPUT = DATA_PERP
        q_ema, q_zatr, q_zema = 8, 14, 20        # config.QR31 EMA_SPAN/Z_ATR_LEN/Z_EMA_LEN
        raw = VP.load_raw(a1, a2, fix=True)
        cfgq = dict(VP.QR31_CFG[t])
        slip = (SLIP[a1] + SLIP[a2]) / 2
        P = VP.prep_cell_c(raw, a1, a2, t, cfgq, alloc_est='ewma')
        idx = pd.DatetimeIndex(P['index'] if 'index' in P else P['idx'])

        df = pd.DataFrame({'Open': raw['Open'], 'High': raw['minutely_high'],
                           'Low': raw['minutely_low'], 'Close': raw['Close']}).ffill().bfill()
        cand = VP.resample_pairs_data_closebased_perc(df, f'{t}T')
        cand = cand.loc[~cand.index.duplicated(), :]
        cand.rename(columns={'High': 'high', 'Open': 'open', 'Close': 'close', 'Low': 'low'},
                    inplace=True)

        def proj(s):
            x = pd.Series(np.nan, index=idx); c = s.index.intersection(idx); x.loc[c] = s.loc[c]
            return x.shift(t - 1).ffill().bfill().to_numpy()

        m = cand['close'].ewm(span=q_ema, adjust=False).mean()
        ap = VP.role_vol(cand, q_zatr, 'atr') * cand['close'] / 100
        a2p = VP.role_vol(cand, 50, 'atr') * cand['close'] / 100
        up = np.minimum(m + cfgq['nbdev'] * ap, m + cfgq['nbdev'] * a2p)
        zz = VP.compute_z_ema(cand, ema_len=q_ema, atr_len=q_zatr, z_ema_len=q_zema)
        middle = proj(m); upper = proj(up)
        z = np.nan_to_num(proj(zz), nan=-1.0)
        msig = np.where(z > 0.0, 1, -1).astype(np.float64)

        MJ = pd.read_parquet(MULT_PATH)
        mult = pd.Series(MJ['mult'].to_numpy(), index=pd.DatetimeIndex(MJ['date'])).sort_index()
        alloc = np.clip(P['alloc'] * mult.reindex(idx.normalize()).fillna(1.0).to_numpy(),
                        0.0, 2.0)

        upc = upper.copy()
        n = len(P['nc1'])
        K, _ = cryptopairs_qr31v2_long_iact(
            P['nc1'], P['sc1'], P['tp'], P['mh'], P['ml'], P['p1'],
            P['nc2'], P['sc2'], TXN_COST, msig, 1, upc, middle,
            P['atr'], P['atr2'], float(cfgq['nbdev']), slip, alloc,
            z, float(cfgq['z_thr']), float(cfgq['x_atr']), P['corr'],
            float(cfgq['c_thr']), VP.NEG, 0, 0, 0, 0, np.zeros(n, dtype=np.int8),
            0, 2, 0, 0, 0, upc)

        pnl, tc = VP.cell_pnl_tc(tuple(K), idx)
        g_ = pd.Series(pnl, index=idx).resample('D').sum()
        c_ = pd.Series(tc, index=idx).resample('D').sum()
        daily = ((np.exp(g_) - 1) - (np.exp(c_) - 1)).loc['2021-01-01':].dropna()
        return a1, a2, t, daily
    except Exception:
        import traceback
        return a1, a2, t, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nproc', type=int, default=10)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    import multiprocessing as mp
    import engines.core.vendored_pairs as VP

    tasks = [(a1, a2, t) for t in TFS for a1, a2 in PAIRS]
    print(f'daily streams: {len(tasks)} cells on {a.nproc} workers ...', flush=True)
    t0 = time.time()
    with mp.Pool(a.nproc, maxtasksperchild=1) as pool:
        res = pool.map(cell_daily, tasks)
    bad = [r for r in res if isinstance(r[3], str)]
    if bad:
        print(bad[0][3][:2000])
        raise SystemExit(f'cell FAILED: {bad[0][:3]}')
    print(f'daily streams done in {time.time()-t0:.0f}s', flush=True)

    cells = {f'{a1}_{a2}_{t}': d for a1, a2, t, d in res}

    # ---- assemble: PRODUCTION pairs_leadlag.py::assemble, verbatim -----------------
    cbt = {tf: [cells[k] for k in cells if k.endswith(f'_{tf}')] for tf in TFS}
    for tf in TFS:
        assert len(cbt[tf]) == len(PAIRS), (tf, len(cbt[tf]))
    book_ref = VP.assemble(cbt)
    TSq = pd.DataFrame({tf: pd.concat(cbt[tf], axis=1).mean(axis=1) for tf in TFS}).sort_index()
    rw = (TSq.rolling(365).mean() / TSq.rolling(365).std() * np.sqrt(365)).clip(lower=0.2)
    rw = rw.resample('ME').first().reindex(TSq.index).ffill()
    rw = rw.div(rw.sum(axis=1), axis=0)
    s3 = (TSq * rw).sum(axis=1).replace(0, np.nan).dropna()
    lev = (VP.TV / s3.rolling(60).std().shift(1)).clip(0.5, 2.0).fillna(1.0)
    chk = pd.concat({'a': s3 * lev, 'b': book_ref}, axis=1).dropna()
    mirror = float((chk.a - chk.b).abs().max())
    assert mirror < 1e-15, f'overlay mirror broke: {mirror}'
    print(f'assembled: overlay mirror max |diff| = {mirror:.2e} '
          f'(vs frozen VP.assemble book)', flush=True)

    overlay = rw.add_prefix('tf_weight_')
    overlay['book_leverage'] = lev
    overlay.index.name = 'Timestamp'
    overlay = overlay.dropna(subset=[f'tf_weight_{TFS[0]}'])
    print(f'overlay: {overlay.index[0].date()} .. {overlay.index[-1].date()}  '
          f'({len(overlay):,} days)  last lev={overlay["book_leverage"].iloc[-1]:.6f}',
          flush=True)

    if a.dry_run:
        print('--dry-run: nothing written')
        return 0
    out = os.path.join(ARTIFACT_DIR, 'overlay_daily.parquet')
    write_parquet_atomic(overlay, out)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
