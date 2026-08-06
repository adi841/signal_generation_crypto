"""Backtest signal producer for pair_momentum (B1_v8_v10).

The B1 twin of backtest_codes/pair_relative_value_backtest.py: a long-lived process that
re-derives the signal series with BACKTEST (vectorised, whole-window) methodology on a
timer and upserts the newest rows into `backtest_signals`, so the backtest and live
streams can be diffed per parent_trading_model.

Like the PRV script -- and unlike the older reversal one -- this does NOT open-code the
strategy math. The sleeve's batch path already exists and is matched bitwise against the
frozen PRODUCTION reference over full history, so we import it:

    GetPairsMomentumSignal.get_numba_parameters  ->  cryptopairs_b1v8v10_long_iact

which is exactly what PRODUCTION's pairs_momentum.run_cell drives. No `_ll` kernel, no
live incremental state: the whole point is that this is the batch path.

SINGLE STREAM. B1 is long-only with one signal series. There is no res2, no
tradeprice3/4, no order_tag2, no `parent_id + 2` scale-in row -- so one frame per cell,
where PRV emits two. Parent ids are the 4xxxxxxx band, stride 2 per TF (the +1 slots are
spare); the DB carries globally unique ids, the frozen bundle carries per-pair LOCAL ones.

R. The entry-sizing multiplier is applied inside get_numba_parameters, so it rides along
for free -- but it cannot affect anything published here. `alloc` has exactly one consumer
in the kernel (`tal[i] = alloc[i]*w`) and appears in no condition, and `backtest_signals`
has no size column, so signal/price/case/order_tag are all R-independent. The artifact is
still read on the normal code path: this script must run what live runs, and a bypass
would manufacture the very divergence it exists to measure.

RECOMPUTE WINDOW. Each pass recomputes a trailing `--lookback_days` window (default 300)
rather than all history. The sleeve's warm-up study measured the slowest state (TF120)
converged by 120 days, so 300 has 2.5x margin. `--lookback_days 0` means "everything
available", for parity runs.

  Run:  python backtest_codes/pair_momentum_backtest.py --hist_replay 0 \
            --cores 8 --ffill_data 0 --curr_time "2026-08-05 10:00:00" --interval_minutes 5

        python backtest_codes/pair_momentum_backtest.py --hist_replay 1 \
            --cores 4 --ffill_data 0 --from_bundle 1 --exec_type 2 \
            --pairs BTCUSDT_AVAXUSDT \
            --hist_replay_dir ./backtest_codes/data_pm \
            --init_hist_replay_dir ./backtest_codes/data_pm
        (hist_replay never writes to the DB, exactly like the reference)

NOTE on --hist_replay 0: it needs the PROD config chain, i.e. CLIENT_ID set. Without it,
utils/base.py stubs get_dealer_socket() to a zero-arg function that prepare_socket calls
with three, there is no config_object.zmq_data, and Base.get_init_data references an
absent config_object.lookback_days -- so DataClassBacktest cannot construct in local mode.
With CLIENT_ID set the live path runs: verified against the live feed on the full 17-pair
universe (see the smoke-test notes in the sleeve's session log).
"""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from collections import namedtuple
from os.path import abspath, dirname
from queue import Queue

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

file_path = dirname(abspath(__file__))
while True:
    if file_path.endswith("signal_generation_crypto"):
        break
    file_path = dirname(file_path)
sys.path.append(file_path)

from config.config_read import config_object
## backtest_utils is imported lazily (live path only) -- it pulls in the prod config
## chain, and hist_replay runs have no need of it.
from last_line_utils.pair_momentum_utils.pair_momentum_utils import (
    EOD_OUTPUT_DIR,
    PAIR_MOMENTUM_PARAM_TUPLE,
    PAIR_MOMENTUM_PARAMS_PATH,
    build_pair_momentum_parameter_dict,
    build_pair_momentum_parameter_dict_from_db,
)
from last_line_utils.pair_momentum_utils.generate_signal_pair_momentum import (
    GetPairsMomentumSignal)
from utils.pair_momentum_utils import cryptopairs_b1v8v10_long_iact
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
STRATEGY_NAME = "pair_momentum"

## backtest_signals.execution_type is the execution_type_checks DOMAIN, which the live DB
## has widened to {1..7}. The PRV script still asserts {1,2,3,4} and would reject a
## legitimate 6/7; crypto-infra/db_scripts/utils_params_dump.py already allows the wider set.
EXEC_TYPE_DOMAIN = (1, 2, 3, 4, 5, 6, 7)

ARGS_TUP = namedtuple("ARGS_TUP", [
    "lookback_days", "db_insert_row_count", "hist_replay", "database_name"])


# ---------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------
def _sleeve_paths():
    """sleeve_config for this sleeve, with the two artifact paths defaulted.

    build_pair_momentum_parameter_dict_from_db hard-indexes `rv_alloc_bounds_path` and
    `r_state_path`. The prod client config carries both; config/local_config.json carries
    only `strategy` and `params_path`, so a local run would KeyError. Default the missing
    keys to the same values the bundle builder uses rather than editing shared config.
    """
    paths = dict(config_object.sleeve_config[STRATEGY_NAME])
    paths.setdefault("params_path", PAIR_MOMENTUM_PARAMS_PATH)
    paths.setdefault("rv_alloc_bounds_path", f"{EOD_OUTPUT_DIR}/rv_alloc_bounds.parquet")
    paths.setdefault("r_state_path", f"{EOD_OUTPUT_DIR}/r_state_daily.parquet")
    return paths


def load_cells_from_db(database_name, pairs=None, parent_strat_ids=None):
    """Live's source of truth: submodel_parameters (is_live = 1). Returns
    [(PARAM_TUPLE, exec_type)] -- exec_type rides alongside because it lives in the raw
    model_parameters JSON and is NOT a field of PAIR_MOMENTUM_PARAM_TUPLE."""
    from shared_codes.utils.config_utils import connect_postgre
    conn, cursor = connect_postgre(db_user="signal_generation", database=database_name)
    try:
        cursor.execute(
            """SELECT parent_trading_model, model_parameters FROM submodel_parameters
               WHERE model_parameters->>'strategy_name' = %s
                 AND (model_parameters->>'is_live')::int = 1""", (STRATEGY_NAME,))
        rows = cursor.fetchall()
    finally:
        conn.close(); cursor.close()
    assert rows, f"no live submodel_parameters rows for {STRATEGY_NAME}"

    by_pair = {}
    for row in rows:
        mp_ = row['model_parameters']
        by_pair.setdefault((mp_['coin1'], mp_['coin2']), {})[row['parent_trading_model']] = mp_

    sleeve_paths = _sleeve_paths()
    cells = []
    for (c1, c2), db_rows in sorted(by_pair.items()):
        if pairs and f"{c1}_{c2}" not in pairs:
            continue
        pdict = build_pair_momentum_parameter_dict_from_db(db_rows, c1, c2, sleeve_paths)
        for pid, tup in sorted(pdict.items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, _exec_type_of(db_rows[pid], pid)))
    assert cells, "no cells selected -- check --pairs / --parent_strat_ids"
    return cells


def load_cells_from_bundle(pairs=None, parent_strat_ids=None):
    """Offline / full-universe path: the frozen bundle, per-pair LOCAL ids. exec_type is
    absent from the bundle, so it must be supplied with --exec_type."""
    with open(PAIR_MOMENTUM_PARAMS_PATH) as f:
        universe = [tuple(p) for p in json.load(f)["universe"]["pairs_leg1_leg2"]]
    cells = []
    for c1, c2 in universe:
        if pairs and f"{c1}_{c2}" not in pairs:
            continue
        for pid, tup in sorted(build_pair_momentum_parameter_dict(c1, c2).items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, None))
    assert cells, "no cells selected -- check --pairs / --parent_strat_ids"
    return cells


def _exec_type_of(model_parameters, pid):
    """backtest_signals.execution_type is NOT NULL and domain-checked. Fail loudly rather
    than inventing a default."""
    v = model_parameters.get('exec_type')
    assert v is not None, f"{pid}: model_parameters has no exec_type (execution_type is NOT NULL)"
    v = int(v)
    assert v in EXEC_TYPE_DOMAIN, \
        f"{pid}: exec_type {v} outside the execution_type_checks domain {EXEC_TYPE_DOMAIN}"
    return v


# ---------------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------------
def load_replay_data(init_dir, symbols, curr_time=None):
    """hist_replay data straight from the `{SYM}_input.parquet` files.

    DataClassBacktest is deliberately NOT used on this path: its __init__ calls
    Base.prepare_socket(), which cannot run in local/test mode; and its replay loop only
    publishes a snapshot every 4th day at 15:30, which buys nothing for a whole-window
    recompute. The live path still uses it, where that machinery is real."""
    data = {}
    for sym in symbols:
        d = pd.read_parquet(os.path.join(init_dir, f"{sym}_input.parquet"))
        d.index.name = 'Timestamp'
        if d.index.tz is None:
            d.index = d.index.tz_localize('UTC')
        if curr_time is not None:
            d = d.loc[:curr_time]
        assert len(d), f'{sym}: no rows at or before {curr_time}'
        data[sym] = d[OHLCV_COLS]
    return data


def _comb_frame(df1, df2):
    """The pair-ratio frame, identical to PairMomentumSignalGenerator.prepare_init_data."""
    comb = df1.merge(df2, left_index=True, right_index=True, suffixes=('_1', '_2'), how='outer')
    comb['open'] = comb['open_1'] / comb['open_2']
    comb['close'] = comb['close_1'] / comb['close_2']
    comb['volume'] = comb['volume_1'] + comb['volume_2']
    comb['high'] = np.max(comb[["open", "close"]], axis=1)
    comb['low'] = np.min(comb[["open", "close"]], axis=1)
    return comb.fillna(method='ffill').fillna(method='bfill')


def _price_and_tags(res, tp1, tp2, ts, parent_id):
    """price + order_tag on THIS sleeve's batch convention.

    Transcribed from generate_signal_pair_momentum.py:238-256: both are computed only on
    bars where the signal CHANGES, written back, then forward-filled. Deliberately not
    PRV's every-bar ratio -- "the methodology used to backtest" is this sleeve's own batch
    definition.

    Note `signal.diff().fillna(0)`, so the window's FIRST bar never counts as a change
    (PRV's np.diff(prepend=0) does). A trailing window that opens mid-position therefore
    ffills from nothing until the first real change; the caller turns those into the ""
    sentinel, because order_tag is NOT NULL.
    """
    sig = pd.Series(np.asarray(res, dtype=np.float64), index=ts)
    changed = sig.diff().fillna(0) != 0

    price = pd.Series(np.nan, index=ts, dtype=float)
    tags = pd.Series(np.nan, index=ts, dtype=object)
    if changed.any():
        num = np.asarray(tp1, dtype=np.float64)[changed.to_numpy()]
        den = np.asarray(tp2, dtype=np.float64)[changed.to_numpy()]
        price.loc[changed] = np.where(den > 0, num / den, np.nan)
        tags.loc[changed] = [
            order_tag_generator(ts=t, parent_id=int(parent_id), signal=int(s))
            for t, s in zip(ts[changed.to_numpy()], sig[changed].to_numpy())
        ]
    return price.ffill(), tags.ffill()


def run_cell(args_tup, data, tup: PAIR_MOMENTUM_PARAM_TUPLE, exec_type):
    """Derive one (pair, TF) cell over the trailing window; return one `upsert`-shaped frame."""
    raw1 = data[tup.coin1]
    raw2 = data[tup.coin2]
    end = min(raw1.index[-1], raw2.index[-1])
    start = max(raw1.index[0], raw2.index[0])
    if args_tup.lookback_days:
        start = max(start, end - dt.timedelta(days=args_tup.lookback_days))

    df1 = raw1.loc[start:end, OHLCV_COLS]
    df2 = raw2.loc[start:end, OHLCV_COLS]
    comb = _comb_frame(df1, df2)

    ## log_r_staleness=False: R rides the normal code path (see the R note in the module
    ## docstring) but cannot reach any column this script publishes, so a stale artifact is
    ## not an incident HERE -- it is one for live.py, which sizes off the same number. Left
    ## on, this would emit 85 CRITICALs per pass into the monitoring system for a condition
    ## this process cannot suffer from, and that is how the real alert gets ignored.
    sig_gen = GetPairsMomentumSignal(tup=None, df1=df1, df2=df2, comb_df=comb,
                                     curr_time=None, process_name="pm_backtest",
                                     log_r_staleness=False)
    out, aux = sig_gen.get_numba_parameters(tup, None)
    (next1, sc1, p1, mlow, next2, sc2, spc, med, upper, atr, lv, zmed, tc, slip, alloc) = out
    sc = tup.strategy_config

    ## tc / slip are SCALARS in this sleeve's output_tup (PRV passes series there).
    array_output, _ = cryptopairs_b1v8v10_long_iact(
        next1, sc1, p1, mlow, next2, sc2, spc, med, upper, atr, lv, zmed,
        tc, slip, alloc,
        float(tup.T1), float(tup.TT), float(tup.DDE), float(sc['GRACE']),
        float(tup.K), float(tup.X), float(tup.Z), float(tup.EZ),
    )

    ts = pd.to_datetime(aux[0])
    pid = int(tup.parent_stratid)
    res = np.asarray(array_output.res_arr, dtype=np.float64)
    price, tags = _price_and_tags(res, array_output.tradeprice1_arr,
                                  array_output.tradeprice2_arr, ts, pid)

    df = pd.DataFrame({
        'Timestamp': ts,
        'parent_trading_model': pid,
        'signal': res.astype(np.int64),
        'tradeprice1': price.to_numpy(),
        'case': 0,                       # B1 has no case output; the column is NOT NULL
        'execution_type': exec_type,
        'order_tag1': tags.to_numpy(),
    })
    df = df.iloc[-args_tup.db_insert_row_count:].copy()
    df['signal_id'] = df['Timestamp'].map(lambda x: int(x.timestamp()))

    ## order_tag is NOT NULL. Rows before the window's first signal change have nothing to
    ## ffill from. Publish them under the EMPTY STRING -- the same "no tag yet" sentinel
    ## this sleeve's own live path uses (generate_signal_pair_momentum.py::generate_signal,
    ## `order_tag1 = ""`), and what the PRV producer writes. Dropping the rows instead
    ## would make the row counts of the two streams disagree for exactly the cells whose
    ## signal is most static, which is the wrong thing to hide from a diff. Still logged:
    ## a whole trailing window with no signal change is worth knowing about.
    null_tag = df['order_tag1'].isna()
    if null_tag.any():
        msg_ = (f'{pid} {tup.coin1}/{tup.coin2} tf={tup.tf}: {int(null_tag.sum())} of '
                f'{len(df)} published rows have no order_tag (no signal change in the '
                f'{args_tup.lookback_days or "full"}-day window before them) -- published '
                f'with the "" sentinel')
        print(f'[CRITICAL] {msg_}', flush=True)
        config_object.masterlog.get_logger('pm_backtest', 'critical')(msg_)
        df.loc[null_tag, 'order_tag1'] = ''

    del sig_gen, out, aux, array_output, comb, df1, df2
    return df.reset_index(drop=True)


def run_pair_wrapper(args_tup, data, cell_payloads):
    """All TF cells of ONE pair, in one worker task.

    Two reasons this is per-PAIR rather than per-cell:
      * PAIR_MOMENTUM_PARAM_TUPLE cannot cross a `spawn` boundary -- the namedtuple's
        typename ("pair_momentum_param_tuple") differs from its module attribute name
        (PAIR_MOMENTUM_PARAM_TUPLE), so pickle's attribute lookup fails with
        "Can't pickle ... attribute lookup pair_momentum_param_tuple ... failed".
        Cells therefore travel as plain dicts and are rebuilt here.
      * `data` is pickled once per task; sending only this pair's two frames, once for all
        5 TFs, avoids copying the whole universe 5x per pair.
    Returns a list of frames; a cell that raises contributes nothing but never takes the
    pass down with it."""
    out = []
    for cell_dict, exec_type in cell_payloads:
        tup = PAIR_MOMENTUM_PARAM_TUPLE(**cell_dict)
        try:
            out.append(run_cell(args_tup, data, tup, exec_type))
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"{tup.parent_stratid} {tup.coin1}/{tup.coin2} tf={tup.tf}: {e}\n{tb_}"
            print(f"[ERROR] {msg_}", flush=True)
            config_object.masterlog.get_logger("pm_backtest", "error")(msg_)
    return out


# ---------------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------------
def upsert_signals(data_df: pd.DataFrame, database_name=None):
    """Same statement/contract as backtest_utils.upsert_data, but honouring
    `database_name` on the WRITE as well -- the shared helper accepts it only on the param
    read, so `--database_name uat_tsdb` there still writes to the default DB."""
    from shared_codes.utils.config_utils import connect_postgre
    conn, cursor = connect_postgre(db_user="signal_generation", database=database_name)

    query = """
    INSERT INTO backtest_signals (signal_floor_time, parent_trading_model, signal_id, signal, case_num, price, execution_type, order_tag)
    VALUES %s
    ON CONFLICT (parent_trading_model, signal_floor_time)
    DO UPDATE SET
        signal = EXCLUDED.signal,
        price = EXCLUDED.price,
        execution_type = EXCLUDED.execution_type,
        order_tag = EXCLUDED.order_tag;
    """

    data_tuples = [
        (
            row['Timestamp'],
            row['parent_trading_model'],
            row['signal_id'],
            row['signal'],
            row['case'],
            None if pd.isna(row['tradeprice1']) else row['tradeprice1'],
            row['execution_type'],
            '' if pd.isna(row['order_tag1']) else row['order_tag1'],
        )
        for _, row in data_df.iterrows()
    ]

    execute_values(cursor, query, data_tuples)
    conn.commit()
    conn.close(); cursor.close()
    return len(data_tuples)


# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--curr_time', default=None,
                    help='UTC "%%Y-%%m-%%d %%H:%%M:%%S"; defaults to now')
    ap.add_argument('--hist_replay', type=int, required=True, choices=[0, 1])
    ap.add_argument('--cores', type=int, required=True)
    ap.add_argument('--ffill_data', type=int, required=True, choices=[0, 1])
    ap.add_argument('--interval_minutes', type=float, default=5.0,
                    help='minutes between passes')
    ap.add_argument('--lookback_days', type=int, default=None,
                    help='trailing window recomputed each pass; 0 = all available '
                         '(default: config_object.lookback_days, else 300)')
    ap.add_argument('--pass_timeout_s', type=float, default=5200.0,
                    help='abandon a pass whose pool has not finished in this long; a hung '
                         'cell must not wedge the producer forever')
    ap.add_argument('--db_insert_row_count', type=int, default=100)
    ap.add_argument('--database_name', default=None, choices=["uat_tsdb", "tsdb"])
    ap.add_argument('--from_bundle', type=int, default=0, choices=[0, 1],
                    help='take cells from the frozen bundle instead of submodel_parameters')
    ap.add_argument('--exec_type', type=int, default=None, choices=list(EXEC_TYPE_DOMAIN),
                    help='required with --from_bundle (execution_type is NOT NULL)')
    ap.add_argument('--pairs', nargs='+', default=None, help='subset as COIN1_COIN2')
    ap.add_argument('--parent_strat_ids', type=int, nargs='+', default=None)
    ap.add_argument('--hist_replay_dir', default=None)
    ap.add_argument('--init_hist_replay_dir', default=None)
    ap.add_argument('--max_passes', type=int, default=0,
                    help='stop after N passes (0 = run forever); for smoke tests')
    ap.add_argument('--dry_run', type=int, default=0, choices=[0, 1],
                    help='compute and report, never touch the DB')
    a = ap.parse_args()

    assert a.cores <= max(1, (os.cpu_count() or 2) - 1), \
        f'--cores {a.cores} leaves nothing for the feeder thread'
    if a.hist_replay:
        assert a.hist_replay_dir and a.init_hist_replay_dir, \
            '--hist_replay 1 needs --hist_replay_dir and --init_hist_replay_dir'

    lookback_days = a.lookback_days
    if lookback_days is None:
        lookback_days = getattr(config_object, 'lookback_days', 300)
    print(f'recompute window: {lookback_days or "ALL"} days', flush=True)

    # ---- cells ---------------------------------------------------------------
    if a.from_bundle:
        assert a.exec_type is not None, '--from_bundle needs --exec_type'
        cells = load_cells_from_bundle(a.pairs, a.parent_strat_ids)
        cells = [(tup, a.exec_type) for tup, _ in cells]
    else:
        cells = load_cells_from_db(a.database_name, a.pairs, a.parent_strat_ids)
    pair_keys = sorted({(t.coin1, t.coin2) for t, _ in cells})
    print(f'{len(cells)} cells over {len(pair_keys)} pairs '
          f'(ids {min(int(t.parent_stratid) for t, _ in cells)}..'
          f'{max(int(t.parent_stratid) for t, _ in cells)})', flush=True)

    ## Bundle ids are per-pair LOCAL (every pair starts at 40000001), so a multi-pair
    ## bundle run would publish several pairs under the SAME parent_trading_model and
    ## silently overwrite via the upsert key. Only the DB path carries globally unique ids.
    if a.from_bundle and len(pair_keys) > 1 and not (a.hist_replay or a.dry_run):
        raise SystemExit(
            f'--from_bundle with {len(pair_keys)} pairs would collide: bundle parent ids '
            f'are per-pair local. Use --pairs for a single pair, add --dry_run, or drop '
            f'--from_bundle so ids come from submodel_parameters.')

    args_tup = ARGS_TUP(lookback_days=lookback_days,
                        db_insert_row_count=a.db_insert_row_count,
                        hist_replay=a.hist_replay,
                        database_name=a.database_name)

    # ---- data feeder ---------------------------------------------------------
    symbols = sorted({s for pair in pair_keys for s in pair})
    data_queue = Queue()
    replay_data = None
    if a.hist_replay:
        curr_time = (pd.Timestamp(a.curr_time, tz='UTC') if a.curr_time else None)
        replay_data = load_replay_data(a.init_hist_replay_dir, symbols, curr_time)
        print('replay data: ' + '  '.join(
            f'{s}={replay_data[s].index[-1]}' for s in symbols), flush=True)
    else:
        ## Identity mapping: DataClassBacktest keys SYMBOL_DATA_DICT by base_sym when it
        ## loads from the DB but by symbol in hist_replay, so identity keeps both
        ## consistent. (The reversal reference reads
        ## config_object.global_variables["SYMBOL_MAPPING"], prod-only.)
        from backtest_utils import DataClassBacktest
        feeder = DataClassBacktest({s: s for s in symbols}, "pm_backtest", "pm_backtest",
                                   data_queue, a.curr_time, hist_replay=0,
                                   ffill_data=a.ffill_data)
        t = threading.Thread(target=feeder.run)
        t.daemon = True
        t.start()

    # ---- loop ----------------------------------------------------------------
    passes = 0
    while True:
        try:
            if a.hist_replay:
                data = replay_data          # fixed snapshot; no feeder thread on this path
            else:
                ## LIVE: drain to the NEWEST snapshot the feeder has produced
                data = None
                while not data_queue.empty():
                    data = data_queue.get()
            if data is None:
                print('waiting for data', flush=True)
                time.sleep(min(60.0, a.interval_minutes * 60))
                continue

            t0 = time.time()
            ## One task per PAIR (all its TF cells), carrying only that pair's frames --
            ## see run_pair_wrapper for why cells travel as dicts and why this is not
            ## per-cell.
            tasks = [(args_tup, {c1: data[c1], c2: data[c2]},
                      [(t._asdict(), ex) for t, ex in cells if (t.coin1, t.coin2) == (c1, c2)])
                     for c1, c2 in pair_keys]
            ## starmap_ASYNC with a timeout: a bare blocking starmap lets one hung cell
            ## wedge the producer forever, with no symptom except rows quietly not arriving.
            with mp.get_context('spawn').Pool(a.cores) as pool:
                async_res = pool.starmap_async(run_pair_wrapper, tasks)
                try:
                    results = async_res.get(timeout=a.pass_timeout_s)
                except mp.TimeoutError:
                    pool.terminate(); pool.join()
                    msg_ = f'pass timed out after {a.pass_timeout_s:.0f}s -- pool terminated'
                    print(f'[CRITICAL] {msg_}', flush=True)
                    config_object.masterlog.get_logger('pm_backtest', 'critical')(msg_)
                    time.sleep(min(60.0, a.interval_minutes * 60))
                    continue

            ok = [frame for pair_out in results for frame in pair_out]
            if len(ok) != len(cells):
                msg_ = f'{len(cells) - len(ok)}/{len(cells)} cells FAILED this pass'
                print(f'[CRITICAL] {msg_}', flush=True)
                config_object.masterlog.get_logger('pm_backtest', 'critical')(msg_)

            passes += 1
            dump_df = pd.concat(ok, axis=0, ignore_index=True) if ok else pd.DataFrame()
            last_ts = dump_df['Timestamp'].max() if len(dump_df) else None
            print(f'pass {passes}: {len(ok)}/{len(cells)} cells, {len(dump_df):,} rows, '
                  f'through {last_ts} ({time.time() - t0:.0f}s)', flush=True)

            ## hist_replay never writes -- same as the reference.
            if a.hist_replay or a.dry_run:
                print('  (no DB write: '
                      f'{"hist_replay" if a.hist_replay else "dry_run"})', flush=True)
            elif len(dump_df):
                n = upsert_signals(dump_df, a.database_name)
                print(f'  upserted {n:,} rows into backtest_signals', flush=True)

            if a.max_passes and passes >= a.max_passes:
                print(f'reached --max_passes {a.max_passes}', flush=True)
                return 0

            time.sleep(a.interval_minutes * 60)

        except KeyboardInterrupt:
            print('interrupted', flush=True)
            return 0
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f'pass failed: {e}\n{tb_}'
            print(f'[CRITICAL] {msg_}', flush=True)
            config_object.masterlog.get_logger('pm_backtest', 'critical')(msg_)
            time.sleep(min(60.0, a.interval_minutes * 60))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
