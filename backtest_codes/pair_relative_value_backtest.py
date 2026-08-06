"""Backtest signal producer for pair_relative_value (QR1_v4).

The PRV twin of backtest_codes/pair_reversal_vol_target_backtest.py: a long-lived
process that re-derives the signal series with BACKTEST (vectorised, whole-window)
methodology on a timer and upserts the newest rows into `backtest_signals`, so the
backtest and live streams can be diffed per parent_trading_model.

KEY DELTA vs the reversal reference: that script open-codes its strategy math because
its sleeve has no other implementation. PRV does — and it is the SAME vectorised
methodology, verified bitwise against the frozen PRODUCTION reference over full
history (see the sleeve's matching harnesses). So this script imports it rather than
re-implementing it:

    GetPairsRelativeValueSignal.get_numba_parameters  ->  cryptopairs_qr1v4_short_iact

which is exactly what PRODUCTION's pairs_relvalue.run_cell drives. No `_ll` kernel, no
live incremental state: the whole point is that this is the batch path.

DUAL TRANCHE. The kernel emits two signal series and both are published, mirroring
pair_relative_value/live.py: tranche 1 (`res`) at parent_id, the scale-in tranche
(`res2`) at parent_id + 2. Parent ids are the 3xxxxxxx band, stride 4 per TF
(30000001/05/09/13/17, tranche-2 streams at 30000003/07/11/15/19).

RECOMPUTE WINDOW. Each pass recomputes a trailing `--lookback-days` window (default
config_object.lookback_days) rather than all history: the sleeve's warm-up study
measured the slowest state (Wilder ATR50 @ TF240) converged to 2.5e-11 by 214 days, so
300 is past the threshold with margin, and a pass costs ~1-2 min instead of ~15.
`--lookback-days 0` means "everything available", for parity runs.

  Run:  python backtest_codes/pair_relative_value_backtest.py --hist_replay 0 \
            --cores 8 --ffill_data 0 --curr_time "2026-08-03 10:00:00" --interval_minutes 5

        python backtest_codes/pair_relative_value_backtest.py --hist_replay 1 \
            --cores 4 --ffill_data 0 --curr_time "2022-01-01 00:00:00" --from_bundle 1 \
            --hist_replay_dir ./backtest_codes/data_prv \
            --init_hist_replay_dir ./backtest_codes/data_prv
        (hist_replay never writes to the DB, exactly like the reference)
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
## backtest_utils is imported lazily (live path only) — it pulls in the prod config
## chain, and hist_replay runs have no need of it.
from last_line_utils.pair_relative_value_utils.pair_relative_value_utils import (
    PAIR_RELATIVE_VALUE_PARAM_TUPLE,
    PAIR_RELATIVE_VALUE_PARAMS_PATH,
    build_pair_relative_value_parameter_dict,
    build_pair_relative_value_parameter_dict_from_db,
)
from last_line_utils.pair_relative_value_utils.generate_signal_pair_relative_value import (
    GetPairsRelativeValueSignal)
from utils.pair_relative_value_utils import cryptopairs_qr1v4_short_iact
from shared_codes.utils.misc_utils import order_tag_generator

## NOTE: `dump_slack_msg` is deliberately NOT imported — it was removed from
## backtest_utils (it carried a hardcoded bot token) and the reversal script's stale
## import of it is why that script currently dies at import time.

import warnings
warnings.filterwarnings("ignore")

OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
STRATEGY_NAME = "pair_relative_value"
TRANCHE2_ID_OFFSET = 2          # live.py posts the scale-in stream at parent_id + 2
PROFIT_TARGET_CONST = 1000.0    # vestigial kernel input, frozen at 1000.0

ARGS_TUP = namedtuple("ARGS_TUP", [
    "lookback_days", "db_insert_row_count", "hist_replay", "database_name"])


# ---------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------
def load_cells_from_db(database_name, pairs=None, parent_strat_ids=None):
    """Live's source of truth: submodel_parameters (is_live = 1). Returns
    [(PARAM_TUPLE, exec_type)] — exec_type also rides alongside so the bundle path
    (load_cells_from_bundle, where the frozen artifacts carry no exec_type) can supply
    it explicitly. The DB path now also populates PARAM_TUPLE.exec_type, which is what
    live.py reads; the two agree by construction here."""
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

    sleeve_paths = config_object.sleeve_config[STRATEGY_NAME]
    cells = []
    for (c1, c2), db_rows in sorted(by_pair.items()):
        if pairs and f"{c1}_{c2}" not in pairs:
            continue
        pdict = build_pair_relative_value_parameter_dict_from_db(db_rows, c1, c2, sleeve_paths)
        for pid, tup in sorted(pdict.items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, _exec_type_of(db_rows[pid], pid)))
    assert cells, "no cells selected — check --pairs / --parent_strat_ids"
    return cells


def load_cells_from_bundle(pairs=None, parent_strat_ids=None):
    """Offline / full-universe path: the frozen bundle, per-pair LOCAL ids. exec_type
    is absent from the bundle, so it must be supplied with --exec_type."""
    with open(PAIR_RELATIVE_VALUE_PARAMS_PATH) as f:
        universe = [tuple(p) for p in json.load(f)["universe"]["pairs_leg1_leg2"]]
    cells = []
    for c1, c2 in universe:
        if pairs and f"{c1}_{c2}" not in pairs:
            continue
        for pid, tup in sorted(build_pair_relative_value_parameter_dict(c1, c2).items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, None))
    assert cells, "no cells selected — check --pairs / --parent_strat_ids"
    return cells


def _exec_type_of(model_parameters, pid):
    """backtest_signals.execution_type is NOT NULL and domain-checked to {1,2,3,4}.
    Fail loudly rather than inventing a default."""
    v = model_parameters.get('exec_type')
    assert v is not None, f"{pid}: model_parameters has no exec_type (execution_type is NOT NULL)"
    v = int(v)
    assert v in (1, 2, 3, 4), f"{pid}: exec_type {v} outside the execution_type_checks domain"
    return v


# ---------------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------------
def load_replay_data(init_dir, symbols, curr_time=None):
    """hist_replay data straight from the `{SYM}_input.parquet` files.

    DataClassBacktest is deliberately NOT used on this path: its __init__ calls
    Base.prepare_socket(), which dereferences prod-only config (config_object.zmq_data)
    and so cannot run in local/test mode; and its replay loop only publishes a snapshot
    every 4th day at 15:30, which buys nothing for a whole-window recompute. The live
    path still uses it, where that machinery is real."""
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
    """The pair-ratio frame, identical to PairRelativeValueSignalGenerator.prepare_init_data."""
    comb = df1.merge(df2, left_index=True, right_index=True, suffixes=('_1', '_2'), how='outer')
    comb['open'] = comb['open_1'] / comb['open_2']
    comb['close'] = comb['close_1'] / comb['close_2']
    comb['volume'] = comb['volume_1'] + comb['volume_2']
    comb['high'] = np.max(comb[["open", "close"]], axis=1)
    comb['low'] = np.min(comb[["open", "close"]], axis=1)
    return comb.fillna(method='ffill').fillna(method='bfill')


def _tag_series(res, ts, parent_id):
    """Order tag per bar: a fresh tag stamped at every signal CHANGE, then ffilled —
    the convention live and the batch warm-up both use. Stamped with the id the row is
    published under (the reversal reference stamps its tranche-2 tags with the
    tranche-1 id; that is a bug, not a convention to copy)."""
    chg = np.flatnonzero(np.diff(res, prepend=0.0) != 0)
    tags = pd.Series(np.nan, index=ts, dtype=object)
    if len(chg):
        tags.iloc[chg] = [order_tag_generator(ts=ts[i], parent_id=int(parent_id),
                                              signal=int(res[i])) for i in chg]
    return tags.ffill()


def _ratio_price(num, den):
    """Fill-price ratio for the DB `price` column (live publishes the pair ratio too)."""
    den = np.asarray(den, dtype=np.float64)
    return np.where(den > 0, np.asarray(num, dtype=np.float64) / den, np.nan)


def run_cell(args_tup, data, tup: PAIR_RELATIVE_VALUE_PARAM_TUPLE, exec_type):
    """Derive one (pair, TF) cell over the trailing window and return the two
    tranche frames in `upsert` column shape."""
    raw1 = data[tup.coin1]
    raw2 = data[tup.coin2]
    end = min(raw1.index[-1], raw2.index[-1])
    start = max(raw1.index[0], raw2.index[0])
    ## LIVE: the feed is ALREADY this deep — DataClassBacktest.init_data ->
    ## Base.get_init_data slices `end - config_object.lookback_days`, and this script
    ## defaults to the same key, so the clamp is a no-op there.
    ## HIST_REPLAY: the parquet carries full history, so this is what bounds the window.
    ## Either way the number is config_object.lookback_days unless --lookback_days
    ## overrides it (0 = everything available, used by the parity harness).
    if args_tup.lookback_days:
        start = max(start, end - dt.timedelta(days=args_tup.lookback_days))

    df1 = raw1.loc[start:end, OHLCV_COLS]
    df2 = raw2.loc[start:end, OHLCV_COLS]
    comb = _comb_frame(df1, df2)

    sig = GetPairsRelativeValueSignal(tup=None, df1=df1, df2=df2, comb_df=comb,
                                      curr_time=None, process_name="prv_backtest")
    out, aux = sig.get_numba_parameters(tup, None)
    (nc1, sc1, p1a, mh, ml, nc2, sc2, msig, upper, middle, lower,
     atrA, atrB, alloc, znA, corrA, skokA, tc_s, sl_s) = out
    n = len(nc1)

    array_output, _ = cryptopairs_qr1v4_short_iact(
        nc1, sc1, np.full(n, PROFIT_TARGET_CONST), mh, ml, p1a, nc2, sc2,
        tc_s, msig, 1,
        upper.copy(), middle.copy(), lower.copy(), atrA, atrB,
        float(tup.strategy_config['SEC_MULT']), sl_s, alloc, znA,
        float(tup.z_thr), float(tup.x_atr), corrA,
        float(tup.strategy_config['C_THR']), float(tup.z_entry), skokA)

    ts = pd.to_datetime(aux[0])
    pid = int(tup.parent_stratid)
    res = np.asarray(array_output.res_arr, dtype=np.float64)
    res2 = np.asarray(array_output.res2_arr, dtype=np.float64)

    frames = []
    for stream_pid, sig_arr, num, den in (
            (pid, res, array_output.tradeprice1_arr, array_output.tradeprice2_arr),
            (pid + TRANCHE2_ID_OFFSET, res2, array_output.tradeprice3_arr,
             array_output.tradeprice4_arr)):
        df = pd.DataFrame({
            'Timestamp': ts,
            'parent_trading_model': stream_pid,
            'signal': sig_arr.astype(np.int64),
            'tradeprice1': _ratio_price(num, den),
            'case': 0,                       # QR1 has no case output; column is NOT NULL
            'execution_type': exec_type,
            'order_tag1': _tag_series(sig_arr, ts, stream_pid).to_numpy(),
        })
        df = df.iloc[-args_tup.db_insert_row_count:].copy()
        df['signal_id'] = df['Timestamp'].map(lambda x: int(x.timestamp()))
        frames.append(df.reset_index(drop=True))

    del sig, out, aux, array_output, comb, df1, df2
    return frames[0], frames[1]


def run_pair_wrapper(args_tup, data, cell_payloads):
    """All TF cells of ONE pair, in one worker task.

    Two reasons this is per-PAIR rather than per-cell:
      * PAIR_RELATIVE_VALUE_PARAM_TUPLE cannot cross a `spawn` boundary — the
        namedtuple's typename ("pair_relative_value_param_tuple") differs from its
        module attribute name, so pickle's lookup fails. Cells therefore travel as
        plain dicts and are rebuilt here.
      * `data` is pickled once per task; sending only this pair's two frames, once for
        all 5 TFs, avoids copying the whole universe 75 times.
    Returns a list of (df_1, df_2); a cell that raises contributes nothing but never
    takes the pass down with it."""
    out = []
    for cell_dict, exec_type in cell_payloads:
        tup = PAIR_RELATIVE_VALUE_PARAM_TUPLE(**cell_dict)
        try:
            out.append(run_cell(args_tup, data, tup, exec_type))
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"{tup.parent_stratid} {tup.coin1}/{tup.coin2} tf={tup.tf}: {e}\n{tb_}"
            print(f"[ERROR] {msg_}", flush=True)
            config_object.masterlog.get_logger("prv_backtest", "error")(msg_)
    return out


# ---------------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------------
def upsert_signals(data_df: pd.DataFrame, database_name=None):
    """Same statement/contract as backtest_utils.upsert_data, but honouring
    `database_name` on the WRITE as well — the shared helper accepts it only on the
    param read, so `--database_name uat_tsdb` there still writes to the default DB.
    Kept local rather than changing the shared helper the reversal script also uses."""
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
            ## A cell with no signal CHANGE in the window has no tag to carry (nothing
            ## to stamp, nothing to ffill from). backtest_signals.order_tag is NOT NULL,
            ## so it cannot be NULL; and letting the float nan through makes psycopg2
            ## write the literal string 'NaN', which reads like a real tag. Use the
            ## EMPTY STRING — the same "no tag yet" sentinel the sleeve's own live path
            ## uses (generate_signal_pair_relative_value.py::_tag_stream, `order_tag = ""`).
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--curr_time', default=None, help='UTC "%%Y-%%m-%%d %%H:%%M:%%S"; defaults to now')
    ap.add_argument('--hist_replay', type=int, required=True, choices=[0, 1])
    ap.add_argument('--cores', type=int, required=True)
    ap.add_argument('--ffill_data', type=int, required=True, choices=[0, 1])
    ap.add_argument('--interval_minutes', type=float, default=5.0, help='minutes between passes (the sleeve runs on a 5-8 min cadence)')
    ap.add_argument('--lookback_days', type=int, default=None, help='trailing window recomputed each pass; 0 = all available (default: config_object.lookback_days)')
    ap.add_argument('--db_insert_row_count', type=int, default=100)
    ap.add_argument('--database_name', default=None, choices=["uat_tsdb", "tsdb"])
    ap.add_argument('--from_bundle', type=int, default=0, choices=[0, 1], help='take cells from the frozen bundle instead of submodel_parameters')
    ap.add_argument('--exec_type', type=int, default=None, choices=[1, 2, 3, 4], help='required with --from_bundle (execution_type is NOT NULL)')
    ap.add_argument('--pairs', nargs='+', default=None, help='subset as COIN1_COIN2')
    ap.add_argument('--parent_strat_ids', type=int, nargs='+', default=None)
    ap.add_argument('--hist_replay_dir', default=None)
    ap.add_argument('--init_hist_replay_dir', default=None)
    ap.add_argument('--max_passes', type=int, default=0, help='stop after N passes (0 = run forever); for smoke tests')
    ap.add_argument('--dry_run', type=int, default=0, choices=[0, 1], help='compute and report, never touch the DB')
    a = ap.parse_args()

    assert a.cores <= max(1, (os.cpu_count() or 2) - 1), f'--cores {a.cores} leaves nothing for the feeder thread'
    if a.hist_replay:
        assert a.hist_replay_dir and a.init_hist_replay_dir, '--hist_replay 1 needs --hist_replay_dir and --init_hist_replay_dir'

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

    ## Bundle ids are per-pair LOCAL (every pair starts at 30000001), so a multi-pair
    ## bundle run would publish several pairs under the SAME parent_trading_model and
    ## silently overwrite via the upsert key. Only the DB path carries globally unique
    ## ids. Multi-pair bundle runs are fine as long as nothing is written.
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
    ## Identity mapping: DataClassBacktest keys SYMBOL_DATA_DICT by base_sym when it
    ## loads from the DB but by symbol in hist_replay, so identity keeps both paths
    ## consistent. (The reference reads config_object.global_variables["SYMBOL_MAPPING"],
    ## which only exists in prod mode and makes local runs impossible.)
    symbols = sorted({s for pair in pair_keys for s in pair})
    data_queue = Queue()
    replay_data = None
    if a.hist_replay:
        ## Whole-window recompute off the parquet feed — no feeder thread (see
        ## load_replay_data for why DataClassBacktest is not used here).
        curr_time = (pd.Timestamp(a.curr_time, tz='UTC') if a.curr_time else None)
        replay_data = load_replay_data(a.init_hist_replay_dir, symbols, curr_time)
        print('replay data: ' + '  '.join(
            f'{s}={replay_data[s].index[-1]}' for s in symbols), flush=True)
    else:
        from backtest_utils import DataClassBacktest
        feeder = DataClassBacktest({s: s for s in symbols}, "prv_backtest", "prv_backtest", data_queue, a.curr_time, hist_replay=0, ffill_data=a.ffill_data)
        t = threading.Thread(target=feeder.run)
        t.daemon = True
        t.start()

    # ---- loop ----------------------------------------------------------------
    passes = 0
    while True:
        try:
            if a.hist_replay:
                ## fixed parquet snapshot; there is no feeder thread on this path
                data = replay_data
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
            ## One task per PAIR (all its TF cells), carrying only that pair's frames.
            tasks = [(args_tup, {c1: data[c1], c2: data[c2]},
                      [(t._asdict(), ex) for t, ex in cells if (t.coin1, t.coin2) == (c1, c2)])
                     for c1, c2 in pair_keys]
            with mp.get_context('spawn').Pool(a.cores) as pool:
                results = pool.starmap(run_pair_wrapper, tasks)
            ok = [frames for pair_out in results for frames in pair_out]
            if len(ok) != len(cells):
                msg_ = f'{len(cells) - len(ok)}/{len(cells)} cells FAILED this pass'
                print(f'[CRITICAL] {msg_}', flush=True)
                config_object.masterlog.get_logger('prv_backtest', 'critical')(msg_)

            passes += 1
            dump_df = pd.concat([d for pair in ok for d in pair], axis=0, ignore_index=True)
            last_ts = dump_df['Timestamp'].max() if len(dump_df) else None
            print(f'pass {passes}: {len(ok)}/{len(cells)} cells, {len(dump_df):,} rows, '
                  f'through {last_ts} ({time.time() - t0:.0f}s)', flush=True)

            ## hist_replay never writes — same as the reference (its output is the
            ## per-pass frames, used by the parity harnesses).
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
            config_object.masterlog.get_logger('prv_backtest', 'critical')(msg_)
            time.sleep(min(60.0, a.interval_minutes * 60))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
