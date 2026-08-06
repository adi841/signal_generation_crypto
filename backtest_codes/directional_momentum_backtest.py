"""Backtest signal producer for directional_momentum (DMP_v3_2).

The DMP twin of backtest_codes/pair_relative_value_backtest.py: a long-lived process that
re-derives the signal series with BACKTEST (vectorised, whole-window) methodology on a timer
and upserts the newest rows into `backtest_signals`, so the backtest and live streams can be
diffed per parent_trading_model.

As with the pair sleeves, this imports the sleeve's own batch path rather than
re-implementing it:

    GetDirectionalMomentumSignal.get_numba_parameters
        -> cryptoasset_dmpv32_long_iact | cryptoasset_dmpv32_short_iact

which is exactly what PRODUCTION's directional_momentum.run_cell drives, verified bitwise
against the frozen reference over full history on both traded assets (Test 1: 160/160 per
asset). No `_ll` kernel, no live incremental state: the whole point is that this is the batch
path.

SINGLE ASSET, TWO SIDES — the structural difference from every other driver here. DMP has no
ratio and no second leg, so there is no comb_df and no ratio price: `price` is the asset's own
trade price. Each (asset, TF) cell runs TWO INDEPENDENT models, a long Keltner breakout and
its mirrored short, and each side is its OWN PARAM_TUPLE with its own parent_stratid. So one
asset contributes 8 cells (4 TFs x {LONG, SHORT}) and `run_cell` returns ONE frame — the
two-streams-per-cell fan-out happens at the cell-list level, not inside run_cell. Bundle ids
are the 1xxxxxxx band, stride 2 (LONG 10000001/03/05/07, SHORT at +1). Measured on the
goldens, both sides are in position simultaneously 72.8% of minutes, so the two streams of a
cell routinely carry opposing signals on the same instrument; netting is the execution
layer's business.

PUBLISHED TIMESTAMPS LAG THE FEED BY 2 MINUTES. get_numba_parameters ends with
`mc = mc.iloc[:-2]` because the fill convention needs t+1 and t+2 (next_close is the mean
OHLC4 of the next two minutes). So `Timestamp.max()` of a pass is the feed's last bar minus 2.
No sibling sleeve has this; it is expected, not a data problem.

RECOMPUTE WINDOW. Each pass recomputes a trailing `--lookback_days` window (default
config_object.lookback_days) rather than all history. The sleeve's warm-up study MEASURED the
requirement at 45 days -- binding feature `long_vol` (EWM span 100) at TF=60, 9.4e-11 relative
at 45d and 2.8e-14 at 60d -- so the default is an order of magnitude past the threshold.
`--lookback_days 0` means "everything available", for parity runs.

  Run:  python backtest_codes/directional_momentum_backtest.py --hist_replay 0 \
            --cores 8 --ffill_data 0 --curr_time "2026-08-04 10:00:00" --interval_minutes 5

        python backtest_codes/directional_momentum_backtest.py --hist_replay 1 \
            --cores 4 --ffill_data 0 --curr_time "2026-04-01 00:00:00" --from_bundle 1 \
            --exec_type 1 \
            --hist_replay_dir ./backtest_codes/data_dm \
            --init_hist_replay_dir ./backtest_codes/data_dm
        (hist_replay never writes to the DB, exactly like the siblings)
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
from last_line_utils.directional_momentum_utils.directional_momentum_utils import (
    DIRECTIONAL_MOMENTUM_PARAM_TUPLE,
    DIRECTIONAL_MOMENTUM_PARAMS_PATH,
    EOD_OUTPUT_DIR,
    build_directional_momentum_parameter_dict,
    build_directional_momentum_parameter_dict_from_db,
)
from last_line_utils.directional_momentum_utils.generate_signal_directional_momentum import (
    GetDirectionalMomentumSignal)
from utils.directional_momentum_utils import (cryptoasset_dmpv32_long_iact,
                                              cryptoasset_dmpv32_short_iact)
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
STRATEGY_NAME = "directional_momentum"

## backtest_signals.execution_type is the execution_type_checks DOMAIN, which the live DB has
## widened to {1..7}. Same reasoning as pair_momentum_backtest.py — the PRV script still
## asserts {1,2,3,4} and would reject a legitimate 6/7.
EXEC_TYPE_DOMAIN = (1, 2, 3, 4, 5, 6, 7)

## Kernel per side. Each PARAM_TUPLE IS one side, so this is a per-cell lookup, not a
## per-bar branch — mirrors generate_signal_directional_momentum.py::generate_signal.
_KERNEL = {"LONG": cryptoasset_dmpv32_long_iact,
           "SHORT": cryptoasset_dmpv32_short_iact}

ARGS_TUP = namedtuple("ARGS_TUP", [
    "lookback_days", "db_insert_row_count", "hist_replay", "database_name"])


# ---------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------
def _sleeve_paths():
    """sleeve_config for this sleeve, with the two artifact paths defaulted.

    build_directional_momentum_parameter_dict_from_db hard-indexes `rv_alloc_bounds_path`
    and `r_state_path`, and both are actually OPENED downstream (load_rv_alloc_bounds at
    cell construction; r_multiplier_array for the SHORT side's allocation). The prod client
    config is expected to carry them; config/local_config.json carries only `strategy` and
    `params_path`, so a local run would KeyError before the first bar. Default the missing
    keys to the same values the bundle builder uses rather than editing shared config.
    """
    paths = dict(config_object.sleeve_config[STRATEGY_NAME])
    paths.setdefault("params_path", DIRECTIONAL_MOMENTUM_PARAMS_PATH)
    paths.setdefault("rv_alloc_bounds_path", f"{EOD_OUTPUT_DIR}/rv_alloc_bounds.parquet")
    paths.setdefault("r_state_path", f"{EOD_OUTPUT_DIR}/r_state_daily.parquet")
    return paths


def load_cells_from_db(database_name, assets=None, parent_strat_ids=None):
    """Live's source of truth: submodel_parameters (is_live = 1). Returns
    [(PARAM_TUPLE, exec_type)] — exec_type rides alongside because it lives in the raw
    model_parameters JSON and is NOT a field of DIRECTIONAL_MOMENTUM_PARAM_TUPLE.

    NOTE the grouping key is `coin` (SINGULAR). The pair sleeves group on
    (coin1, coin2); DMP rows carry one asset, and the builder asserts `mp["coin"] == coin1`.
    """
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

    by_asset = {}
    for row in rows:
        mp_ = row['model_parameters']
        by_asset.setdefault(mp_['coin'], {})[row['parent_trading_model']] = mp_

    sleeve_paths = _sleeve_paths()
    cells = []
    for coin, db_rows in sorted(by_asset.items()):
        if assets and coin not in assets:
            continue
        pdict = build_directional_momentum_parameter_dict_from_db(db_rows, coin, sleeve_paths)
        for pid, tup in sorted(pdict.items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, _exec_type_of(db_rows[pid], pid)))
    assert cells, "no cells selected — check --assets / --parent_strat_ids"
    return cells


def load_cells_from_bundle(assets=None, parent_strat_ids=None):
    """Offline / full-universe path: the frozen bundle, per-ASSET LOCAL ids. exec_type is
    absent from the bundle, so it must be supplied with --exec_type.

    The bundle universe is a FLAT list of assets (`universe.assets`), not the pair sleeves'
    `universe.pairs_leg1_leg2`.
    """
    with open(DIRECTIONAL_MOMENTUM_PARAMS_PATH) as f:
        universe = list(json.load(f)["universe"]["assets"])
    cells = []
    for coin in universe:
        if assets and coin not in assets:
            continue
        for pid, tup in sorted(build_directional_momentum_parameter_dict(coin).items()):
            if parent_strat_ids and pid not in parent_strat_ids:
                continue
            cells.append((tup, None))
    assert cells, "no cells selected — check --assets / --parent_strat_ids"
    return cells


def _exec_type_of(model_parameters, pid):
    """backtest_signals.execution_type is NOT NULL and domain-checked. Fail loudly rather
    than inventing a default."""
    v = model_parameters.get('exec_type')
    assert v is not None, f"{pid}: model_parameters has no exec_type (execution_type is NOT NULL)"
    v = int(v)
    assert v in EXEC_TYPE_DOMAIN, f"{pid}: exec_type {v} outside the execution_type_checks domain"
    return v


# ---------------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------------
def load_replay_data(init_dir, symbols, curr_time=None):
    """hist_replay data straight from the `{SYM}_input.parquet` files.

    DataClassBacktest is deliberately NOT used on this path: its __init__ calls
    Base.prepare_socket(), which dereferences prod-only config (config_object.zmq_data) and
    so cannot run in local/test mode; and its replay loop only publishes a snapshot every
    4th day at 15:30, which buys nothing for a whole-window recompute. The live path still
    uses it, where that machinery is real."""
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


def _price_and_tags(res, tradeprice, ts, parent_id):
    """price + order_tag on THIS sleeve's batch convention.

    Transcribed from generate_signal_directional_momentum.py::generate_signal — both are
    written only on bars where the signal CHANGES, then forward-filled:

        tmp_df['signal_diff'] = tmp_df['signal'].diff().fillna(0)
        tmp_df_signal_diff    = tmp_df[tmp_df['signal_diff'] != 0]
        ... stamp price/order_tag there, then .ffill()

    Deliberately NOT pair_relative_value's every-bar price: "the methodology used to
    backtest" is this sleeve's own batch definition, and DMP's is pair_momentum-shaped.

    SINGLE PRICE — the kernels return one `tradeprice_arr`, not a per-leg pair, so there is
    no ratio to form here.

    Note `signal.diff().fillna(0)`, so the window's FIRST bar never counts as a change. A
    trailing window that opens mid-position therefore ffills from nothing until the first
    real change; the caller turns those into the "" sentinel, because order_tag is NOT NULL.
    Both sides behave identically under this test: res is +1/0 on LONG and -1/0 on SHORT, so
    a diff is non-zero on exactly that side's entry and exit minutes.
    """
    sig = pd.Series(np.asarray(res, dtype=np.float64), index=ts)
    changed = sig.diff().fillna(0) != 0

    price = pd.Series(np.nan, index=ts, dtype=float)
    tags = pd.Series(np.nan, index=ts, dtype=object)
    if changed.any():
        m = changed.to_numpy()
        price.loc[changed] = np.asarray(tradeprice, dtype=np.float64)[m]
        tags.loc[changed] = [
            order_tag_generator(ts=t, parent_id=int(parent_id), signal=int(s))
            for t, s in zip(ts[m], sig[changed].to_numpy())
        ]
    return price.ffill(), tags.ffill()


def run_cell(args_tup, data, tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE, exec_type):
    """Derive ONE (asset, TF, side) stream over the trailing window and return it in
    `upsert` column shape.

    One frame, not two: each side is already its own cell with its own parent_stratid.
    """
    raw = data[tup.coin1]
    end = raw.index[-1]
    start = raw.index[0]
    ## LIVE: the feed is ALREADY this deep — DataClassBacktest.init_data ->
    ## Base.get_init_data slices `end - config_object.lookback_days`, and this script
    ## defaults to the same key, so the clamp is a no-op there.
    ## HIST_REPLAY: the parquet carries full history, so this is what bounds the window.
    if args_tup.lookback_days:
        start = max(start, end - dt.timedelta(days=args_tup.lookback_days))

    df1 = raw.loc[start:end, OHLCV_COLS]
    ## The generator takes the RAW minute frame plus its filled twin — exactly what
    ## DirectionalMomentumSignalGenerator.prepare_init_data builds. The reference derives
    ## both its candle inputs (vendored_b1_dmp.prep:106) and its minute fields (:126) from
    ## the FILLED frame, so one filled copy reproduces both; df1 stays raw because the live
    ## path needs the NaNs to know which minutes were genuinely missing.
    warm_df = df1.ffill().bfill()

    sig = GetDirectionalMomentumSignal(tup=None, df1=df1, warm_df=warm_df,
                                       curr_time=None, process_name="dm_backtest")
    out, aux = sig.get_numba_parameters(tup, None)
    (next_close, same_close, p1, minutely_extreme, asset_close, median_line, band_line,
     atr_eq, long_vol, z_median, txn_cost, slippage, alloc) = out

    ## Slots 3 and 6 of `out` are ALREADY side-resolved by get_numba_parameters
    ## (minutely_low|minutely_high, upper_line|lower_line), so the kernel call is the same
    ## shape for both sides. txn_cost / slippage are SCALARS here, not series.
    array_output, _ = _KERNEL[tup.side](
        next_close, same_close, p1, minutely_extreme, asset_close, median_line,
        band_line, atr_eq, long_vol, z_median, txn_cost, slippage, alloc,
        float(tup.T1), float(tup.TT), float(tup.dde_mult),
        float(tup.K), float(tup.X), float(tup.Z), float(tup.EZ))

    ts = pd.to_datetime(aux[0])
    pid = int(tup.parent_stratid)
    res = np.asarray(array_output.res_arr, dtype=np.float64)
    price, tags = _price_and_tags(res, array_output.tradeprice_arr, ts, pid)

    df = pd.DataFrame({
        'Timestamp': ts,
        'parent_trading_model': pid,
        'signal': res.astype(np.int64),
        'tradeprice1': price.to_numpy(),
        'case': 0,                       # DMP has no case output; the column is NOT NULL
        'execution_type': exec_type,
        'order_tag1': tags.to_numpy(),
    })
    df = df.iloc[-args_tup.db_insert_row_count:].copy()
    df['signal_id'] = df['Timestamp'].map(lambda x: int(x.timestamp()))

    ## order_tag is NOT NULL. Rows before the window's first signal change have nothing to
    ## ffill from. Publish them under the EMPTY STRING — the same "no tag yet" sentinel this
    ## sleeve's own batch path uses (generate_signal_directional_momentum.py,
    ## `order_tag1 = ""`). Dropping the rows instead would make the two sides of a cell
    ## disagree on row count for exactly the cells whose signal is most static, which is the
    ## wrong thing to hide from a diff. Still logged — a whole trailing window with no signal
    ## change is worth knowing about, and on this sleeve it is rare: exposure is ~85-95%.
    null_tag = df['order_tag1'].isna()
    if null_tag.any():
        msg_ = (f'{pid} {tup.coin1} tf={tup.tf} {tup.side}: {int(null_tag.sum())} of '
                f'{len(df)} published rows have no order_tag (no signal change in the '
                f'{args_tup.lookback_days or "full"}-day window before them) — published '
                f'with the "" sentinel')
        print(f'[warn] {msg_}', flush=True)
        config_object.masterlog.get_logger("dm_backtest", "error")(msg_)

    del sig, out, aux, array_output, df1, warm_df
    return df.reset_index(drop=True)


def run_asset_wrapper(args_tup, data, cell_payloads):
    """All 8 cells of ONE asset (4 TFs x 2 sides), in one worker task.

    Two reasons this is per-ASSET rather than per-cell:
      * DIRECTIONAL_MOMENTUM_PARAM_TUPLE cannot cross a `spawn` boundary — the namedtuple's
        typename ("directional_momentum_param_tuple") differs from its module attribute
        name, so pickle's lookup fails. Cells therefore travel as plain dicts and are
        rebuilt here.
      * `data` is pickled once per task; sending only this asset's frame, once for all 8
        cells, avoids copying the whole universe 8 times.
    Returns a list of frames; a cell that raises contributes nothing but never takes the
    pass down with it."""
    out = []
    for cell_dict, exec_type in cell_payloads:
        tup = DIRECTIONAL_MOMENTUM_PARAM_TUPLE(**cell_dict)
        try:
            out.append(run_cell(args_tup, data, tup, exec_type))
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"{tup.parent_stratid} {tup.coin1} tf={tup.tf} {tup.side}: {e}\n{tb_}"
            print(f"[ERROR] {msg_}", flush=True)
            config_object.masterlog.get_logger("dm_backtest", "error")(msg_)
    return out


# ---------------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------------
def upsert_signals(data_df: pd.DataFrame, database_name=None):
    """Same statement/contract as backtest_utils.upsert_data, but honouring `database_name`
    on the WRITE as well — the shared helper accepts it only on the param read, so
    `--database_name uat_tsdb` there still writes to the default DB. Kept local rather than
    changing the shared helper the other drivers also use."""
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
            ## NaN would be written by psycopg2 as the literal string 'NaN', which reads
            ## like a real tag. Use the "" sentinel (see run_cell).
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
    ap.add_argument('--interval_minutes', type=float, default=5.0, help='minutes between passes')
    ap.add_argument('--lookback_days', type=int, default=None, help='trailing window recomputed each pass; 0 = all available (default: config_object.lookback_days). MEASURED requirement is 45 days.')
    ap.add_argument('--db_insert_row_count', type=int, default=100)
    ap.add_argument('--database_name', default=None, choices=["uat_tsdb", "tsdb"])
    ap.add_argument('--from_bundle', type=int, default=0, choices=[0, 1], help='take cells from the frozen bundle instead of submodel_parameters')
    ap.add_argument('--exec_type', type=int, default=None, choices=list(EXEC_TYPE_DOMAIN), help='required with --from_bundle (execution_type is NOT NULL)')
    ap.add_argument('--assets', nargs='+', default=None, help='subset as exchange symbols, e.g. BTCUSDT ADAUSDT')
    ap.add_argument('--parent_strat_ids', type=int, nargs='+', default=None)
    ap.add_argument('--hist_replay_dir', default=None)
    ap.add_argument('--init_hist_replay_dir', default=None)
    ap.add_argument('--max_passes', type=int, default=0, help='stop after N passes (0 = run forever); for smoke tests')
    ap.add_argument('--dry_run', type=int, default=0, choices=[0, 1], help='compute and report, never touch the DB')
    ap.add_argument('--pass_timeout_s', type=float, default=5200.0, help='terminate the pool if a pass exceeds this')
    a = ap.parse_args()

    assert a.cores <= max(1, (os.cpu_count() or 2) - 1), f'--cores {a.cores} leaves nothing for the feeder thread'
    if a.hist_replay:
        assert a.hist_replay_dir and a.init_hist_replay_dir, '--hist_replay 1 needs --hist_replay_dir and --init_hist_replay_dir'

    lookback_days = a.lookback_days
    if lookback_days is None:
        lookback_days = getattr(config_object, 'lookback_days', 300)
    print(f'recompute window: {lookback_days or "ALL"} days '
          f'(measured requirement 45d; binding feature long_vol/EWM-100 at TF=60)', flush=True)

    # ---- cells ---------------------------------------------------------------
    if a.from_bundle:
        assert a.exec_type is not None, '--from_bundle needs --exec_type'
        cells = load_cells_from_bundle(a.assets, a.parent_strat_ids)
        cells = [(tup, a.exec_type) for tup, _ in cells]
    else:
        cells = load_cells_from_db(a.database_name, a.assets, a.parent_strat_ids)
    asset_keys = sorted({t.coin1 for t, _ in cells})
    n_long = sum(1 for t, _ in cells if t.side == 'LONG')
    print(f'{len(cells)} cells over {len(asset_keys)} assets '
          f'({n_long} LONG / {len(cells) - n_long} SHORT, ids '
          f'{min(int(t.parent_stratid) for t, _ in cells)}..'
          f'{max(int(t.parent_stratid) for t, _ in cells)})', flush=True)

    ## Bundle ids are per-ASSET LOCAL (every asset starts at 10000001), so a multi-asset
    ## bundle run would publish several assets under the SAME parent_trading_model and
    ## silently overwrite via the upsert key (parent_trading_model, signal_floor_time).
    ## Only the DB path carries globally unique ids. Multi-asset bundle runs are fine as
    ## long as nothing is written.
    if a.from_bundle and len(asset_keys) > 1 and not (a.hist_replay or a.dry_run):
        raise SystemExit(
            f'--from_bundle with {len(asset_keys)} assets would collide: bundle parent ids '
            f'are per-asset local (10000001..10000008 for every asset). Use --assets for a '
            f'single asset, add --dry_run, or drop --from_bundle so ids come from '
            f'submodel_parameters.')

    args_tup = ARGS_TUP(lookback_days=lookback_days,
                        db_insert_row_count=a.db_insert_row_count,
                        hist_replay=a.hist_replay,
                        database_name=a.database_name)

    # ---- data feeder ---------------------------------------------------------
    ## Identity mapping: DataClassBacktest keys SYMBOL_DATA_DICT by base_sym when it loads
    ## from the DB but by symbol in hist_replay, so identity keeps both paths consistent.
    symbols = list(asset_keys)
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
        feeder = DataClassBacktest({s: s for s in symbols}, "dm_backtest", "dm_backtest", data_queue, a.curr_time, hist_replay=0, ffill_data=a.ffill_data)
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
            ## One task per ASSET (all 8 of its cells), carrying only that asset's frame.
            tasks = [(args_tup, {coin: data[coin]},
                      [(t._asdict(), ex) for t, ex in cells if t.coin1 == coin])
                     for coin in asset_keys]
            ## starmap_ASYNC with a timeout: a bare blocking starmap lets one hung cell wedge
            ## the producer forever, with no symptom except rows quietly not arriving.
            with mp.get_context('spawn').Pool(a.cores) as pool:
                async_res = pool.starmap_async(run_asset_wrapper, tasks)
                try:
                    results = async_res.get(timeout=a.pass_timeout_s)
                except mp.TimeoutError:
                    pool.terminate(); pool.join()
                    msg_ = f'pass timed out after {a.pass_timeout_s:.0f}s -- pool terminated'
                    print(f'[CRITICAL] {msg_}', flush=True)
                    config_object.masterlog.get_logger('dm_backtest', 'critical')(msg_)
                    time.sleep(min(60.0, a.interval_minutes * 60))
                    continue

            ok = [frame for asset_out in results for frame in asset_out]
            if len(ok) != len(cells):
                msg_ = f'{len(cells) - len(ok)}/{len(cells)} cells FAILED this pass'
                print(f'[CRITICAL] {msg_}', flush=True)
                config_object.masterlog.get_logger('dm_backtest', 'critical')(msg_)

            passes += 1
            dump_df = pd.concat(ok, axis=0, ignore_index=True) if ok else pd.DataFrame()
            last_ts = dump_df['Timestamp'].max() if len(dump_df) else None
            ## `last_ts` trails the feed's last bar by 2 minutes — get_numba_parameters
            ## drops them because next_close needs t+1 and t+2. Expected.
            print(f'pass {passes}: {len(ok)}/{len(cells)} cells, {len(dump_df):,} rows, '
                  f'through {last_ts} ({time.time() - t0:.0f}s)', flush=True)

            ## hist_replay never writes — same as the siblings (its output is the per-pass
            ## frames, used by the parity harnesses).
            if a.hist_replay or a.dry_run:
                print('  (no DB write: '
                      f'{"hist_replay" if a.hist_replay else "dry_run"})', flush=True)
            elif len(dump_df):
                n = upsert_signals(dump_df, a.database_name)
                print(f'  upserted {n:,} rows into backtest_signals'
                      f'{" (" + a.database_name + ")" if a.database_name else ""}', flush=True)

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
            config_object.masterlog.get_logger('dm_backtest', 'critical')(msg_)
            time.sleep(min(60.0, a.interval_minutes * 60))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
