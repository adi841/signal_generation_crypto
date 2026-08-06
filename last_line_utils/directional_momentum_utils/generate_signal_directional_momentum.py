import pandas as pd
import numpy as np
import datetime as dt
from functools import lru_cache
import copy
import json
import math
import sys
from pytz import timezone

from config.config_read import config_object

from last_line_utils.directional_momentum_utils.directional_momentum_utils import (
    DIRECTIONAL_MOMENTUM_PARAM_TUPLE, SLEEVE, plain_symbol)
from last_line_utils.directional_momentum_utils.r_state import r_multiplier_array
from utils.directional_momentum_utils import (cryptoasset_dmpv32_long_iact,
                                              cryptoasset_dmpv32_short_iact,
                                              TradeOutputDirectionalMomentumLastOutputCls)
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

#
import os


## ---------------------------------------------------------------------------
## DMP_v3_2 feature calculations — pandas port of the reference pipeline
## (crypto_sims/PRODUCTION/engines/directional_momentum.py::run_cell). Deliberately
## pandas (not numpy/bottleneck): this is once-per-init batch code, and the reference
## leans on `ewm(span).std()` adjust=True debiased semantics everywhere — identical ops
## give bitwise parity with the golden book. Note the deliberate asymmetry: atr_eq /
## long_vol / rv use adjust=True (pandas default), z_median uses adjust=False. Every
## span/window/tunable comes from the frozen snapshot via the tuple / strategy_config —
## nothing hardcoded.
##
## PORT `run_cell`, NOT `prep`. vendored_b1_dmp.prep() builds median20/atr_eq/z_median/
## rv_alloc at its module-level SPAN_BAND = 14, and run_cell then THROWS THOSE AWAY and
## rebuilds the band lines at DMP['SPAN_BAND'] = 10 (directional_momentum.py:61-79).
## From prep() the reference consumes only: idx, next_close, same_close (minutely_close),
## p1 (signal_times), minutely_low, minutely_high and asset_close (the projected candle
## close). Everything else on the kernel call line is run_cell's own recomputation, which
## is what get_numba_parameters below reproduces.
##
## The rv_alloc IS clip bounds are READ from a frozen artifact (production init data
## never reaches the 2021-2025 calibration window).
## ---------------------------------------------------------------------------


@lru_cache(maxsize=200)
def load_rv_alloc_bounds(coin1, tf, bounds_path):
    """Frozen rv_alloc clip bounds (q_lo, q_hi) for one (asset, TF) cell, from the
    pre-dumped artifact (built by eod_scripts/directional_momentum/build_rv_alloc_bounds.py).
    These are the final expanding-quantile values over the IS calibration window — constant
    for every post-IS bar, i.e. every bar production will ever process. Raise if the cell is
    absent: never run with an unclipped rv.

    ONE bound pair per (asset, TF), SHARED BY BOTH SIDES. The reference clips `rv` once in
    run_cell, before either kernel is called; there is no per-side calibration, so the LONG
    and SHORT strategy objects of a cell load the identical row.

    Takes the full artifact PATH (strategy_config['rv_alloc_bounds_path']), not a directory:
    this file is not a frozen-bundle artifact resolved off `data_dir`, and the bundle copy
    holds only whatever cells a test run happened to build. lru_cache is safe here — unlike
    r_state, this artifact is rebuilt only when the IS window changes, never on a schedule.

    CALIBRATION-WINDOW WARNING for whoever builds the artifact: DMP slices the window with
    STRING labels (`rv.loc['2021-01-01':'2025-03-31']`, vendored_b1_dmp.py:15 and
    directional_momentum.py:25), so the WHOLE of 2025-03-31 is inside it. pair_momentum's B1
    path uses `ISE = pd.Timestamp('2025-03-31')` and excludes it. Measured on BTCUSDT the two
    conventions differ by 287 IS candles at TF=5 (23 at TF=60) and move q_hi by 1.0e-04 —
    enough to break bitwise parity. Never share bounds, or bound-building code, across the
    two sleeves.

    IN-WINDOW CAVEAT: the reference applies the EXPANDING quantile bar by bar, not these
    frozen scalars, so inside 2021-01-01..2025-03-31 a frozen-bounds reproduction can differ
    wherever the clip binds; before 2021 the reference's bounds are NaN and its rv_alloc is
    backfilled from the first 2021 value. Both regimes are pre-production and cost only
    SIZE, never timing (rv_alloc reaches the kernel exclusively through `allocation`, which
    appears in no condition). Full-history parity work must reproduce the expanding clip.

    SYMBOL FORM: the artifact is keyed by PLAIN symbols ("BTCUSDT") — the EOD builder reads
    the universe out of params.json. Live hands us the EXCHANGE-INTERNAL names
    ("BTC-USDT.PERP") that coin_param / submodel_parameters / ohlcv_data use, so translate
    here rather than renaming the symbol everywhere else. plain_symbol is a no-op on
    already-plain input, so the frozen-bundle path and every recorded matching fixture are
    unaffected. Without this, every DB-sourced cell raises the KeyError below and the whole
    sleeve refuses to start — all three pair sleeves hit this trap first.
    """
    coin1 = plain_symbol(coin1)
    df = pd.read_parquet(bounds_path)
    row = df[(df['coin1'] == coin1) & (df['tf'] == int(tf))]
    if len(row) != 1:
        raise KeyError(f"rv_alloc_bounds: no unique row for ({coin1}, {tf}T) "
                       f"in {bounds_path} (got {len(row)}). Build all 44 cells with "
                       f"eod_scripts/directional_momentum/build_rv_alloc_bounds.py")
    return float(row['q_lo'].iloc[0]), float(row['q_hi'].iloc[0])


class GetDirectionalMomentumSignal:
    """Batch warm-up driver for one ASSET — one instance serves all 8 of its cells.

    Constructed once per process by the signal generator; `_update_tup(tup)` points it at
    the next cell before that cell's `generate_signal()` runs.

    `df1` is the raw minute frame (NaNs intact); `warm_df` is its ffill/bfill'd twin. The
    reference derives every input from the FILLED frame — the candle resample at
    vendored_b1_dmp.prep:106 and the minute fields at :126 — so `warm_df` is what the
    pipeline reads. `df1` is retained for provenance and for the raw-tail handoff to live.
    """

    def __init__(self, tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE, df1, warm_df, curr_time, process_name):
        self.tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE = tup
        self.df1 = df1
        self.warm_df = warm_df
        self.curr_time = curr_time
        self.debug = False
        self.process_name = process_name
        self.dump_pandas_ls = []

    def get_sim_key(self, tup):
        sim_key = "_".join([str(getattr(tup, i)) for i in tup._asdict()])
        print(sim_key)
        return sim_key

    def _update_tup(self, tup):
        self.tup = tup

    def get_numba_parameters(self, tup, curr_time):
        """Build the DMP_v3_2 kernel inputs for ONE (asset, TF, side) cell on the minutely
        frame: features on the asset's TF candles (median20 / atr_eq@10 / long_vol /
        rv_alloc frozen-clip / z_median), shifted to the closing minute; next-2-min OHLC4
        fills; vol-target alloc, times the daily R on the SHORT side only.

        Returns [output_tup (this SIDE's kernel-arg order), aux_tup (debug arrays)].
        """
        sim_key = self.get_sim_key(tup)
        sc = tup.strategy_config
        tf = int(tup.tf)
        side = tup.side
        assert config_object.time_zone == "UTC", "DMP_v3_2 requires UTC day/bin edges"
        assert side in ("LONG", "SHORT"), f"unknown side {side!r}"

        cd = self.warm_df

        # ---- minutely working frame. warm_df is already the asset's own OHLCV with
        # ffill+bfill applied column-wise, matching the reference's
        # `raw[cols].ffill().bfill()` (prep:106) and its minute-field fill (prep:126).
        # SINGLE ASSET: no ratio is constructed anywhere, and there is exactly one OHLC4.
        mc = pd.DataFrame(index=cd.index)
        mc['mopen'] = cd['open']
        mc['mclose'] = cd['close']
        mc['mhigh'] = cd['high']
        mc['mlow'] = cd['low']
        mc['ob'] = (cd['open'] + cd['high'] + cd['low'] + cd['close']) / 4.0

        # ---- TF candle frame (resample_ohlc: dedup/sort + plain resample, KEEP NaN bins) --
        px = cd[['open', 'high', 'low', 'close']]
        px = px.loc[~px.index.duplicated(), :].sort_index()
        df = px.resample(f"{tf}T").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df = df.loc[~df.index.duplicated(), :]

        # ---- TF features (spans/windows off the tuple's strategy_config) -----------
        # atr_ewm_span is 10 here — run_cell's DMP['SPAN_BAND'], NOT prep's module-level 14.
        # The strategy object asserts this at construction; the value still flows from
        # config, so there is one source of truth.
        df['median20'] = df['close'].rolling(int(sc['median_window_bars'])).median()
        lr = np.log(df['close']).diff()
        df['long_vol'] = lr.ewm(span=int(sc['long_vol_ewm_span'])).std()
        df['atr_eq'] = df['close'] * lr.ewm(span=int(sc['atr_ewm_span'])).std()
        rv = lr.ewm(span=int(sc['rv_ewm_span'])).std() * float(sc['rv_scale'])
        q_lo, q_hi = load_rv_alloc_bounds(tup.coin1, tf, sc['rv_alloc_bounds_path'])
        df['rv_alloc'] = rv.clip(lower=q_lo, upper=q_hi)
        df['z_median'] = ((df['close'] - df['median20']) / df['atr_eq'].clip(lower=1e-12)) \
            .ewm(span=int(sc['zmed_ewm_span']), adjust=bool(sc['zmed_ewm_adjust'])).mean()
        # `ac` = the reference's `asset_close` — the projected CANDLE close, which is what
        # both kernels compare to the bands and use for the extreme/develop tests. It is NOT
        # the minute close (that is `mclose` -> same_close, used only for marking).
        df = df.rename(columns={'close': 'ac'})

        # ---- merge onto the minutely frame + shift feats to the CLOSING minute -----
        # (prep:122-126: outer merge; ROW shift tf-1; p1 from the shifted candle close
        # BEFORE the ffill; then ffill/bfill; forward-2-min OHLC4 fill.)
        #
        # The outer merge INJECTS candle-start labels that are absent from the minute index
        # whenever the feed has a gap. That is deliberate and matches the reference — the
        # shift is positional over the merged index, so the injected rows shift the grid the
        # same way there as here. `mopen` is left out of the ffill list because prep's `cf`
        # leaves minutely_open out too; it is debug-only and never reaches a kernel.
        feats = ['median20', 'atr_eq', 'long_vol', 'rv_alloc', 'z_median', 'ac']
        mc = mc.merge(df[feats], left_index=True, right_index=True, how='outer')
        mc[feats] = mc[feats].shift(tf - 1)
        mc['signal_times'] = np.where(~pd.isna(mc['ac']), 1, 0)
        ff = feats + ['mhigh', 'mlow', 'mclose', 'ob']
        mc[ff] = mc[ff].ffill().bfill()
        mc['next_close'] = (mc['ob'].shift(-1) + mc['ob'].shift(-2)) / 2.0

        # ---- trim the fill lookahead: last 2 minutes have NaN next_close. The batch never
        # trades them; live processes them incrementally from last_processed_ts.
        mc = mc.iloc[:-2]

        # ---- vol-target alloc + both Keltner lines (outside-kernel arrays) ---------------
        med = mc['median20'].to_numpy(dtype=np.float64)
        atr = mc['atr_eq'].to_numpy(dtype=np.float64)
        upper = med + float(tup.K) * atr
        lower = med - float(tup.K) * atr

        alloc = pd.Series(
            np.clip((float(sc['VT']) / mc['rv_alloc'].to_numpy(dtype=np.float64)) / float(tup.annf), 0, 1),
            index=mc.index,
        ).bfill().values

        # The daily market-vol multiplier R scales ENTRY sizing on the SHORT SIDE ONLY
        # (PRODUCTION directional_momentum.py:58-59, 83-84 — `alv * mm` is passed to
        # kern_short; kern_long receives the bare `alv`). R is capped at 1.0, so it can only
        # throttle. It is a CROSS-ASSET daily state (mean 15-minute rv over the 11 traded
        # assets, normalised by its rolling-60d median, expanding strict-less percentile,
        # 1-day lag) that a single-asset process cannot derive; it arrives from the artifact
        # built by eod_scripts/directional_momentum/build_r_state.py.
        #
        # This MUST stay in step with the streaming site in
        # directional_momentum_last_line_utils.py::update — applying R at only one of the
        # two makes a warm-up replay and a live run disagree on every short entry size.
        #
        # The LONG branch does not read the artifact at all: a "multiply by 1.0" would
        # couple the long side to a nightly file it has no business depending on, and would
        # turn a missing artifact into a silent no-op for half the sleeve.
        if side == "SHORT":
            Rm = r_multiplier_array(sc['r_state_path'], mc.index,
                                    logger_name=f"signal_gen_{self.process_name}")
            alloc = alloc * Rm
        else:
            Rm = None

        ts_array = mc.index.values.astype('datetime64[ns]').astype(np.int64)

        # ---- kernel-input bundle, in THIS SIDE's argument order. The two kernels differ in
        # exactly two slots — the minute extreme and the band line — so the tuple is built
        # per side rather than passing both and letting the kernel choose. The per-cell
        # scalars T1/TT/dde_mult/K/X/Z/EZ ride the tuple. 13 elements (no grace: DMP's
        # kernels have none, unlike B1's kern_v7). --------------------------------------
        minutely_extreme = mc['mlow'] if side == "LONG" else mc['mhigh']
        band_line = upper if side == "LONG" else lower

        output_tup = (
            mc['next_close'].to_numpy(dtype=np.float64),    # next_close
            mc['mclose'].to_numpy(dtype=np.float64),        # same_close
            mc['signal_times'].to_numpy(dtype=np.float64),  # p1
            minutely_extreme.to_numpy(dtype=np.float64),    # minutely_low | minutely_high
            mc['ac'].to_numpy(dtype=np.float64),            # asset_close
            med,                                            # median_line
            band_line,                                      # upper_line | lower_line
            atr,                                            # atr_eq
            mc['long_vol'].to_numpy(dtype=np.float64),      # long_vol
            mc['z_median'].to_numpy(dtype=np.float64),      # z_median
            float(sc['txn_cost']),                          # txn_cost
            float(sc['slippage_per_turn']),                 # slippage
            alloc,                                          # allocation
        )

        aux_tup = (
            ts_array,
            mc['mopen'].to_numpy(dtype=np.float64), mc['mhigh'].to_numpy(dtype=np.float64),
            mc['mlow'].to_numpy(dtype=np.float64), mc['mclose'].to_numpy(dtype=np.float64),
            mc['ob'].to_numpy(dtype=np.float64),
            mc['rv_alloc'].to_numpy(dtype=np.float64),
            upper, lower, Rm,
        )
        return [output_tup, aux_tup]

    def generate_signal(self):
        output_tup, aux_tup = self.get_numba_parameters(self.tup, self.curr_time)

        (ts_array,
         minutely_open, minutely_high, minutely_low, minutely_close,
         ohlc_based, rv_alloc, upper, lower, Rm) = aux_tup

        (next_close, same_close, p1, minutely_extreme, asset_close, median_line, band_line,
         atr_eq, long_vol, z_median, txn_cost, slippage, alloc) = output_tup

        tup = self.tup
        sc = tup.strategy_config

        # ONE kernel per object — this object IS one side. The dispatch is on the tuple, so
        # a cell's two objects run two different kernels over identical feature arrays.
        # `dde_mult` already carries M_LONG=4.0 / M_SHORT=1.5 times the per-TF DDE law.
        kernel = (cryptoasset_dmpv32_long_iact if tup.side == "LONG"
                  else cryptoasset_dmpv32_short_iact)
        array_output, last_output = kernel(
            next_close, same_close, p1, minutely_extreme, asset_close, median_line,
            band_line, atr_eq, long_vol, z_median, txn_cost, slippage, alloc,
            float(tup.T1), float(tup.TT), float(tup.dde_mult),
            float(tup.K), float(tup.X), float(tup.Z), float(tup.EZ),
            # development-latch selector -- see utils.directional_momentum_utils.
            # dmp_development_latch. 0 reproduces production exactly; .get() rather than
            # hard-indexing because live rows in `submodel_parameters` predate these keys.
            int(sc.get('latch_mode', 0)), float(sc.get('latch_bars', 20.0)),
            float(sc.get('latch_ret', 0.01)),
            int(sc.get('signal_invert', 0)),
        )

        # Wrap last-bar state (incl. carried kernel state) for the orchestrator / live.
        numba_output_cls = TradeOutputDirectionalMomentumLastOutputCls(**last_output._asdict())

        parent_id = self.tup.parent_stratid

        # Trade df. Single asset -> a single trade price; no leg division. `signal` is +1/0
        # on the long stream and -1/0 on the short stream, so signal_diff is non-zero on
        # exactly the entry and exit minutes of this side.
        tmp_df = pd.DataFrame({'signal': array_output.res_arr,
                               'tradeprice': array_output.tradeprice_arr}, index=ts_array)
        tmp_df.index = pd.to_datetime(tmp_df.index, unit='ns')
        tmp_df['signal_diff'] = tmp_df['signal'].diff().fillna(0)
        tmp_df['signal_id'] = tmp_df.index.map(lambda x: int(x.timestamp()))
        tmp_df['parent_trading_model'] = parent_id
        tmp_df['is_tp_sl'] = 0
        ## live_signals columns that Base.dump_hist_signal_df reads off the row by NAME.
        ## Without them the warm-up dump raises AttributeError inside prepare_init_data and
        ## kills the process before the signal loop ever starts. It only fires when
        ## hist_replay_dir is None, which is why every replay test passed.
        ##   case_num       0 -- DMP emits no case, and the column is NOT NULL.
        ##   execution_type the cell's DB route.
        ## Both match what backtest_codes/directional_momentum_backtest.py writes to
        ## backtest_signals, so the live and backtest streams stay directly diffable.
        tmp_df['case_num'] = 0
        tmp_df['execution_type'] = self.tup.exec_type

        tmp_df_signal_diff = tmp_df[tmp_df['signal_diff'] != 0]
        tmp_df_signal_diff['price'] = tmp_df_signal_diff['tradeprice']

        order_tag1 = ""
        order_tag_full = None      # FULL-length stream, for the debug dump only
        if len(tmp_df_signal_diff):
            tmp_df_signal_diff['order_tag'] = tmp_df_signal_diff.apply(
                lambda x: order_tag_generator(ts=x.name, parent_id=int(x['parent_trading_model']), signal=int(x['signal'])),
                axis=1,
            )
            tmp_df.loc[tmp_df_signal_diff.index, ['order_tag', 'price']] = tmp_df_signal_diff[['order_tag', 'price']]
            tmp_df['order_tag'] = tmp_df['order_tag'].ffill()
            tmp_df['price'] = tmp_df['price'].ffill()
            order_tag1 = tmp_df['order_tag'].iloc[-1]  # ffilled "active" tag at last bar

            ## Keep the full-length order-tag stream BEFORE the slice below. create_debug_df
            ## aligns every column against ts_array, so handing it the sliced (last-day)
            ## frame raises a length mismatch.
            order_tag_full = tmp_df['order_tag'].copy()

            ## Slice to the last day window (mirrors the sibling sleeves).
            last_ts = tmp_df.index[-1]
            start_time = last_ts.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=10)
            tmp_df = tmp_df.loc[start_time:last_ts, :]
            self.dump_pandas_ls.append(tmp_df.copy(deep=True))

        if self.debug:
            self.create_debug_df(
                ts_array,
                # Kernel inputs (per-minute arrays), under their kernel-argument names.
                next_close=next_close, same_close=same_close, signal_times=p1,
                minutely_extreme=minutely_extreme, asset_close=asset_close,
                median_line=median_line, band_line=band_line, atr_eq=atr_eq,
                long_vol=long_vol, z_median=z_median, allocation=alloc, rv_alloc=rv_alloc,
                # Both bands and both minute extremes, so a LONG dump and a SHORT dump of
                # the same cell are directly comparable column by column.
                upper_line=upper, lower_line=lower,
                minutely_open=minutely_open, minutely_high=minutely_high,
                minutely_low=minutely_low, minutely_close=minutely_close,
                ohlc_based=ohlc_based, r_multiplier=Rm,
                # Kernel array outputs.
                **array_output._asdict(),
                # Order-tag stream, full length (see order_tag_full above).
                order_tag=order_tag_full,
            )

        # Last timestamp the kernel actually processed (= ts_array[-1] AFTER the
        # tail-trim-2 in get_numba_parameters). live.py uses this as the anchor for the
        # first incremental update, so the 2 minutes the batch trim drops are processed
        # incrementally rather than skipped.
        last_processed_ts = pd.Timestamp(int(ts_array[-1]), tz='UTC')

        return numba_output_cls, order_tag1, last_processed_ts

    def create_debug_df(self, ts_array, **kwargs):
        df_ = pd.DataFrame({'ts': ts_array})
        for key, value in kwargs.items():
            if value is None:
                continue
            df_[key] = np.asarray(value)
        df_['ts'] = pd.to_datetime(df_['ts'], unit='ns')

        ## Per-SLEEVE, per-ASSET subdirectory: numpy_pandas_matching is shared by every
        ## sleeve, and parent_stratids are per-asset LOCAL (every asset reuses
        ## 10000001..10000008), so a flat path would have two assets overwrite each other's
        ## dumps. The two SIDES of a cell do not collide — they have different ids.
        out_dir = f"./numpy_pandas_matching/{SLEEVE}/{self.tup.coin1}"
        os.makedirs(out_dir, exist_ok=True)
        df_.to_parquet(f"{out_dir}/{self.tup.parent_stratid}_numpy.parquet")
