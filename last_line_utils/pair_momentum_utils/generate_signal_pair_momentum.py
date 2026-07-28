
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

from last_line_utils.pair_momentum_utils.pair_momentum_utils import PAIR_MOMENTUM_PARAM_TUPLE, SLEEVE, plain_symbol
from last_line_utils.pair_momentum_utils.r_state import r_multiplier_array
from utils.pair_momentum_utils import cryptopairs_b1v8v10_long_iact, TradeOutputPairMomentumLastOutputCls
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

#
import os


## ---------------------------------------------------------------------------
## B1_v8_v10 feature calculations — pandas port of the reference prep
## (crypto_sims/PRODUCTION/engines/pairs_momentum.py::prep). Deliberately pandas
## (not numpy/bottleneck): this is once-per-init batch code, and the reference prep
## leans on `ewm(span).std()` adjust=True debiased semantics everywhere — identical
## ops give bitwise parity with the golden book. Note the deliberate asymmetry:
## atr_eq / long_vol / rv use adjust=True (pandas default), z_median_rv uses
## adjust=False. Every span/window/tunable comes from the frozen snapshot via the
## tuple / strategy_config — nothing hardcoded.
##
## The rv_alloc IS clip bounds are READ from a frozen artifact (production init data
## never reaches the 2021-2025 calibration window).
##
## NOTE vs B1_v8_v9: v8_v10 has NO daily dominance gate — entries are not filtered by
## z_major > z_minor. The v8_v9 gate helpers have been removed. v8_v10 instead carries
## a cross-pair daily market-vol multiplier R on ENTRY SIZING only, read from the nightly
## artifact (see r_state.py) and applied in get_numba_parameters.
## ---------------------------------------------------------------------------


@lru_cache(maxsize=200)
def load_rv_alloc_bounds(coin1, coin2, tf, bounds_path):
    """Frozen rv_alloc clip bounds (q_lo, q_hi) for one (pair, TF) cell, from the
    pre-dumped artifact (built by eod_scripts/pair_momentum/build_rv_alloc_bounds.py).
    These are the final expanding-quantile values over the IS calibration window — constant
    for every post-IS bar, i.e. every bar production will ever process. Raise if the cell is
    absent: never run with an unclipped rv.

    Takes the full artifact PATH (strategy_config['rv_alloc_bounds_path']), not a directory:
    this file is not a frozen-bundle artifact resolved off `data_dir`, and the copy that used
    to sit there held only the 5 BTC/AVAX rows. lru_cache is safe here — unlike r_state, this
    artifact is rebuilt only when the IS window changes, never on a schedule.

    The artifact is keyed by PLAIN symbols ("BTCUSDT") — the EOD builder reads the
    universe out of params.json. Live hands us the EXCHANGE-INTERNAL names
    ("BTC-USDT.PERP") that coin_param / submodel_parameters / ohlcv_data use, so translate
    here rather than renaming the symbol everywhere else. plain_symbol is a no-op on
    already-plain input, so the frozen-bundle and offline paths are unaffected. Without
    this every DB-sourced cell raises the KeyError below and the whole sleeve refuses to
    start; pair_relative_value hit the same trap (load_qr1_cell_constants)."""
    coin1, coin2 = plain_symbol(coin1), plain_symbol(coin2)
    df = pd.read_parquet(bounds_path)
    row = df[(df['coin1'] == coin1) & (df['coin2'] == coin2) & (df['tf'] == int(tf))]
    if len(row) != 1:
        raise KeyError(f"rv_alloc_bounds: no unique row for ({coin1}, {coin2}, {tf}T) "
                       f"in {bounds_path} (got {len(row)}). Build all 85 cells with "
                       f"eod_scripts/pair_momentum/build_rv_alloc_bounds.py")
    return float(row['q_lo'].iloc[0]), float(row['q_hi'].iloc[0])


class GetPairsMomentumSignal:

    def __init__(self, tup: PAIR_MOMENTUM_PARAM_TUPLE, df1, df2, comb_df, curr_time, process_name,
                 log_r_staleness=True):

        self.tup: PAIR_MOMENTUM_PARAM_TUPLE = tup
        self.df1 = df1
        self.df2 = df2
        self.comb_df = comb_df
        self.curr_time = curr_time
        self.debug = False
        self.process_name = process_name
        ## Whether a stale r_state artifact raises a CRITICAL from this consumer.
        ##
        ## Defaults ON, and must stay that way for anything that SIZES: R multiplies the
        ## allocation of every new entry over a [0.25, 1.75] range, so trading off a stale
        ## R silently mis-sizes, and the alert is the only symptom.
        ##
        ## A consumer may opt out only when it demonstrably cannot be affected — i.e. it
        ## publishes nothing derived from `alloc`. `alloc` is read at exactly one place in
        ## each kernel variant (`tal[i] = alloc[i]*w`, `tal = alloc*w`) and `tal` is only
        ## carried and emitted, never tested, so signal / fill price / order tag are
        ## R-independent. The backtest producer publishes only those and opts out; it would
        ## otherwise raise 85 CRITICALs per pass about a condition it cannot suffer from,
        ## which is how a real staleness alert gets trained out of the monitoring system.
        self.log_r_staleness = log_r_staleness
        self.dump_pandas_ls = []

    def get_sim_key(self, tup):
        sim_key = "_".join([str(getattr(tup, i)) for i in tup._asdict()])
        print(sim_key)
        return sim_key

    def _update_tup(self, tup):
        self.tup = tup

    # @profile
    def get_numba_parameters(self, tup, curr_time):
        """Build the B1_v8_v10 kernel inputs for ONE (pair, TF) cell on the minutely
        frame: features on the ratio TF candles (median20 / atr_eq / long_vol /
        rv_alloc frozen-clip / z_median_rv), shifted to the closing minute; per-leg
        next-2-min OHLC4 fills; vol-target alloc. Pandas port of the reference prep
        (PRODUCTION/engines/pairs_momentum.py::prep).
        Returns [output_tup (kernel-arg order), aux_tup (debug arrays)]."""
        sim_key = self.get_sim_key(tup)
        sc = tup.strategy_config
        tf = int(tup.tf)
        assert config_object.time_zone == "UTC", "B1_v8_v10 requires UTC day/bin edges"

        cd = self.comb_df

        # ---- minutely working frame (ratio + legs). comb_df is already the outer
        # merge of the legs with ratio O/C + high=max(O,C)/low=min(O,C), ffill+bfill
        # applied column-wise (identical to prep_v7's raw frame after its ff fill;
        # legs go missing row-wise so mean/ffill commute for the OHLC4 below).
        mc = pd.DataFrame(index=cd.index)
        mc['mopen'] = cd['open']
        mc['mclose'] = cd['close']
        mc['mhigh'] = cd['high']
        mc['mlow'] = cd['low']
        mc['sc1'] = cd['close_1']
        mc['sc2'] = cd['close_2']
        mc['ob1'] = (cd['open_1'] + cd['high_1'] + cd['low_1'] + cd['close_1']) / 4.0
        mc['ob2'] = (cd['open_2'] + cd['high_2'] + cd['low_2'] + cd['close_2']) / 4.0

        # ---- TF candle frame (prep_v7: dedup/sort + plain resample, KEEP NaN bins) --
        ratio = cd[['open', 'high', 'low', 'close']]
        ratio = ratio.loc[~ratio.index.duplicated(), :].sort_index()
        df = ratio.resample(f"{tf}T").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df = df.loc[~df.index.duplicated(), :]

        # ---- TF features (spans/windows off the tuple's strategy_config) -----------
        df['median20'] = df['close'].rolling(int(sc['median_window_bars'])).median()
        lr = np.log(df['close']).diff()
        df['long_vol'] = lr.ewm(span=int(sc['long_vol_ewm_span'])).std()
        df['atr_eq'] = df['close'] * lr.ewm(span=int(sc['atr_ewm_span'])).std()
        rv = lr.ewm(span=int(sc['rv_ewm_span'])).std() * float(sc['rv_scale'])
        q_lo, q_hi = load_rv_alloc_bounds(tup.coin1, tup.coin2, tf, sc['rv_alloc_bounds_path'])
        df['rv_alloc'] = rv.clip(lower=q_lo, upper=q_hi)
        df['z_median_rv'] = ((df['close'] - df['median20']) / df['atr_eq'].clip(lower=1e-12)) \
            .ewm(span=int(sc['zmed_ewm_span']), adjust=bool(sc['zmed_ewm_adjust'])).mean()
        df = df.rename(columns={'close': 'spc'})

        # ---- merge onto the minutely frame + shift feats to the CLOSING minute -----
        # (prep_v7 lines 60-69: outer merge; ROW shift tf-1; p1 from the shifted candle
        # close BEFORE the ffill; then ffill/bfill; forward-2-min OHLC4 fills.)
        feats = ['median20', 'atr_eq', 'long_vol', 'rv_alloc', 'z_median_rv', 'spc']
        mc = mc.merge(df[feats], left_index=True, right_index=True, how='outer')
        mc[feats] = mc[feats].shift(tf - 1)
        mc['signal_times'] = np.where(~pd.isna(mc['spc']), 1, 0)
        ff = feats + ['mhigh', 'mlow', 'sc1', 'sc2', 'ob1', 'ob2']
        mc[ff] = mc[ff].ffill().bfill()
        mc['next1'] = (mc['ob1'].shift(-1) + mc['ob1'].shift(-2)) / 2.0
        mc['next2'] = (mc['ob2'].shift(-1) + mc['ob2'].shift(-2)) / 2.0

        # ---- trim the fill lookahead: last 2 minutes have NaN next1/next2. The batch
        # never trades them; live processes them incrementally from last_processed_ts.
        mc = mc.iloc[:-2]

        # ---- vol-target alloc + Keltner upper (outside-kernel arrays) ---------------
        # The daily market-vol multiplier R scales ENTRY sizing (PRODUCTION
        # pairs_momentum.py:112-115). Note the ORDER: clip to [0,1] FIRST, then multiply,
        # so the product may exceed 1 and reach 1.75. R is a CROSS-PAIR daily state (mean
        # 15T rv_alloc over all 17 pairs, normalised by its rolling-60d median, expanding
        # strict-less percentile, 1-day lag) that a single-pair process cannot derive; it
        # arrives from the artifact built by eod_scripts/pair_momentum/build_r_state.py.
        # This MUST stay in step with the streaming site in
        # pair_momentum_last_line_utils.py::update — applying R at only one of the two
        # makes a warm-up replay and a live run disagree on every entry size.
        ## logger_name=None makes r_state's _log a no-op -- see log_r_staleness in __init__
        ## for when a consumer is entitled to that.
        Rm = r_multiplier_array(
            sc['r_state_path'], mc.index,
            logger_name=(f"signal_gen_{self.process_name}" if self.log_r_staleness else None))
        alloc = pd.Series(
            np.clip((float(sc['VT']) / mc['rv_alloc'].to_numpy(dtype=np.float64)) / float(tup.annf), 0, 1),
            index=mc.index,
        ).bfill().values * Rm
        med = mc['median20'].to_numpy(dtype=np.float64)
        atr = mc['atr_eq'].to_numpy(dtype=np.float64)
        upper = med + float(tup.K) * atr

        ts_array = mc.index.values.astype('datetime64[ns]').astype(np.int64)

        # ---- kernel-input bundle (argument order of cryptopairs_b1v8v10_long_iact;
        # the per-cell scalars T1/TT/DDE/GRACE/K/X/Z/EZ ride the tuple). 15 elements —
        # v8_v9's trailing zM/zm dominance arrays are gone. ---------------------------
        output_tup = (
            mc['next1'].to_numpy(dtype=np.float64),        # next1
            mc['sc1'].to_numpy(dtype=np.float64),          # sc1
            mc['signal_times'].to_numpy(dtype=np.float64), # p1
            mc['mlow'].to_numpy(dtype=np.float64),         # mlow
            mc['next2'].to_numpy(dtype=np.float64),        # next2
            mc['sc2'].to_numpy(dtype=np.float64),          # sc2
            mc['spc'].to_numpy(dtype=np.float64),          # spc
            med,                                           # med
            upper,                                         # upper
            atr,                                           # atr
            mc['long_vol'].to_numpy(dtype=np.float64),     # lv
            mc['z_median_rv'].to_numpy(dtype=np.float64),  # zmed
            float(sc['txn_cost']),                         # tc
            float(sc['slippage_per_leg_per_turn']),        # slip
            alloc,                                         # alloc
        )

        aux_tup = (
            ts_array,
            mc['mopen'].to_numpy(dtype=np.float64), mc['mhigh'].to_numpy(dtype=np.float64),
            mc['mlow'].to_numpy(dtype=np.float64), mc['mclose'].to_numpy(dtype=np.float64),
            mc['ob1'].to_numpy(dtype=np.float64), mc['ob2'].to_numpy(dtype=np.float64),
            mc['rv_alloc'].to_numpy(dtype=np.float64),
        )
        return [output_tup, aux_tup]

    def generate_signal(self):
        output_tup, aux_tup = self.get_numba_parameters(self.tup, self.curr_time)

        (ts_array,
         pair_minutely_open, pair_minutely_high, pair_minutely_low, pair_minutely_close,
         coin1_ohlc_based, coin2_ohlc_based,
         rv_alloc) = aux_tup

        (next1, sc1, p1, mlow, next2, sc2, spc, med, upper, atr, lv, zmed,
         txn_cost, slippage, alloc) = output_tup

        tup = self.tup
        sc = tup.strategy_config

        # Single long-only stream — one kernel, no dispatch variants and no second
        # entry (B1 has one signal stream; the reversal2 res2/order_tag2 machinery
        # does not apply).
        array_output, last_output = cryptopairs_b1v8v10_long_iact(
            next1, sc1, p1, mlow, next2, sc2, spc, med, upper, atr, lv, zmed,
            txn_cost, slippage, alloc,
            float(tup.T1), float(tup.TT), float(tup.DDE), float(sc['GRACE']),
            float(tup.K), float(tup.X), float(tup.Z), float(tup.EZ),
        )

        # Wrap last-bar state (incl. carried kernel state) for the orchestrator / live.
        numba_output_cls = TradeOutputPairMomentumLastOutputCls(**last_output._asdict())

        parent_id = self.tup.parent_stratid

        # Trade df: leg fills tradeprice1/tradeprice2 (pair-ratio fill = tp1/tp2,
        # consistent with the reference's log(tp1)-log(tp2) PnL).
        tmp_df = pd.DataFrame({'signal': array_output.res_arr,
                               'tradeprice1': array_output.tradeprice1_arr,
                               'tradeprice2': array_output.tradeprice2_arr}, index=ts_array)
        tmp_df.index = pd.to_datetime(tmp_df.index, unit='ns')
        tmp_df['signal_diff'] = tmp_df['signal'].diff().fillna(0)
        tmp_df['signal_id'] = tmp_df.index.map(lambda x: int(x.timestamp()))
        tmp_df['parent_trading_model'] = parent_id
        tmp_df['is_tp_sl'] = 0
        ## live_signals columns that Base.dump_hist_signal_df reads off the row by NAME.
        ## Without them the warm-up dump raises AttributeError inside prepare_init_data and
        ## kills the process before the signal loop ever starts. It only fires when
        ## hist_replay_dir is None, which is why every replay test passed.
        ##   case_num       0 -- B1 emits no case, and the column is NOT NULL.
        ##   execution_type the cell's DB route.
        ## Both match what backtest_codes/pair_momentum_backtest.py writes to
        ## backtest_signals, so the live and backtest streams stay directly diffable.
        tmp_df['case_num'] = 0
        tmp_df['execution_type'] = self.tup.exec_type

        tmp_df_signal_diff = tmp_df[tmp_df['signal_diff'] != 0]
        tmp_df_signal_diff['price'] = tmp_df_signal_diff['tradeprice1'] / tmp_df_signal_diff['tradeprice2']

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
            ## frame raises a length mismatch — the debug path had never been exercised.
            order_tag_full = tmp_df['order_tag'].copy()

            ## Slice to the last day window (mirrors SA pattern).
            last_ts = tmp_df.index[-1]
            start_time = last_ts.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=10)
            tmp_df = tmp_df.loc[start_time:last_ts, :]
            self.dump_pandas_ls.append(tmp_df.copy(deep=True))

        if self.debug:
            self.create_debug_df(
                ts_array,
                # Kernel inputs (per-minute arrays).
                next_close1=next1, same_close1=sc1, signal_times=p1,
                minutely_low=mlow, next_close2=next2, same_close2=sc2,
                spc=spc, med=med, upper=upper, atr=atr, lv=lv, zmed=zmed,
                allocation=alloc, rv_alloc=rv_alloc,
                # Pair-ratio + per-leg auxiliary at minute resolution.
                minutely_open=pair_minutely_open, minutely_high=pair_minutely_high,
                minutely_close=pair_minutely_close,
                coin1_ohlc_based=coin1_ohlc_based, coin2_ohlc_based=coin2_ohlc_based,
                # Kernel array outputs.
                **array_output._asdict(),
                # Order-tag stream, full length (see order_tag_full above).
                order_tag=order_tag_full,
            )

        # Last timestamp the kernel actually processed (= ts_array[-1] AFTER the
        # tail-trim-2 in get_numba_parameters). live.py uses this as the anchor
        # for the first incremental update, so the 2 minutes the batch trim drops
        # are processed incrementally rather than skipped.
        last_processed_ts = pd.Timestamp(int(ts_array[-1]), tz='UTC')

        return numba_output_cls, order_tag1, last_processed_ts

    def create_debug_df(self, ts_array, **kwargs):
        df_ = pd.DataFrame({'ts': ts_array})
        for key, value in kwargs.items():
            if value is None:
                continue
            df_[key] = np.asarray(value)
        df_['ts'] = pd.to_datetime(df_['ts'], unit='ns')

        ## Per-SLEEVE, per-PAIR subdirectory: numpy_pandas_matching is shared by all five
        ## sleeves, and parent_stratids are per-pair LOCAL (every pair reuses
        ## 40000001/03/05/07/09, see the band comment in pair_momentum_utils.py), so a flat path would have two
        ## pairs overwrite each other's dumps. Only self.tup is in scope here — the base
        ## coin names (BTC/AVAX) are not — hence the exchange-symbol form, which also
        ## matches the coin1/coin2 columns in rv_alloc_bounds.parquet.
        out_dir = f"./numpy_pandas_matching/{SLEEVE}/{self.tup.coin1}_{self.tup.coin2}"
        os.makedirs(out_dir, exist_ok=True)
        df_.to_parquet(f"{out_dir}/{self.tup.parent_stratid}_numpy.parquet")
