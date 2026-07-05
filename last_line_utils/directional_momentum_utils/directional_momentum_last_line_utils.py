
from collections import deque
import numpy as np
import datetime as dt
from enum import Enum
from dataclasses import dataclass
from pytz import timezone
import pandas as pd
import copy
from collections import namedtuple

from config.config_read import config_object
from utils.utils import resample_stock_data
from shared_codes.utils.misc_utils import check_nan

from utils.directional_momentum_utils import TradeOutputDirectionalMomentumLastOutputCls
from last_line_utils.directional_momentum_utils.r_state import get_r

from .directional_momentum_utils import (DirectionalMomentumBaseClass,
                                         DIRECTIONAL_MOMENTUM_PARAM_TUPLE,
                                         DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE)
from .generate_signal_directional_momentum import GetDirectionalMomentumSignal, load_rv_alloc_bounds

## DMP_v3_2 streaming feature classes — five thin calcs, ALL on the asset's candle frame.
## Each is self-sufficient (owns its primitives, consumes only OHLC), so there are no
## cross-calc kwargs and NO ordering constraint on the driver loop — unlike the lead-lag
## sleeve, where the ATR family had to run before the Bollinger calc.
##
## The LONG and SHORT strategy objects of a cell each own their own instances of all five.
## They are fed the identical candle stream and therefore hold identical values; the two
## sides diverge only at the comparison level (upper vs lower band, low vs high, peak vs
## trough) and in `dde_mult`. Sharing one set of calcs between the two objects would save a
## few float ops per candle at the cost of a shared-mutable-state coupling between two
## models that are otherwise completely independent — not a trade worth making.
from .features.median_utils import RollingMedianCalc
from .features.vol_utils import AtrEqCalc, LongVolCalc, RvAllocCalc
from .features.zmed_utils import ZMedianRvCalc

#
import os


## MARKET MA AND HHLL CACHE
MARKET_MA_CACHE = {}
MARKET_HHLL_CACHE = {}
DATA_CACHE = {}

###
class MarketState(Enum):
    OPEN = 1
    CLOSE = 0

@dataclass
class OHLCV:
    st_time: dt.datetime
    end_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

##
class DirectionalMomentumOHLCV_Aggregator(DirectionalMomentumBaseClass):
    ## Copied VERBATIM from pair_momentum's aggregator (per the per-sleeve convention that
    ## nothing is imported across sleeves). It is instrument-agnostic — it aggregates
    ## whatever 1-minute OHLCV stream it is handed — so the only change is the class name.
    ## Here it is fed the ASSET's own bar rather than a synthesized ratio bar.
    def __init__(self, tup, timeframe_minutes: int, offset="00:00:00"):
        ## Add checks that data given as input will be in UTC.
        self.tup = tup
        self.timeframe = dt.timedelta(minutes=timeframe_minutes)
        self.offset_time = self._parse_offset(offset)
        self.current_ohlcv = None
        self.time_bucket_start = None
        self.agg_ohlcv_deque = deque(maxlen=5)  # deque to store finalized OHLCV bars
        self.min_ohlcv_deque = deque(maxlen=5)  # deque to store 1 min OHLCV bars

        self.input_tz = timezone("UTC") # All the input data will be in UTC.
        self.target_tz = timezone(config_object.time_zone) # All the input will be converted to this timezone.

        ##
        self.bar_completed = False
        self.market_close_time = None
        self.market_start_time = None
        self.latest_state = None
        self.last_process_time = None
        self.prev_minutely_ohlcv = None
        self.gap_in_minutes = 1

    def check_tz(self, tz1, tz2):
        return tz1.zone == tz2.zone

    def update_market_times(self, base_timestamp, market_state: MarketState):
        ## Timestamp should be tz aware.
        assert self.check_tz(base_timestamp.tz, self.input_tz), "Timestamp should be timezone aware."
        # assert base_timestamp.tzinfo == self.base_tz, "Timestamp should be timezone aware."
        timestamp = base_timestamp.astimezone(self.target_tz)
        assert self.check_tz(timestamp.tzinfo, self.target_tz), "Timestamp should be timezone aware."

        ## market state check
        assert isinstance(market_state, MarketState), "Market state should be of type MarketState."

        if market_state == MarketState.OPEN and self.latest_state != MarketState.OPEN:
            self.latest_state = MarketState.OPEN
            if self.market_close_time is None:
                self.gap_in_minutes = 1
            else:
                self.gap_in_minutes = int((timestamp - self.market_close_time).total_seconds() // 60)

            self.market_start_time = timestamp
            msg_ = f"{timestamp}: Market state changed to OPEN."

        elif market_state == MarketState.CLOSE and self.latest_state != MarketState.CLOSE:
            self.latest_state = MarketState.CLOSE
            self.market_close_time = timestamp
            msg_ = f"{timestamp}: Market state changed to CLOSE."
        else:
            self.gap_in_minutes = 1
            msg_ = f"{timestamp}: Market state is same as previous state. {market_state}"

        ###
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

    def _parse_offset(self, offset):
        """Parse the offset string in 'HH:MM:SS' format."""
        hours, minutes, seconds = map(int, offset.split(':'))
        return dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def _init_data(self, df, type_='minutely'):
        # df: minutely data
        assert self.check_tz(df.index[0].tz, self.target_tz), "Timestamp should be timezone aware."
        assert len(df) == 1, "Data should be of length 1."

        if type_ == "minutely":
            st_ts = df.index[0]
            end_ts = st_ts + dt.timedelta(minutes=1)
            ohlcv_obj = OHLCV(st_time=st_ts, end_time=end_ts, open=df.iloc[0]['open'], high=df.iloc[0]['high'], low=df.iloc[0]['low'], close=df.iloc[0]['close'], volume=df.iloc[0]['volume'])
            self.prev_minutely_ohlcv = ohlcv_obj
            self.min_ohlcv_deque.append(ohlcv_obj)

        elif type_ == "resampled":
            st_ts = df.index[0]
            end_ts = st_ts + self.timeframe - dt.timedelta(seconds=1)
            ohlcv_obj = OHLCV(st_time=st_ts, end_time=end_ts, open=df.iloc[0]['open'], high=df.iloc[0]['high'], low=df.iloc[0]['low'], close=df.iloc[0]['close'], volume=df.iloc[0]['volume'])
            self.agg_ohlcv_deque.append(ohlcv_obj)

    def _initialize_ohlcv(self, open_price, high_price, low_price, close_price, volume, timestamp):
        """Initialize a new OHLCV entry."""
        # assert timestamp.tzinfo == None, "Timestamp should not be timezone aware."
        assert self.check_tz(timestamp.tz, self.target_tz), "Timestamp should be timezone aware."

        ## Since default is UTC, we need to convert it to target timezone.
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # timestamp = timestamp.replace(tzinfo=self.base_tz).astimezone(self.target_tz)
        self.time_bucket_start = self._calculate_time_bucket_start(timestamp)
        self.bucket_end = self.time_bucket_start + self.timeframe - dt.timedelta(seconds=1)
        self.bar_completed = False

        self.current_ohlcv = OHLCV(st_time=self.time_bucket_start, end_time=self.bucket_end, open=open_price,
                                   high=high_price, low=low_price, close=close_price, volume=volume)

    def _aggregate_ohlcv(self, high_price, low_price, close_price, volume):
        """Update the OHLCV values based on new incoming data."""
        if self.current_ohlcv is None:
            return
        self.current_ohlcv.high = max(self.current_ohlcv.high, high_price)
        self.current_ohlcv.low = min(self.current_ohlcv.low, low_price)
        self.current_ohlcv.close = close_price
        self.current_ohlcv.volume += volume

    def _calculate_time_bucket_start_old(self, timestamp):
        """Calculate the start of the time bucket based on the offset and timeframe."""
        # time_adjusted = timestamp - self.offset_time
        # bucket_start = time_adjusted - (time_adjusted - dt.datetime.min) % self.timeframe
        # return bucket_start + self.offset_time
        pass

    def _calculate_time_bucket_start(self, timestamp):
        ##
        if timestamp.tzinfo is not None:
            # Ensure offset_time and datetime.min match the timezone of timestamp
            min_dt = dt.datetime.min.replace(tzinfo=timestamp.tzinfo)
        else:
            # If timestamp is naive, keep everything naive
            min_dt = dt.datetime.min

        ##
        time_adjusted = timestamp - self.offset_time
        bucket_start = time_adjusted - (time_adjusted - min_dt) % self.timeframe
        return bucket_start + self.offset_time

    def update(self, open_price, high_price, low_price, close_price, volume, data_timestamp):
        """Update the OHLCV data based on the latest 1-minute data."""
        assert self.check_tz(data_timestamp.tz, timezone('UTC')), "Timestamp should be timezone aware."

        ## OHLCV base timestamp
        min_ohlcv_strt_ts = data_timestamp.astimezone(self.target_tz)
        min_ohlcv_end_ts = min_ohlcv_strt_ts + dt.timedelta(minutes=1)

        ##
        if check_nan(open_price) or check_nan(high_price) or check_nan(low_price) or check_nan(close_price):
            open_price = self.prev_minutely_ohlcv.open
            high_price = self.prev_minutely_ohlcv.high
            low_price = self.prev_minutely_ohlcv.low
            close_price = self.prev_minutely_ohlcv.close
            volume = self.prev_minutely_ohlcv.volume
            msg_ = f"OHLCV data is nan. Using previous minutely OHLCV data. {min_ohlcv_strt_ts} {open_price}, {high_price}, {low_price}, {close_price}, {volume}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        else:
            self.prev_minutely_ohlcv = OHLCV(st_time=min_ohlcv_strt_ts, end_time=min_ohlcv_end_ts, open=open_price, high=high_price, low=low_price, close=close_price, volume=volume)

        ###
        if self.last_process_time is None: # UTC
            self.last_process_time = data_timestamp

        else:
            assert data_timestamp > (self.last_process_time), f"Timestamp should be in increasing order. {data_timestamp} {self.last_process_time}"
            self.last_process_time = data_timestamp

        ## Appending minutely OHLCV
        if self.market_close_time < self.market_start_time: # market is open.
            st_ts = min_ohlcv_strt_ts
            end_ts = min_ohlcv_end_ts
            ohlcv_obj = OHLCV(st_time=st_ts, end_time=end_ts, open=open_price, high=high_price, low=low_price, close=close_price, volume=volume)
            self.min_ohlcv_deque.append(ohlcv_obj)

        else:
            msg_ = f"{min_ohlcv_strt_ts}: Market is closed."
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        ##
        if self.current_ohlcv is None and self.latest_state == MarketState.OPEN:
            msg_ = f"Initalizing OHLCV bar. {min_ohlcv_strt_ts} {open_price}, {high_price}, {low_price}, {close_price}, {volume}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            self._initialize_ohlcv(open_price, high_price, low_price, close_price, volume, min_ohlcv_strt_ts)
            return

        if min_ohlcv_end_ts >= self.time_bucket_start + self.timeframe:
            # import ipdb; ipdb.set_trace()
            msg_ = f"{min_ohlcv_strt_ts}: OHLCV bar completed."
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            self._aggregate_ohlcv(high_price, low_price, close_price, volume)
            self._finalize_current_ohlcv()
        else:
            msg_ = f"{min_ohlcv_strt_ts}: OHLCV bar not completed."
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            # Aggregate data into the current OHLCV
            self._aggregate_ohlcv(high_price, low_price, close_price, volume)

    def _finalize_current_ohlcv(self):
        """Finalize the current OHLCV and add it to the list."""
        bucket_start = self.time_bucket_start
        bucket_end = self.bucket_end

        ## Check that bucket end and market_start_time have same timezone.
        assert self.check_tz(bucket_end.tzinfo, self.target_tz), "Timestamp should be timezone aware."
        assert self.check_tz(bucket_start.tzinfo, self.target_tz), "Timestamp should be timezone aware."
        assert self.check_tz(self.market_start_time.tzinfo, self.target_tz), "Timestamp should be timezone aware."
        assert self.check_tz(self.market_close_time.tzinfo, self.target_tz), "Timestamp should be timezone aware."

        ##
        if self.market_close_time > self.market_start_time:
            msg_ = f"{bucket_start}: {bucket_end}: Market is closed."
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            if self.current_ohlcv is None:
                msg_ = f"{bucket_start}: {bucket_end}: OHLCV bar not completed."
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                self.bar_completed = False
                self.current_ohlcv = None
                return

            if bucket_start >= self.market_close_time and bucket_end >= self.market_close_time:
                self.current_ohlcv = None
                self.bar_completed = None

                msg_ = f"{bucket_start}: {bucket_end}: is after market close time."
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                self.bar_completed = False
                self.current_ohlcv = None
                return

        else:
            if bucket_start >= self.market_close_time and bucket_end <= self.market_start_time:
                msg_ = f"{bucket_start}: {bucket_end}: is after market close time."
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                self.bar_completed = False
                self.current_ohlcv = None
                return

        ##
        msg_ = f"{bucket_start}: {bucket_end}: OHLCV bar completed. {self.current_ohlcv}"
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        ##
        self.agg_ohlcv_deque.append(self.current_ohlcv)
        self.bar_completed = True
        self.current_ohlcv = None

###
class DIRECTIONAL_MOMENTUM_PARENT_ID_STRAT_OBJ(DirectionalMomentumBaseClass):
    """
    Per-(asset, TF, SIDE) streaming strategy object for DMP_v3_2 Directional Momentum. Owns:
      - The asset-frame OHLCV aggregator (1-min -> candle TF).
      - The five streaming feature updaters listed at the top of this module
        (median20 / atr_eq / long_vol / rv_alloc / z_median).
      - (LATER STEPS) the batch warm-up and per-bar emission of the numba input tuple
        for the _ll kernel.

    ONE SIDE PER OBJECT. A DMP cell runs a long Keltner breakout and a mirrored short
    simultaneously — measured on the goldens, both are on at once 72.8% of minutes — and we
    model them as two fully independent objects with their own parent_stratid rather than
    one dual-stream object. Consequences worth knowing:
      * `tup.side` selects the kernel and is the ONLY thing that differs, together with
        `tup.dde_mult` (M_LONG=4.0 vs M_SHORT=1.5 times the per-TF DDE law).
      * The SHORT side additionally multiplies its allocation by the daily R state; the
        LONG side never sees R. R is capped at 1.0, so a DMP target never exceeds 1.0.
      * There is one order tag per object, not two. The `order_tag2` machinery on the pair
        sleeves exists for a genuine second stream WITHIN one model (QR1's scale-in
        tranche); DMP's second stream is a second model, with its own tag at its own id.

    SINGLE ASSET: there is no ratio and no second leg. The aggregator is fed the asset's own
    minute bar, and the only per-minute quantities the kernels need beyond the candle
    features are `minutely_low` / `minutely_high` (intrabar hard stop + can_new re-arm),
    which come straight off the aggregator's min_ohlcv_deque rather than from a calc.

    Migration state: `__init__` is wired. `initialize()` and `update()` are the next build
    steps and currently raise — they are NOT retired bodies from another strategy.
    """

    def __init__(self, tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE, logger_name: str):
        self.tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE = tup
        DirectionalMomentumBaseClass.logger_name = logger_name

        # Fail fast on a side we cannot dispatch, rather than at the kernel call site
        # hundreds of minutes into a replay.
        assert tup.side in ("LONG", "SHORT"), f"unknown side {tup.side!r} for {tup.parent_stratid}"

        # Feature registries. The driver fans each completed candle into
        # resample_update_func_dict and each minute into minute_update_func_dict.
        # DMP_v3_2 computes every feature on the asset's CANDLE frame, so
        # minute_update_func_dict stays EMPTY — the only minute-level quantities the
        # kernels need (minutely_low / minutely_high, for the hard stop and the can_new
        # re-arm) come straight off the aggregator's min_ohlcv_deque, not from a calc.
        self.minute_update_func_dict = {}
        self.resample_update_func_dict = {}
        self.agg_ohlcv_dict = {"open_agg": None, "high_agg": None, "low_agg": None,
                               "close_agg": None, "volume_agg": None}

        agg_int = int(tup.agg_time.split("T")[0])
        self.OHLCV_AGGREGATOR = DirectionalMomentumOHLCV_Aggregator(tup, agg_int, offset=config_object.offset)

        # ---- streaming feature calcs (candle frame) --------------------------------
        # Every span/window comes off the frozen snapshot via strategy_config; nothing
        # is hardcoded. Keyed by plain strings rather than the shared FEATURE_NAME enum,
        # whose ATR / Z_MEDIAN members denote different quantities in sa_directional.
        sc = tup.strategy_config

        # IS-frozen rv clip bounds for THIS (asset, TF) cell — the same row for both sides.
        # Raises if absent: never run with an unclipped rv.
        q_lo, q_hi = load_rv_alloc_bounds(tup.coin1, tup.tf, sc['rv_alloc_bounds_path'])

        self.median_calc = RollingMedianCalc(int(sc['median_window_bars']))
        self.resample_update_func_dict['median20'] = self.median_calc

        # span 10 (DMP['SPAN_BAND']), NOT vendored_b1_dmp's module-level 14 — run_cell
        # rebuilds the band lines and prep's are discarded. Asserted below.
        self.atr_eq_calc = AtrEqCalc(int(sc['atr_ewm_span']))
        self.resample_update_func_dict['atr_eq'] = self.atr_eq_calc

        self.long_vol_calc = LongVolCalc(int(sc['long_vol_ewm_span']))
        self.resample_update_func_dict['long_vol'] = self.long_vol_calc

        self.rv_alloc_calc = RvAllocCalc(int(sc['rv_ewm_span']), float(sc['rv_scale']), q_lo, q_hi)
        self.resample_update_func_dict['rv_alloc'] = self.rv_alloc_calc

        self.zmed_calc = ZMedianRvCalc(int(sc['median_window_bars']), int(sc['atr_ewm_span']),
                                       int(sc['zmed_ewm_span']), bool(sc['zmed_ewm_adjust']))
        self.resample_update_func_dict['zmed'] = self.zmed_calc

        # The single most likely way to silently reproduce the WRONG strategy is to build
        # the bands at prep()'s span 14 instead of run_cell's 10 — the result still looks
        # like a plausible Keltner book. Cheap to assert once per cell at construction.
        assert self.atr_eq_calc.span == 10, (
            f"atr_ewm_span is {self.atr_eq_calc.span}, expected 10. DMP_v3_2 builds its band "
            f"lines at DMP['SPAN_BAND']=10 (directional_momentum.py:71); 14 is the value "
            f"vendored_b1_dmp.prep() uses and run_cell discards.")
        assert self.zmed_calc.atr_span == self.atr_eq_calc.span, (
            "z_median must divide by the SAME atr_eq the bands use "
            f"({self.zmed_calc.atr_span} vs {self.atr_eq_calc.span}).")

        # NOTE: no second-leg state is held here — DMP is single-asset. live.py owns the
        # per-minute prices (same_close, and the fill that goes in the next_close slot) and
        # passes them straight to the kernel from its INPUT_DATA_TUPLE.
        self.is_bar_completed = False

    def initialize(self, warm_df, inst_status_df, curr_time, process_name,
                   signal_generator: GetDirectionalMomentumSignal):
        """
        Historical warm-up. Drives `signal_generator.generate_signal()` once to seed
        `numba_cls` / the order tag, then batch-feeds the candle frame to every feature
        updater so live `update()` calls can resume from the last bar.
        """
        msg_ = f"Initializing the strategy: {self.tup}"
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        # Drive the historical array kernel and pull the order tag. Each DMP object is ONE
        # SIDE and therefore ONE stream — one entry, one exit, size constant for the life of
        # the trade. There is no second entry and no partial take (TT only latches the
        # trail), so generate_signal returns a 3-tuple and there is no order_tag2; the
        # QR1 dual-tranche contract does not apply. The other side of this cell is a
        # separate object that ran, or will run, its own generate_signal.
        #
        # last_processed_ts = the last bar batch's kernel actually processed (= one bar
        # before the trim-2 cliff, since next_close needs t+1 and t+2). live.py uses it as
        # the start anchor for the first incremental update so trim-dropped bars get
        # processed instead of skipped.
        self.numba_cls, self.order_tag1, self.last_processed_ts = signal_generator.generate_signal()
        self.numba_cls: TradeOutputDirectionalMomentumLastOutputCls

        agg_time = self.tup.agg_time

        # Warm-up must stop exactly where the batch kernel stopped. `warm_df` arrives
        # UNTRIMMED — prepare_init_data trims df1/warm_df only AFTER the loop that calls us
        # — so it still carries the two minutes generate_signal dropped via `mc.iloc[:-2]`,
        # and live replays those same two minutes. Seeding them here as well double-feeds
        # them, and when `warm_df` ends on a candle boundary a replayed minute belonging to
        # an ALREADY-CLOSED bucket gets folded into the open one — update() only tests
        # whether the bucket has ended, never whether the incoming minute belongs to it —
        # permanently corrupting that candle's high/low and everything downstream of it.
        # Trim so warm-up owns [.., last_processed_ts] and live owns
        # [last_processed_ts + 1min, ..], with no overlap at all.
        warm_df = warm_df.loc[:self.last_processed_ts]

        # ONE resample: the asset's own candle frame. DMP computes every feature on it
        # (median20 / atr_eq / long_vol / rv_alloc / z_median) — there is no ratio, no
        # second leg and no index intersection.
        #
        # CACHE KEY DIFFERS FROM THE PAIR SLEEVES, deliberately. They key on
        # (parent_stratid, frame, agg_time); here the two SIDES of a cell are two objects
        # with two different parent_stratids but identical candle frames, so a pid-keyed
        # cache would resample all 8 cells instead of 4. The resample depends only on
        # (asset, TF), so that is what it is keyed on. coin1 must stay in the key: __main__
        # can build several assets in ONE process, and per-asset LOCAL parent ids would
        # otherwise collide outright.
        agg_key_asset = (self.tup.coin1, 'asset', agg_time)
        min_key_asset = (self.tup.coin1, 'asset', '1T')

        if agg_key_asset in DATA_CACHE:
            asset_cdl = DATA_CACHE[agg_key_asset]
        else:
            asset_cdl = resample_stock_data(warm_df.copy(), agg_time, offset='00:00:00')
            asset_cdl = asset_cdl.loc[~asset_cdl.index.duplicated(), :]
            asset_cdl = asset_cdl[~pd.isna(asset_cdl['open'])]
            DATA_CACHE[agg_key_asset] = asset_cdl

        if min_key_asset in DATA_CACHE:
            one_min_df = DATA_CACHE[min_key_asset]
        else:
            one_min_df = warm_df.copy()
            DATA_CACHE[min_key_asset] = one_min_df

        # The final candle is EITHER still forming OR already closed, and that decides who
        # owns it. The test is NOT "does the frame end on this bucket's last minute" — the
        # trimmed frame can stop short of `last_processed_ts`, because
        # `.loc[:last_processed_ts]` is a label slice and that label need not exist:
        # get_numba_parameters outer-merges `mc` against a COMPLETE tf grid, injecting
        # bucket-start minutes that are absent from warm_df, and `mc.iloc[:-2]` can land on
        # one of those injected labels. (Gaps do reach warm_df on the live path:
        # base.py::get_init_data returns the raw DB frame, not a minute-reindexed one.)
        #
        # The question that actually matters is whether any minute live will replay can
        # still land in this bucket. Live starts at `last_processed_ts + 1min`, so the
        # bucket is closed iff it ENDS at or before `last_processed_ts`. Handing a closed
        # candle to _initialize_ohlcv would drop it from the warm frame and make update()
        # fold the first replayed minute into it and finalize it corrupted.
        bar_still_forming = (asset_cdl.index[-1] + self.OHLCV_AGGREGATOR.timeframe
                             - dt.timedelta(minutes=1)) > self.last_processed_ts

        if bar_still_forming:
            # DELETE last row — it becomes the in-progress bar driven by live ticks, so it
            # must not also be warmed up.
            last_row = asset_cdl.iloc[[-1]]
            asset_cdl = asset_cdl.iloc[:-1]
        else:
            # Candle is COMPLETE: it belongs in the warm-up frame and agg_ohlcv_deque, and
            # live's first minute opens the NEXT bucket from scratch. Handing a finished
            # candle to _initialize_ohlcv instead would make update() fold that first
            # replayed minute into it and finalize it corrupted. Leaving current_ohlcv as
            # None is already a supported state — _finalize_current_ohlcv sets it to None
            # and update() re-initializes on the next minute.
            last_row = None

        # Seed OHLCV aggregator deques with the last 10 minutely + resampled bars.
        for i in range(-10, 0):
            row = one_min_df.iloc[[i]]
            row.index = row.index.tz_localize(None).tz_localize(self.OHLCV_AGGREGATOR.target_tz)
            self.OHLCV_AGGREGATOR._init_data(row, type_='minutely')

        for i in range(-10, 0):
            row = asset_cdl.iloc[[i]]
            row.index = row.index.tz_localize(None).tz_localize(self.OHLCV_AGGREGATOR.target_tz)
            self.OHLCV_AGGREGATOR._init_data(row, type_='resampled')

        if last_row is not None:
            init_ts = last_row.index[-1].tz_localize(None).tz_localize(self.OHLCV_AGGREGATOR.target_tz)
            self.OHLCV_AGGREGATOR._initialize_ohlcv(
                last_row.iloc[-1]['open'], last_row.iloc[-1]['high'], last_row.iloc[-1]['low'],
                last_row.iloc[-1]['close'], last_row.iloc[-1]['volume'], init_ts,
            )

        for tup in inst_status_df.itertuples():
            market_state = MarketState.OPEN if tup.is_trading == True else MarketState.CLOSE
            ts_ = tup.timestamp.astimezone(self.OHLCV_AGGREGATOR.input_tz)
            self.OHLCV_AGGREGATOR.update_market_times(ts_, market_state)

        # Warm-up frame = the asset candle frame itself. Every DMP calc is self-sufficient
        # (it owns its own primitives and consumes only OHLC), so there are no derived
        # columns to attach here — no leg closes, no piped ATR series. That also means the
        # warm-up path cannot silently disagree with the live path: both see exactly the
        # same inputs. It is also why the two SIDES of a cell need no coordination: fed the
        # same candles they hold identical feature state by construction.
        warm_candles = asset_cdl

        # Batch-warm each updater. Updaters tolerate kwargs they don't need.
        for name, cls_ in self.resample_update_func_dict.items():
            msg_ = f"Initializing: {name}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            cls_._initialize(warm_candles)

        for name, cls_ in self.minute_update_func_dict.items():
            msg_ = f"Initializing: {name}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            cls_._initialize(one_min_df)

    def update(self, curr_time, open_: float, high: float, low: float, close: float, volume: int):
        """
        Per-minute live update.
            open_, high, low, close, volume : 1-min OHLCV of THIS ASSET (driver: warm_df /
            live.py's INPUT_DATA_TUPLE). No ratio is constructed anywhere.

        Advances the candle aggregator every minute and, on a completed candle, the five
        feature calcs. Returns the per-bar `DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE`
        snapshot; it does NOT call the kernel — live.py does that, adding the per-minute
        prices it owns (same_close, and the realized fill that goes in the next_close slot).
        """
        market_state = MarketState.OPEN
        run_minutely = self.OHLCV_AGGREGATOR.latest_state == MarketState.OPEN

        ### DON'T CHANGE THE ORDER OF THE BELOW CODE.
        self.OHLCV_AGGREGATOR.update(open_price=open_, high_price=high, low_price=low,
                                     close_price=close, volume=volume, data_timestamp=curr_time)
        self.OHLCV_AGGREGATOR.update_market_times(curr_time, market_state)

        self.is_bar_completed = self.OHLCV_AGGREGATOR.bar_completed
        if self.is_bar_completed:
            last_bar = self.OHLCV_AGGREGATOR.agg_ohlcv_deque[-1]
            msg_ = f"Bar completed: {last_bar}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            open_agg, high_agg, low_agg, close_agg, volume_agg = (
                last_bar.open, last_bar.high, last_bar.low, last_bar.close, last_bar.volume,
            )

            # Plain loop, ANY order. Every DMP calc is self-sufficient — it owns its own
            # primitives and consumes only OHLC — so there are no cross-calc kwargs and no
            # ordering constraint (contrast the lead-lag sleeve, where the ATR family had to
            # run before the Bollinger calc could read atr_price_*).
            for name, cls_ in self.resample_update_func_dict.items():
                msg_ = f"Updating TimeFrame: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time, open_agg, high_agg, low_agg, close_agg)

            self.agg_ohlcv_dict = {"open_agg": open_agg, "high_agg": high_agg, "low_agg": low_agg,
                                   "close_agg": close_agg, "volume_agg": volume_agg}

        if run_minutely:
            for name, cls_ in self.minute_update_func_dict.items():
                msg_ = f"Updating Minutely: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time=curr_time, open_=open_, high=high, low=low, close=close)

        # ---- Build the per-bar DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE (13 fields) -------
        # This tuple carries the candle features, both minute extremes and costs. The
        # kernels' two per-minute price arguments (next_close, same_close) are supplied by
        # live.py from its INPUT_DATA_TUPLE — deliberately not mirrored here.
        min_bar = self.OHLCV_AGGREGATOR.min_ohlcv_deque[-1]
        agg_bar = self.OHLCV_AGGREGATOR.agg_ohlcv_deque[-1]

        # Feature values as of the LAST COMPLETED candle, held between closes. That is
        # exactly what the batch's `.shift(tf - 1)` + ffill produces on the minute grid.
        med_v = float(self.median_calc.median_last)
        atr_v = float(self.atr_eq_calc.atr_eq_last)
        lv_v = float(self.long_vol_calc.long_vol_last)
        rv_v = float(self.rv_alloc_calc.rv_alloc_last)
        zmed_v = float(self.zmed_calc.zmed_last)

        # asset_close = the last COMPLETED candle's close (never the in-progress bar). This
        # is the kernels' `asset_close`, which they compare to the bands and ratchet the
        # peak/trough against — NOT the minute close, which only ever marks the position.
        asset_close_v = float(agg_bar.close)

        # Both Keltner lines. Computed here, as the batch computes them outside the kernel:
        # upper = med + K*atr_eq, lower = med - K*atr_eq. NaN propagates, matching the batch.
        # This object only needs its own side's line, but the tuple is uniform across sides
        # so one dump schema serves the whole sleeve.
        upper_v = med_v + float(self.tup.K) * atr_v
        lower_v = med_v - float(self.tup.K) * atr_v

        # allocation = clip(VT / rv_alloc / annf, 0, 1), times the daily R on the SHORT SIDE
        # ONLY (PRODUCTION directional_momentum.py:83-84 passes `alv * mm` to kern_short and
        # the bare `alv` to kern_long). R is capped at 1.0, so unlike B1 the product can
        # never exceed 1.0 and the clip order is not observable — but the multiply is still
        # applied after the clip, as the reference does it.
        #
        # The LONG branch does not read the artifact at all. A "multiply by a defaulted 1.0"
        # would couple the long side to a nightly file it has no business depending on, and
        # would turn a missing/stale artifact into a silent no-op for half the sleeve.
        #
        # get_r is keyed on `curr_time`, never on wall-clock, so hist_replay walks historical
        # R exactly as live walks today's. MUST stay in step with the batch site in
        # generate_signal_directional_momentum.py::get_numba_parameters.
        vt = float(self.tup.strategy_config['VT'])
        annf = float(self.tup.annf)
        is_short = self.tup.side == "SHORT"
        r_mult = (get_r(self.tup.strategy_config['r_state_path'], curr_time,
                        logger_name=self.logger_name) if is_short else 1.0)
        if np.isnan(rv_v):
            # Sizing vol unknown (warm-up, or too little history). Emit NaN so it cannot
            # masquerade as a real size, but never let it pass unnoticed. NaN * R stays NaN,
            # so no R is applied here by construction.
            allocation_v = float('nan')
            config_object.masterlog.get_logger(self.logger_name, "critical")(
                f"{curr_time}: rv_alloc is NaN for {self.tup.coin1} {self.tup.agg_time} "
                f"{self.tup.side} — allocation emitted as NaN; any entry this bar would "
                f"size to NaN.")
        elif rv_v == 0.0:
            # Batch does VT/0 -> inf -> clip(.,0,1) -> 1.0 then * Rm on numpy arrays.
            # Reproduce that rather than diverge, but a zero sizing vol is pathological.
            allocation_v = 1.0 * r_mult
            config_object.masterlog.get_logger(self.logger_name, "critical")(
                f"{curr_time}: rv_alloc is 0 for {self.tup.coin1} {self.tup.agg_time} "
                f"{self.tup.side} — allocation saturated to 1.0*R={allocation_v:.6f} "
                f"(batch parity).")
        else:
            allocation_v = float(np.clip((vt / rv_v) / annf, 0.0, 1.0)) * r_mult

        p1_v = 1.0 if self.is_bar_completed else 0.0

        self.numba_tup = DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE(
            p1=p1_v,
            minutely_low=float(min_bar.low),
            minutely_high=float(min_bar.high),
            asset_close=asset_close_v,
            median_line=med_v,
            upper_line=upper_v,
            lower_line=lower_v,
            atr_eq=atr_v,
            long_vol=lv_v,
            z_median=zmed_v,
            txn_cost=float(self.tup.strategy_config['txn_cost']),
            slippage=float(self.tup.strategy_config['slippage_per_turn']),
            allocation=allocation_v,
        )
        return self.numba_tup

    def update_numba_tup(self, numba_tup):
        """Override the per-bar numba input tuple (used by the orchestrator)."""
        self.numba_tup = numba_tup

    @property
    def get_ohlcv_log_dict(self):
        ret_ = self.agg_ohlcv_dict
        self.agg_ohlcv_dict = {x: None for x in self.agg_ohlcv_dict.keys()}
        return ret_

    def get_logging_dict(self, base_logging_dict):
        for name, cls_ in self.resample_update_func_dict.items():
            cls_log_dict = cls_.get_logging_dict
            for key, val in cls_log_dict.items():
                base_logging_dict.setdefault(key, []).append(val)

        for name, cls_ in self.minute_update_func_dict.items():
            cls_minute_log_dict = cls_.get_logging_dict
            for key, val in cls_minute_log_dict.items():
                base_logging_dict.setdefault(key, []).append(val)

        ohlcv_dict = self.get_ohlcv_log_dict
        for key, val in ohlcv_dict.items():
            base_logging_dict.setdefault(key, []).append(val)

        ## Per-bar kernel INPUTS (`numba_input.` prefix) and carried kernel STATE
        ## (`numba_cls.` prefix, including res_lo — the signal itself). Same convention as
        ## the sibling sleeves. Without these the dumped CSV would carry only the feature
        ## calcs, and nothing downstream of them could be matched against the batch.
        numba_input_tup = getattr(self, "numba_tup", None)
        if numba_input_tup is not None:
            for key, val in numba_input_tup._asdict().items():
                base_logging_dict.setdefault(f"numba_input.{key}", []).append(val)

        numba_cls = getattr(self, "numba_cls", None)
        if numba_cls is not None:
            for key, val in numba_cls.__dict__.items():
                if key.startswith("__"):
                    continue
                base_logging_dict.setdefault(f"numba_cls.{key}", []).append(val)

        ## `side` is logged so a dump can be read without joining back to the parameter
        ## dict — the two ids of a cell differ in nothing else that appears in the CSV.
        base_logging_dict.setdefault("side", []).append(self.tup.side)
        base_logging_dict.setdefault("order_tag1", []).append(getattr(self, "order_tag1", None))

    def clear_data_cache(self):
        DATA_CACHE.clear()
        import gc; gc.collect()

    def update_order_tag1(self, order_tag1):
        self.order_tag1 = order_tag1

    ## NOTE: no update_order_tag2. The pair sleeves carry one because QR1_v4 has a genuine
    ## SECOND STREAM INSIDE a single model (the scale-in tranche), and pair_momentum /
    ## pair_lead_lag keep the setter only for schema stability. DMP's second stream is a
    ## second MODEL — its own object, its own parent_stratid, its own order_tag1 — so a
    ## second tag here would have nothing to hold.
