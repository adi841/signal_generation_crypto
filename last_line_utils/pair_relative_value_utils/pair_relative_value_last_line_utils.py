
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
from utils.pair_relative_value_utils import TradeOutputPairRelativeValueLastOutputCls
from shared_codes.utils.misc_utils import check_nan

from .pair_relative_value_utils import (
    PairRelativeValueBaseClass,
    PAIR_RELATIVE_VALUE_PARAM_TUPLE,
    PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE,
)
from .generate_signal_pair_relative_value import (
    GetPairsRelativeValueSignal,
    load_qr1_cell_constants,
    load_entry_score_model,
    load_expo_daily,
    score_entry_quality_scalar,
)

## QR1_v4 streaming feature classes — five calcs, all self-sufficient (each owns its
## primitives, no cross-calc kwargs, ANY driver order):
##   QR1BandCalc  — RATIO candles: EMA10 adjust=True centre + IS-clipped Wilder
##                  ATR14/ATR50, min/max bands, kernel-% ATRs (price round trip),
##                  band-width %; stashes its own previous lines (p1 convention).
##   ZnCalc       — RATIO candles: the v4 entry+exit z (adjust=False EMAs over
##                  price-unit adjust=True EWM vol).
##   Ev10Calc     — RATIO candles: EV10 sizing vol. CANDLE-FRESH by construction —
##                  advanced before the minute's allocation is assembled.
##   SkGateCalc   — LEG candle closes: short-flipped skew gate (sk(leg2)-sk(leg1),
##                  shift-1-bar emission; NaN returns ZEROED per pandas .where).
##   CorrCalc     — LEG candle closes: mean rolling Pearson (DEAD gate, parity only).
## Unlike B1, QR1 is NOT ratio-frame-only: the leg closes ride update() kwargs and
## the warm frame's leg1_close/leg2_close columns. Per-cell IS constants (kv / wmed /
## ATR clip bounds) are LOADED from data/qr1_cell_constants.parquet and raise if the
## row is absent — never run with unclipped ATRs or an unscaled sizing vol.
from .features.band_utils import QR1BandCalc
from .features.zn_utils import ZnCalc
from .features.vol_utils import Ev10Calc
from .features.corr_utils import CorrCalc, SkGateCalc

import line_profiler
import atexit

#
import os

# Use a shared global profiler instead of creating a new one
# try:
#     profile  # type: ignore  # noqa: F821
# except NameError:
#     file_name = os.path.basename(__file__)
#     profile = line_profiler.LineProfiler()
#     atexit.register(profile.print_stats)
#     atexit.register(profile.dump_stats, file_name.replace("py", "txt"))


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
class PairRelativeValueOHLCV_Aggregator(PairRelativeValueBaseClass):
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
class PAIR_RELATIVE_VALUE_PARENT_ID_STRAT_OBJ(PairRelativeValueBaseClass):
    """
    Per-(pair, TF) streaming strategy object for QR1_v4 Pairs Relative Value. Owns:
      - The ratio-frame OHLCV aggregator (1-min -> candle TF).
      - The five streaming feature calcs (bands / zn / EV10 / skew gate / corr) and
        per-minute emission of PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE for the _ll
        kernel. QR1 needs per-LEG candle closes for the skew gate — unlike B1 this
        sleeve is not ratio-frame-only; live.py passes the leg minute closes into
        update(), the completed bin's last value feeds the leg calcs.
      - The minute-level score state (tsm middle-touch counter, seeded from the
        batch in initialize(); dist_bp is stateless) — the entry-quality score and
        the EV10 vol-target compose into the per-minute allocation in update().

    NOTE: the EXPO overlay is a cross-market daily state a single-pair process cannot
    compute — it arrives from a shared daily artifact and is wired in the sizing step
    (same policy as pair_momentum's R: no silent default to 1.0).

    Migration state: __init__ / initialize() / features / update() are wired; the
    remaining un-migrated piece of this sleeve is the _ll kernel + live.py step.
    """

    def __init__(self, tup: PAIR_RELATIVE_VALUE_PARAM_TUPLE, logger_name: str):
        self.tup: PAIR_RELATIVE_VALUE_PARAM_TUPLE = tup
        PairRelativeValueBaseClass.logger_name = logger_name

        # Feature registries. The driver fans each completed candle into
        # resample_update_func_dict and each minute into minute_update_func_dict.
        self.minute_update_func_dict = {}
        self.resample_update_func_dict = {}
        self.agg_ohlcv_dict = {"open_agg": None, "high_agg": None, "low_agg": None,
                               "close_agg": None, "volume_agg": None}

        agg_int = int(tup.agg_time.split("T")[0])
        self.OHLCV_AGGREGATOR = PairRelativeValueOHLCV_Aggregator(tup, agg_int, offset=config_object.offset)

        sc = tup.strategy_config

        # IS-frozen per-cell constants + the frozen entry-score model. Raises if the
        # artifact is absent — never run with unclipped ATRs / an unscaled EV10.
        kv, wmed, a14_lo, a14_hi, a50_lo, a50_hi = load_qr1_cell_constants(
            tup.coin1, tup.coin2, tup.tf, sc['data_dir'])
        self.kv = float(kv)
        self.wmed = float(wmed)
        self.entry_score_model = load_entry_score_model(sc['entry_score_model_path'])

        # Daily EXPO overlay: load at init (raises if the artifact is absent — never
        # run with an unsized overlay); the per-day value refreshes in update() on
        # each UTC date change (the EOD job rewrites the artifact nightly).
        self._expo_series = load_expo_daily(sc['data_dir'])
        self._expo_date = None
        self._expo_val = 1.0

        # ---- streaming feature calcs (candle frame; every span/window off the
        # frozen snapshot via strategy_config — nothing hardcoded). Keyed by plain
        # strings (the sa_mft/momentum convention, not the FEATURE_NAME enum).
        self.band_calc = QR1BandCalc(float(tup.nbdev), int(sc['band_centre_ema_span']),
                                     int(sc['atr_band_fast_len']), int(sc['atr_band_slow_len']),
                                     a14_lo, a14_hi, a50_lo, a50_hi)
        self.resample_update_func_dict['bands'] = self.band_calc

        self.zn_calc = ZnCalc(int(sc['zn_centre_ema_span']), int(sc['zn_vol_ewm_span']),
                              int(sc['zn_smooth_ewm_span']), float(sc['zn_denom_floor']))
        self.resample_update_func_dict['zn'] = self.zn_calc

        self.ev10_calc = Ev10Calc(int(sc['sizing_vol_ewm_span']))
        self.resample_update_func_dict['ev10'] = self.ev10_calc

        self.skew_calc = SkGateCalc(int(sc['SKEW_N']))
        self.resample_update_func_dict['sk_gate'] = self.skew_calc

        self.corr_calc = CorrCalc(tuple(sc['corr_lookbacks']))
        self.resample_update_func_dict['corr'] = self.corr_calc

        # minute registry stays EMPTY: the only minute-level quantities (dist_bp and
        # the tsm touch counter for the score) are stateless-or-tiny and maintained
        # inline in update() with the p1 active-line convention.

        # live score-state trackers (tsm is seeded from the batch in initialize()).
        self._tsm = None
        self._cur_leg1_close = float('nan')
        self._cur_leg2_close = float('nan')

        self.is_bar_completed = False

    def initialize(self, df1, df2, comb_df, inst_status_df, curr_time, process_name,
                   signal_generator: GetPairsRelativeValueSignal):
        """
        Historical warm-up. Drives `signal_generator.generate_signal()` once to
        seed `numba_cls` / order tags, then batch-feeds the candle-frame to every
        feature updater so live `update()` calls can resume from the last bar.
        """
        msg_ = f"Initializing the strategy: {self.tup}"
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        # Drive the historical array kernel and pull BOTH tranche tags — QR1 is a
        # DUAL-tranche stream (res -> order_tag1, scale-in res2 -> order_tag2), so
        # generate_signal returns a 4-tuple, unlike B1's 3.
        # last_processed_ts = the last bar the batch kernel actually processed
        # (= one bar before the trim-2 cliff). live.py uses it as the start
        # anchor for the first incremental update so trim-dropped bars get
        # processed instead of skipped.
        self.numba_cls, self.order_tag1, self.order_tag2, self.last_processed_ts = signal_generator.generate_signal()
        self.numba_cls: TradeOutputPairRelativeValueLastOutputCls

        # Seed the live tsm counter (the score's tsm_min feature) from the batch so
        # it continues seamlessly across the batch->live handoff (None = the batch
        # never observed a middle touch).
        self._tsm = getattr(signal_generator, '_tsm_seed', None)

        agg_time = self.tup.agg_time

        # Warm-up must stop exactly where the batch kernel stopped. The frames arrive
        # UNTRIMMED — prepare_init_data trims df1/df2/comb_df only AFTER the loop that
        # calls us — so they still carry the two minutes generate_signal dropped via
        # `mc.iloc[:-2]`, and live replays those same two minutes. Seeding them here as
        # well double-feeds them, and when the data ends on a candle boundary a
        # replayed minute belonging to an ALREADY-CLOSED bucket gets folded into the
        # open one — update() only tests whether the bucket has ended, never whether
        # the incoming minute belongs to it — permanently corrupting that candle's
        # high/low and everything downstream (incl. the candle-fresh EV10). Trim so
        # warm-up owns [.., last_processed_ts] and live owns
        # [last_processed_ts + 1min, ..], with no overlap at all. The leg trims keep
        # the .resample().last() feeds on the same invariant.
        comb_df = comb_df.loc[:self.last_processed_ts]
        df1 = df1.loc[:self.last_processed_ts]
        df2 = df2.loc[:self.last_processed_ts]

        # ---- warm frames via the shared per-process cache ---------------------------
        # THREE frames: the ratio candle frame (bands / zn / EV10 / band-width all live
        # on ratio candles) plus the two RAW leg candle-close series — QR1's skew gate
        # (and the dead corr) are LEG-level, so unlike B1 this sleeve is not
        # ratio-frame-only. Leg closes are built exactly as the batch path builds them
        # (df1/df2 are UNFILLED; .resample().last(); NaN preserved for whole-bin gaps;
        # NO cross-frame index intersection) so the features-step warm-up cannot
        # silently disagree with get_numba_parameters.
        agg_key_ratio = (self.tup.parent_stratid, 'ratio', agg_time)
        agg_key_leg1 = (self.tup.parent_stratid, 'leg1_close', agg_time)
        agg_key_leg2 = (self.tup.parent_stratid, 'leg2_close', agg_time)
        min_key_ratio = (self.tup.parent_stratid, 'ratio', '1T')

        if agg_key_ratio in DATA_CACHE:
            ratio_cdl = DATA_CACHE[agg_key_ratio]
        else:
            ratio_cdl = resample_stock_data(comb_df.copy(), agg_time, offset='00:00:00')
            ratio_cdl = ratio_cdl.loc[~ratio_cdl.index.duplicated(), :]
            ratio_cdl = ratio_cdl[~pd.isna(ratio_cdl['open'])]
            DATA_CACHE[agg_key_ratio] = ratio_cdl

        if agg_key_leg1 in DATA_CACHE:
            leg1_close_cdl = DATA_CACHE[agg_key_leg1]
        else:
            leg1_close_cdl = df1['close'].resample(agg_time).last()
            DATA_CACHE[agg_key_leg1] = leg1_close_cdl

        if agg_key_leg2 in DATA_CACHE:
            leg2_close_cdl = DATA_CACHE[agg_key_leg2]
        else:
            leg2_close_cdl = df2['close'].resample(agg_time).last()
            DATA_CACHE[agg_key_leg2] = leg2_close_cdl

        if min_key_ratio in DATA_CACHE:
            one_min_df = DATA_CACHE[min_key_ratio]
        else:
            one_min_df = comb_df.copy()
            DATA_CACHE[min_key_ratio] = one_min_df

        # The final candle is EITHER still forming OR already closed, and that decides
        # who owns it. The test is NOT "does the frame end on this bucket's last
        # minute" — the trimmed frame can stop SHORT of `last_processed_ts`, because
        # `.loc[:last_processed_ts]` is a label slice and that label need not exist:
        # get_numba_parameters outer-merges `mc` against a COMPLETE tf grid, injecting
        # bucket-start minutes that are absent from comb_df, and `mc.iloc[:-2]` can
        # land on one of those injected labels. (Gaps do reach comb_df on the live
        # path: base.py::get_init_data returns the raw DB frame, not a
        # minute-reindexed one.)
        #
        # The question that actually matters is whether any minute live will replay
        # can still land in this bucket. Live starts at `last_processed_ts + 1min`,
        # so the bucket is closed iff it ENDS at or before `last_processed_ts`.
        bar_still_forming = (ratio_cdl.index[-1] + self.OHLCV_AGGREGATOR.timeframe
                             - dt.timedelta(minutes=1)) > self.last_processed_ts

        if bar_still_forming:
            # DELETE last row — it becomes the in-progress bar driven by live ticks,
            # so it must not also be warmed up.
            last_row = ratio_cdl.iloc[[-1]]
            ratio_cdl = ratio_cdl.iloc[:-1]
        else:
            # Candle is COMPLETE: it belongs in the warm-up frame (the calcs must warm
            # THROUGH it — its values landed on row last_processed_ts, which the batch
            # kernel consumed) and in agg_ohlcv_deque; live's first minute opens the
            # NEXT bucket from scratch. Handing a finished candle to _initialize_ohlcv
            # instead would make update() fold that first replayed minute into it and
            # finalize it corrupted. Leaving current_ohlcv as None is a supported
            # state — _finalize_current_ohlcv sets it to None and update()
            # re-initializes on the next minute.
            last_row = None

        # Seed OHLCV aggregator deques with the last 10 minutely + resampled bars.
        for i in range(-10, 0):
            row = one_min_df.iloc[[i]]
            row.index = row.index.tz_localize(None).tz_localize(self.OHLCV_AGGREGATOR.target_tz)
            self.OHLCV_AGGREGATOR._init_data(row, type_='minutely')

        for i in range(-10, 0):
            row = ratio_cdl.iloc[[i]]
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

        # Warm-up frame = the (trimmed) ratio candle frame + the raw leg candle closes
        # aligned onto its index. No pre-piped indicator columns — every QR1 calc owns
        # its own primitives (the momentum features convention), so the lead-lag-era
        # atr_price_fast/slow pre-compute is gone.
        # NOTE for the features step: the batch path KEEPS NaN candle bins while this
        # warm frame drops NaN-open rows (momentum-identical) and live always forms
        # bars from ffilled minutes — any gap-region parity burden lands on the
        # features-step calc design (same latent property as the proven momentum
        # sleeve).
        warm_df = ratio_cdl.copy()
        warm_df['leg1_close'] = leg1_close_cdl.reindex(warm_df.index)
        warm_df['leg2_close'] = leg2_close_cdl.reindex(warm_df.index)

        # Batch-warm each updater. Updaters tolerate columns/kwargs they don't need
        # (registries are populated in the features step; empty loops are the wired
        # contract until then).
        for name, cls_ in self.resample_update_func_dict.items():
            msg_ = f"Initializing: {name}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            cls_._initialize(warm_df)

        for name, cls_ in self.minute_update_func_dict.items():
            msg_ = f"Initializing: {name}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            cls_._initialize(one_min_df)

    def update(self, curr_time, open_: float, high: float, low: float, close: float, volume: int,
               leg1_close_min: float = None, leg2_close_min: float = None):
        """
        Per-minute live update.
            open_, high, low, close, volume : 1-min RATIO OHLCV (driver: comb_df).
            leg1_close_min, leg2_close_min  : 1-min closes of leg1 / leg2 (live.py
                                              owns the legs) — the completed bin's
                                              last value feeds the leg-candle
                                              skew/corr calcs on bar completion.

        Advances the candle aggregator every minute and, on a completed candle,
        the five feature calcs — EV10 is CANDLE-FRESH by construction: the calcs
        advance BEFORE this minute's allocation is assembled, so an entry decided
        on a boundary minute is sized off the candle that closed that minute.
        Returns the per-bar `PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE` snapshot; it
        does NOT call the kernel — live.py does that, adding the per-leg prices
        it owns.
        """
        market_state = MarketState.OPEN
        run_minutely = self.OHLCV_AGGREGATOR.latest_state == MarketState.OPEN

        ### DON'T CHANGE THE ORDER OF THE BELOW CODE.
        self.OHLCV_AGGREGATOR.update(open_price=open_, high_price=high, low_price=low,
                                     close_price=close, volume=volume, data_timestamp=curr_time)
        self.OHLCV_AGGREGATOR.update_market_times(curr_time, market_state)

        # Track the current bin's last-seen leg closes: at bar completion these are
        # the bin's last minute closes == the batch's .resample().last() (modulo the
        # ffill commute — live feeds are ffilled upstream).
        if leg1_close_min is not None and not check_nan(leg1_close_min):
            self._cur_leg1_close = float(leg1_close_min)
        if leg2_close_min is not None and not check_nan(leg2_close_min):
            self._cur_leg2_close = float(leg2_close_min)

        self.is_bar_completed = self.OHLCV_AGGREGATOR.bar_completed
        if self.is_bar_completed:
            last_bar = self.OHLCV_AGGREGATOR.agg_ohlcv_deque[-1]
            msg_ = f"Bar completed: {last_bar}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            open_agg, high_agg, low_agg, close_agg, volume_agg = (
                last_bar.open, last_bar.high, last_bar.low, last_bar.close, last_bar.volume,
            )

            # Plain loop, ANY order — every QR1 calc is self-sufficient (band_calc
            # stashes its own previous lines internally for the p1 convention).
            # Leg closes ride kwargs; calcs that don't need them ignore them.
            for name, cls_ in self.resample_update_func_dict.items():
                msg_ = f"Updating TimeFrame: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time, open_agg, high_agg, low_agg, close_agg,
                             leg1_close=self._cur_leg1_close, leg2_close=self._cur_leg2_close)

            self.agg_ohlcv_dict = {"open_agg": open_agg, "high_agg": high_agg, "low_agg": low_agg,
                                   "close_agg": close_agg, "volume_agg": volume_agg}

        if run_minutely:
            for name, cls_ in self.minute_update_func_dict.items():
                msg_ = f"Updating Minutely: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time=curr_time, open_=open_, high=high, low=low, close=close)

        # ---- Build the per-bar PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE (15 fields) ----
        # RATIO-side quantities + costs only; the kernel's per-leg price args come
        # from live.py's INPUT_DATA_TUPLE, deliberately not mirrored here.
        min_bar = self.OHLCV_AGGREGATOR.min_ohlcv_deque[-1]
        p1_v = 1.0 if self.is_bar_completed else 0.0
        mh_v = float(min_bar.high)
        ml_v = float(min_bar.low)

        # Feature values as of the LAST COMPLETED candle — exactly what the batch's
        # shift(tf-1) + ffill projection holds between closes.
        mid_v = float(self.band_calc.s_middle_last)
        up_v = float(self.band_calc.s_upper_last)
        low_v = float(self.band_calc.s_lower_last)
        atr_v = float(self.band_calc.atr_pct_last)
        atr2_v = float(self.band_calc.atr2_pct_last)
        fbw_v = float(self.band_calc.f_bandwp_last)
        ev10_v = float(self.ev10_calc.ev10_last)
        zn_raw = float(self.zn_calc.zn_last)
        corr_raw = float(self.corr_calc.corr_last)
        sk_ok = bool(self.skew_calc.sk_gate_last >= 0.5)

        sc = self.tup.strategy_config
        zn_v = zn_raw if not np.isnan(zn_raw) else float(sc['zn_projection_fillna'])
        corr_v = corr_raw if not np.isnan(corr_raw) else 1.0

        # Active lines for THIS minute's intrabar tests: on the boundary minute the
        # fresh candle was not closed while the minute traded, so the batch/kernel
        # p1 convention reads the PREVIOUS candle's lines.
        if p1_v == 1.0:
            mid_act = float(self.band_calc.prev_s_middle_last)
            up_act = float(self.band_calc.prev_s_upper_last)
        else:
            mid_act = mid_v
            up_act = up_v

        # tsm: minutes since minutely_low last touched the active middle. Seeded
        # from the batch in initialize(); None = never observed a touch.
        if (not np.isnan(mid_act)) and ml_v <= mid_act:
            self._tsm = 0
        elif self._tsm is not None:
            self._tsm += 1

        # Entry gates -> msig (NaN zn was filled above; NaN band-width compares
        # False, matching the batch's array semantics).
        msig_v = 1.0 if ((zn_v > float(self.tup.z_entry)) and sk_ok
                         and (not np.isnan(fbw_v)) and (fbw_v > self.wmed)) else -1.0

        # ---- allocation = EV10 vol-target x entry-quality score  [TODO(EXPO)] -----
        # Score features at the decision minute (active-line convention as above);
        # batch's never-touched tsm sentinel is the array length — any value at or
        # beyond the frozen grid's top bins identically, 1e12 here.
        dist_bp = (mh_v / up_act - 1.0) * 1e4 if (not np.isnan(up_act)) and up_act != 0.0 else float('nan')
        tsm_v = float(self._tsm) if self._tsm is not None else 1e12
        score = score_entry_quality_scalar(self.entry_score_model, self.tup.tf,
                                           fbw_v, dist_bp, tsm_v)
        med = float(self.entry_score_model['med_entry_score'])
        sc_lo, sc_hi = float(sc['SC_CLIP'][0]), float(sc['SC_CLIP'][1])
        score_mult = min(max(score / med, sc_lo), sc_hi)

        # ---- daily EXPO overlay (entry sizing only) ------------------------------
        # Refresh on each UTC date change: re-read the artifact (the EOD job rewrites
        # it nightly) and look up today's value. A date missing from the artifact
        # resolves to the reference's fillna(1.0) semantics, but with a CRITICAL log —
        # a stale artifact must be loud, never silent.
        d0 = pd.Timestamp(curr_time).normalize()
        d0n = d0.tz_localize(None) if d0.tz is not None else d0
        if d0n != self._expo_date:
            try:
                self._expo_series = load_expo_daily(sc['data_dir'])
            except Exception as e:
                config_object.masterlog.get_logger(self.logger_name, "critical")(
                    f"{curr_time}: expo_daily reload failed ({e}) — keeping the "
                    f"previously loaded series.")
            self._expo_date = d0n
            if d0n in self._expo_series.index:
                self._expo_val = float(self._expo_series.loc[d0n])
            else:
                self._expo_val = 1.0
                config_object.masterlog.get_logger(self.logger_name, "critical")(
                    f"{curr_time}: expo_daily has no row for {d0n.date()} — using 1.0 "
                    f"(reference fillna semantics); is the EOD job stale?")

        if np.isnan(ev10_v):
            # Sizing vol unknown (warm-up, or too little history). Emit NaN so it
            # cannot masquerade as a real size, but never let it pass unnoticed.
            allocation_v = float('nan')
            config_object.masterlog.get_logger(self.logger_name, "critical")(
                f"{curr_time}: EV10 is NaN for {self.tup.coin1}/{self.tup.coin2} "
                f"{self.tup.agg_time} — allocation emitted as NaN; any entry this "
                f"bar would size to NaN.")
        else:
            denom = ev10_v * self.kv
            if denom < 1e-9:
                denom = 1e-9          # batch: (ev10 * kv).clip(1e-9)
            al_ew = min(max((float(sc['VT']) / denom) / float(self.tup.annf), 0.0), 1.0)
            # (al_ew * score) * expo — the batch/reference multiplication order.
            allocation_v = al_ew * score_mult * self._expo_val

        self.numba_tup = PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE(
            p1=p1_v,
            minutely_high=mh_v,
            minutely_low=ml_v,
            msig=msig_v,
            upper=up_v,
            middle=mid_v,
            lower=low_v,
            atr=atr_v,
            atr2=atr2_v,
            zn=zn_v,
            corr=corr_v,
            skew_ok=sk_ok,
            txn_cost=float(sc['txn_cost']),
            slippage=float(sc['slippage_per_leg_per_turn']),
            allocation=allocation_v,
        )
        return self.numba_tup

    @property
    def prev_middle_line(self):
        """The channel centre as of the candle BEFORE the most recent one — the
        line the kernel's p1 convention reads on a boundary minute. Only consumed
        by live.py when p1 == 1 (stale between completions, harmlessly)."""
        return float(self.band_calc.prev_s_middle_last)

    @property
    def prev_upper_line(self):
        return float(self.band_calc.prev_s_upper_last)

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

        base_logging_dict.setdefault("order_tag1", []).append(getattr(self, "order_tag1", None))
        base_logging_dict.setdefault("order_tag2", []).append(getattr(self, "order_tag2", None))

    def clear_data_cache(self):
        DATA_CACHE.clear()
        import gc; gc.collect()

    def update_order_tag1(self, order_tag1):
        self.order_tag1 = order_tag1

    def update_order_tag2(self, order_tag2):
        self.order_tag2 = order_tag2