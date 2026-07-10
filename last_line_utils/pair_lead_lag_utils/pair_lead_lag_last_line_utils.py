
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
from utils.pair_lead_lag_utils import TradeOutputPairLeadLagLastOutputCls
from shared_codes.utils.misc_utils import check_nan

from .pair_lead_lag_utils import PairLeadLagBaseClass, PAIR_LEAD_LAG_PARAM_TUPLE, PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE
from .generate_signal_pair_lead_lag import GetPairsLeadLagSignal, load_qr31_cell_constants
from .mult_state import get_mult

## QR31_v2 streaming feature classes — five calcs on the RATIO candle frame, plus the
## corr calc which additionally consumes the two LEGS' candle closes (passed as kwargs;
## every other calc is self-sufficient, so the only ordering constraint is none at all —
## the driver loop may run them in any order).
from .features.atr_utils import ATRPctClippedCalc
from .features.z_ema_utils import ZEMACalc
from .features.corr_utils import CorrCalc
from .features.sizing_vol_utils import SizingVolCalc

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
class PairLeadLagOHLCV_Aggregator(PairLeadLagBaseClass):
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
class PAIR_LEAD_LAG_PARENT_ID_STRAT_OBJ(PairLeadLagBaseClass):
    """
    Per-(pair, TF) streaming strategy object for QR31_v2 Pairs Lead-Lag. Owns:
      - The ratio-frame OHLCV aggregator (1-min -> candle TF).
      - (LATER STEPS) the streaming feature updaters (EMA8 centre / clipped ATR14+ATR50
        / z_ema chain / 4-window leg corr / scaled+clipped sizing vol) and per-bar
        emission of `PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE` for the _ll kernel.

    NOTE: entry sizing carries a cross-pair daily R x I multiplier, which a single-pair
    process cannot compute — it arrives from a shared daily artifact
    (strategy_config['mult_daily_path']) and is wired in the sizing step, batch and
    streaming simultaneously.

    Fully wired for QR31_v2: `initialize` (batch warm-up: generate_signal seeds
    numba_cls / order tag / last_processed_ts; trim + conditional candle handover),
    the five streaming feature calcs, and the per-minute `update()` that emits the
    15-field numba input tuple. live.py owns the legs and the kernel call.
    """

    def __init__(self, tup: PAIR_LEAD_LAG_PARAM_TUPLE, logger_name: str):
        self.tup: PAIR_LEAD_LAG_PARAM_TUPLE = tup
        PairLeadLagBaseClass.logger_name = logger_name

        self.minute_update_func_dict = {}
        self.resample_update_func_dict = {}
        self.agg_ohlcv_dict = {"open_agg": None, "high_agg": None, "low_agg": None,
                               "close_agg": None, "volume_agg": None}

        agg_int = int(tup.agg_time.split("T")[0])
        self.OHLCV_AGGREGATOR = PairLeadLagOHLCV_Aggregator(tup, agg_int, offset=config_object.offset)

        # Last-known per-leg minutely close (fed in by the live layer via `update`
        # kwargs). At bar completion the stashed value IS the leg's candle close —
        # the last minutely close inside the bucket, matching the batch's ffilled
        # `.resample(tf).last()` — and is what the corr calc consumes.
        self._last_leg1_close_min = float('nan')
        self._last_leg2_close_min = float('nan')

        # ---- streaming feature calcs (candle frame) --------------------------------
        # Every span/window comes off the frozen snapshot via strategy_config; the
        # per-cell IS-frozen constants (ATR clip bounds, sizing scale + bounds) come
        # from the qr31_cell_constants artifact and FAIL LOUDLY if absent — never run
        # with unclipped ATRs or an unscaled sizing vol.
        sc = tup.strategy_config
        (atr14_lo, atr14_hi, atr50_lo, atr50_hi,
         sizing_scale, sizing_lo, sizing_hi) = load_qr31_cell_constants(
            tup.coin1, tup.coin2, tup.tf, sc['qr31_cell_constants_path'])

        self.atr14_calc = ATRPctClippedCalc(int(sc['atr_band_fast_len']), atr14_lo, atr14_hi)
        self.resample_update_func_dict['atr14'] = self.atr14_calc

        self.atr50_calc = ATRPctClippedCalc(int(sc['atr_band_slow_len']), atr50_lo, atr50_hi)
        self.resample_update_func_dict['atr50'] = self.atr50_calc

        # ZEMACalc's internal EMA doubles as the band middle (`ema_last`) — the frozen
        # v2 spec makes centre and z centre ONE representation object (EMA_SPAN=8).
        self.z_ema_calc = ZEMACalc(int(sc['EMA_SPAN']), int(sc['Z_ATR_LEN']), int(sc['Z_EMA_LEN']))
        self.resample_update_func_dict['z_ema'] = self.z_ema_calc

        self.corr_calc = CorrCalc(tuple(int(x) for x in sc['corr_lookbacks']))
        self.resample_update_func_dict['corr'] = self.corr_calc

        self.sizing_vol_calc = SizingVolCalc(int(sc['sizing_vol_ewm_span']),
                                             sizing_scale, sizing_lo, sizing_hi)
        self.resample_update_func_dict['sizing_vol'] = self.sizing_vol_calc

        # Previous-minute line snapshots for the kernel's [i-1] reads on p1 minutes
        # (entry middle test, virtual-second trigger atr). NaN until the first live
        # completion; on non-p1 minutes prev == current by construction.
        self._middle_prev = float('nan')
        self._atr14_pct_prev = float('nan')

        self.is_bar_completed = False

    def initialize(self, df1, df2, comb_df, inst_status_df, curr_time, process_name, signal_generator: GetPairsLeadLagSignal):
        """
        Historical warm-up. Drives `signal_generator.generate_signal()` once to
        seed `numba_cls` / order tags, then batch-feeds the candle-frame to every
        feature updater so live `update()` calls can resume from the last bar.
        """
        msg_ = f"Initializing the strategy: {self.tup}"
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        # Drive the historical array kernel and pull the order tag. QR31_v2 is a
        # SINGLE long-only stream: at the frozen flags (sec_mode=2) the second entry
        # is VIRTUAL — it books no position, so res2 is identically 0 and there is no
        # order_tag2 (the reversal2 dual-stream contract does not apply).
        # last_processed_ts = the last bar batch's kernel actually processed
        # (= one bar before the trim-2 cliff). live.py uses it as the start
        # anchor for the first incremental update so trim-dropped bars get
        # processed instead of skipped.
        self.numba_cls, self.order_tag1, self.last_processed_ts = signal_generator.generate_signal()
        self.numba_cls: TradeOutputPairLeadLagLastOutputCls

        agg_time = self.tup.agg_time

        # Warm-up must stop exactly where the batch kernel stopped. The frames arrive
        # UNTRIMMED — prepare_init_data trims df1/df2/comb_df only AFTER the loop that
        # calls us (pair_lead_lag_signal_generator.py) — so they still carry the two
        # minutes generate_signal dropped via `mc.iloc[:-2]`, and live replays those
        # same two minutes. Seeding them here as well double-feeds them, and when the
        # frame ends on a candle boundary a replayed minute belonging to an
        # ALREADY-CLOSED bucket would get folded into the open one — the aggregator's
        # update() only tests whether the bucket has ended, never whether the incoming
        # minute belongs to it — permanently corrupting that candle's high/low and
        # everything downstream. Trim so warm-up owns [.., last_processed_ts] and live
        # owns [last_processed_ts + 1min, ..], with no overlap at all. The legs are
        # trimmed identically — they feed the corr warm-up, which must cover exactly
        # the batch's completed candles.
        comb_df = comb_df.loc[:self.last_processed_ts]
        df1 = df1.loc[:self.last_processed_ts]
        df2 = df2.loc[:self.last_processed_ts]

        # Build candle-frame dfs for ratio, leg1, leg2 — keyed by parent_stratid+agg_time
        # so concurrent buckets sharing the same TF reuse the resample.
        agg_key_ratio = (self.tup.parent_stratid, 'ratio', agg_time)
        agg_key_leg1  = (self.tup.parent_stratid, 'leg1',  agg_time)
        agg_key_leg2  = (self.tup.parent_stratid, 'leg2',  agg_time)
        min_key_ratio = (self.tup.parent_stratid, 'ratio', '1T')

        if agg_key_ratio in DATA_CACHE:
            ratio_cdl = DATA_CACHE[agg_key_ratio]
        else:
            ratio_cdl = resample_stock_data(comb_df.copy(), agg_time, offset='00:00:00')
            ratio_cdl = ratio_cdl.loc[~ratio_cdl.index.duplicated(), :]
            ratio_cdl = ratio_cdl[~pd.isna(ratio_cdl['open'])]
            DATA_CACHE[agg_key_ratio] = ratio_cdl

        if agg_key_leg1 in DATA_CACHE:
            leg1_cdl = DATA_CACHE[agg_key_leg1]
        else:
            leg1_cdl = resample_stock_data(df1.copy(), agg_time, offset='00:00:00')
            leg1_cdl = leg1_cdl.loc[~leg1_cdl.index.duplicated(), :]
            leg1_cdl = leg1_cdl[~pd.isna(leg1_cdl['open'])]
            DATA_CACHE[agg_key_leg1] = leg1_cdl

        if agg_key_leg2 in DATA_CACHE:
            leg2_cdl = DATA_CACHE[agg_key_leg2]
        else:
            leg2_cdl = resample_stock_data(df2.copy(), agg_time, offset='00:00:00')
            leg2_cdl = leg2_cdl.loc[~leg2_cdl.index.duplicated(), :]
            leg2_cdl = leg2_cdl[~pd.isna(leg2_cdl['open'])]
            DATA_CACHE[agg_key_leg2] = leg2_cdl

        if min_key_ratio in DATA_CACHE:
            one_min_df = DATA_CACHE[min_key_ratio]
        else:
            one_min_df = comb_df.copy()
            DATA_CACHE[min_key_ratio] = one_min_df

        # Align candle-frame indices across the three resamples (drop bars missing
        # from any leg) so the corr _initialize sees aligned leg closes.
        common_ts = ratio_cdl.index.intersection(leg1_cdl.index).intersection(leg2_cdl.index)
        ratio_cdl = ratio_cdl.loc[common_ts]
        leg1_cdl = leg1_cdl.loc[common_ts]
        leg2_cdl = leg2_cdl.loc[common_ts]

        # The final candle is EITHER still forming OR already closed, and that decides
        # who owns it. The test is NOT "does the frame end on this bucket's last
        # minute" — the trimmed frame can stop short of `last_processed_ts`, because
        # `.loc[:last_processed_ts]` is a label slice and that label need not exist:
        # get_numba_parameters outer-merges `mc` against a COMPLETE tf grid, injecting
        # bucket-start minutes that are absent from comb_df, and `mc.iloc[:-2]` can
        # land on one of those injected labels. (Gaps do reach comb_df on the live
        # path: base.py::get_init_data returns the raw DB frame, not the
        # minute-reindexed one.)
        #
        # The question that actually matters is whether any minute live will replay
        # can still land in this bucket. Live starts at `last_processed_ts + 1min`, so
        # the bucket is closed iff it ENDS at or before `last_processed_ts`. Handing a
        # closed candle to _initialize_ohlcv would drop it from warm_df and make the
        # aggregator fold the first replayed minute into it and finalize it corrupted.
        bar_still_forming = (ratio_cdl.index[-1] + self.OHLCV_AGGREGATOR.timeframe
                             - dt.timedelta(minutes=1)) > self.last_processed_ts

        if bar_still_forming:
            # DELETE last row — it becomes the in-progress bar driven by live ticks, so
            # it must not also be warmed up. The legs drop the same row: the corr calc
            # must only ever see COMPLETED candles.
            last_row = ratio_cdl.iloc[[-1]]
            ratio_cdl = ratio_cdl.iloc[:-1]
            leg1_cdl_warm = leg1_cdl.iloc[:-1]
            leg2_cdl_warm = leg2_cdl.iloc[:-1]
        else:
            # Candle is COMPLETE: it belongs in the warm-up frame and agg_ohlcv_deque,
            # and live's first minute opens the NEXT bucket from scratch. Handing a
            # finished candle to _initialize_ohlcv instead would make the aggregator
            # fold that first replayed minute into it and finalize it corrupted.
            # Leaving current_ohlcv as None is already a supported state —
            # _finalize_current_ohlcv sets it to None and update() re-initializes on
            # the next minute.
            last_row = None
            leg1_cdl_warm = leg1_cdl
            leg2_cdl_warm = leg2_cdl

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

        # Build the warm-up df with leg1/leg2 close columns (the corr calc needs the
        # two legs' CANDLE closes; every other QR31_v2 calc is self-sufficient — it
        # owns its own primitives and consumes only the ratio OHLC. The old piped-ATR
        # columns served the retired v16_v2 Bollinger calc and are gone.)
        warm_df = ratio_cdl.copy()
        warm_df['leg1_close'] = leg1_cdl_warm['close'].values
        warm_df['leg2_close'] = leg2_cdl_warm['close'].values

        # Batch-warm each updater. Updaters tolerate kwargs they don't need
        # (e.g. the ATR calcs ignore leg1_close/leg2_close).
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
            open_, high, low, close, volume : 1-min RATIO OHLCV (driver: comb_df;
                                              high/low = max/min(open, close)).
            leg1_close_min, leg2_close_min  : 1-min FFILLED closes of leg1 / leg2 —
                                              the corr calc's candle closes are the
                                              stashed values at bar completion.

        Advances the candle aggregator every minute and, on a completed candle, the
        five ratio/leg feature calcs. Returns the per-bar
        `PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE` snapshot; it does NOT call the kernel —
        live.py does that, adding the per-leg prices and the carried state it owns.
        """
        market_state = MarketState.OPEN
        run_minutely = self.OHLCV_AGGREGATOR.latest_state == MarketState.OPEN

        ### DON'T CHANGE THE ORDER OF THE BELOW CODE.
        self.OHLCV_AGGREGATOR.update(open_price=open_, high_price=high, low_price=low,
                                     close_price=close, volume=volume, data_timestamp=curr_time)
        self.OHLCV_AGGREGATOR.update_market_times(curr_time, market_state)

        # Stash the most recent per-leg close — at bar completion this is the leg's
        # candle close (last minutely close inside the bucket == the batch's ffilled
        # `.resample(tf).last()`), which the corr calc consumes.
        if leg1_close_min is not None:
            self._last_leg1_close_min = float(leg1_close_min)
        if leg2_close_min is not None:
            self._last_leg2_close_min = float(leg2_close_min)

        self.is_bar_completed = self.OHLCV_AGGREGATOR.bar_completed
        if self.is_bar_completed:
            last_bar = self.OHLCV_AGGREGATOR.agg_ohlcv_deque[-1]
            msg_ = f"Bar completed: {last_bar}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            open_agg, high_agg, low_agg, close_agg, volume_agg = (
                last_bar.open, last_bar.high, last_bar.low, last_bar.close, last_bar.volume,
            )

            # Snapshot the PREVIOUS-minute line values BEFORE the calcs advance: on
            # this (p1) minute the kernel's [i-1] reads must see the candle-BEFORE-
            # this-close values (entry middle test, virtual-second trigger atr).
            self._middle_prev = float(self.z_ema_calc.ema_last)
            self._atr14_pct_prev = float(self.atr14_calc.atr_pct_last)

            # Plain loop, ANY order. Only corr needs kwargs (the leg candle closes);
            # every calc tolerates kwargs it does not consume.
            for name, cls_ in self.resample_update_func_dict.items():
                msg_ = f"Updating TimeFrame: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time, open_agg, high_agg, low_agg, close_agg,
                             leg1_close=self._last_leg1_close_min,
                             leg2_close=self._last_leg2_close_min)

            self.agg_ohlcv_dict = {"open_agg": open_agg, "high_agg": high_agg, "low_agg": low_agg,
                                   "close_agg": close_agg, "volume_agg": volume_agg}

        if run_minutely:
            for name, cls_ in self.minute_update_func_dict.items():
                msg_ = f"Updating Minutely: {name}"
                config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                cls_._update(curr_time=curr_time, open_=open_, high=high, low=low, close=close)

        # ---- Build the per-bar PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE (15 fields) ----------
        # Feature values as of the LAST COMPLETED candle, held between closes — exactly
        # what the batch's `.shift(tf - 1)` + ffill produces on the minute grid. On the
        # completion minute itself the *_prev fields carry the candle-before values;
        # on every other minute prev == current (the projected array is flat there).
        min_bar = self.OHLCV_AGGREGATOR.min_ohlcv_deque[-1]

        sc = self.tup.strategy_config
        mid_v = float(self.z_ema_calc.ema_last)
        atr14_v = float(self.atr14_calc.atr_pct_last)
        atr50_v = float(self.atr50_calc.atr_pct_last)
        z_v = float(self.z_ema_calc.z_ema_last)
        corr_v = float(self.corr_calc.corr_last)
        sizing_v = float(self.sizing_vol_calc.sizing_vol_last)

        p1_v = 1.0 if self.is_bar_completed else 0.0
        mid_prev_v = self._middle_prev if self.is_bar_completed else mid_v
        atr14_prev_v = self._atr14_pct_prev if self.is_bar_completed else atr14_v

        # RAW upper band, price units: min over the fast/slow clipped ATRs. NaN
        # propagates (batch parity — no band until the ATRs exist). The kernel's
        # POST-RATCHET effective band is trade state (numba_cls.upper_eff_prev_lo).
        nbdev = float(self.tup.nbdev)
        upper_v = min(mid_v + nbdev * self.atr14_calc.atr_price_last,
                      mid_v + nbdev * self.atr50_calc.atr_price_last)

        # Batch post-projection transforms (generate_signal parity).
        z_in = float(sc['z_projection_fillna']) if np.isnan(z_v) else z_v
        corr_in = float(sc['corr_fillna']) if np.isnan(corr_v) else corr_v
        m1_v = 1.0 if z_in > float(sc['entry_z_gate']) else -1.0

        # allocation = clip( clip((VT/annf)/sizing_vol, 0, 1) * mult, 0, 2 ).
        # mult is the daily cross-market R x I state, read from the nightly artifact
        # (mult_state.get_mult — the QR31 law: exact day or 1.0, NO ffill, loud when
        # stale). Keyed on `curr_time`, never wall-clock, so hist_replay walks
        # historical mult exactly as live walks today's. MUST stay in step with the
        # batch site in generate_signal_pair_lead_lag.py.
        vt = float(sc['VT'])
        annf = float(self.tup.annf)
        clip_lo, clip_hi = (float(x) for x in sc['alloc_outer_clip'])
        mult_v = get_mult(sc['mult_daily_path'], curr_time, logger_name=self.logger_name)
        if np.isnan(sizing_v):
            # Sizing vol unknown (warm-up, or too little history). Emit NaN so it
            # cannot masquerade as a real size, but never let it pass unnoticed.
            # NaN * mult stays NaN, so no mult applied here by construction.
            allocation_v = float('nan')
            config_object.masterlog.get_logger(self.logger_name, "critical")(
                f"{curr_time}: sizing_vol is NaN for {self.tup.coin1}/{self.tup.coin2} "
                f"{self.tup.agg_time} — allocation emitted as NaN; any entry this bar "
                f"would size to NaN.")
        elif sizing_v == 0.0:
            # Batch does (VT/annf)/0 -> inf -> clip(.,0,1) -> 1.0 on numpy arrays.
            # Reproduce that rather than diverge, but a zero sizing vol is pathological.
            allocation_v = float(np.clip(1.0 * mult_v, clip_lo, clip_hi))
            config_object.masterlog.get_logger(self.logger_name, "critical")(
                f"{curr_time}: sizing_vol is 0 for {self.tup.coin1}/{self.tup.coin2} "
                f"{self.tup.agg_time} — allocation saturated to "
                f"clip(1.0*mult)={allocation_v:.6f} (batch parity).")
        else:
            allocation_v = float(np.clip(
                np.clip((vt / annf) / sizing_v, 0.0, 1.0) * mult_v, clip_lo, clip_hi))

        self.numba_tup = PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE(
            p1=p1_v,
            minutely_high=float(min_bar.high),
            minutely_low=float(min_bar.low),
            middle=mid_v,
            middle_prev=mid_prev_v,
            upper=upper_v,
            atr14_pct=atr14_v,
            atr14_pct_prev=atr14_prev_v,
            atr50_pct=atr50_v,
            z_ema=z_in,
            m1=m1_v,
            corr=corr_in,
            txn_cost=float(sc['txn_cost']),
            slippage=float(sc['slippage_per_leg_per_turn']),
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