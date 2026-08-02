
import pandas as pd
import numpy as np
import os
import datetime as dt
import time
import traceback
import argparse

from os.path import dirname, abspath, basename
import sys
import asyncio
from collections import deque, namedtuple


file_path = dirname(abspath(__file__))
while True:
    if file_path.endswith("signal_generation_crypto"):
        break

    file_path = dirname(file_path)

##
sys.path.append(file_path)


from config.config_read import config_object
from utils.directional_momentum_signal_generator import DirectionalMomentumSignalGenerator
from utils.directional_momentum_utils import (TradeOutputDirectionalMomentumLastOutput,
                                              TradeOutputDirectionalMomentumLastOutputCls)
from utils.directional_momentum_utils import (cryptoasset_dmpv32_long_iact_ll,
                                              cryptoasset_dmpv32_short_iact_ll)
from base_checks import BASE_CHECKS
from last_line_utils.directional_momentum_utils.directional_momentum_utils import (
    DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE, DIRECTIONAL_MOMENTUM_PARAM_TUPLE, SLEEVE)
from shared_codes.utils.dataclass_utils import StrategyType
from utils.utils import get_order_tag

## Prod import triggers a `git fetch origin` + a client lookup at module load
## (config_utils runs check_sync_with_origin() / check_db_host() at import time) —
## skip in local/test mode. Mirrors the sibling sleeves.
if os.environ.get("CLIENT_ID"):
    from shared_codes.utils.config_utils import connect_postgre
else:
    def connect_postgre(db_user=None):
        return None, None

import warnings
warnings.filterwarnings("ignore")


## SINGLE ASSET: 8 fields, not the pair sleeves' 16. There is no coin2 and no ratio.
INPUT_DATA_TUPLE = namedtuple("RUN_SIM_DATA", ["coin1_open_ffill", "coin1_open_raw",
                                               "coin1_high_ffill", "coin1_high_raw",
                                               "coin1_low_ffill", "coin1_low_raw",
                                               "coin1_close_ffill", "coin1_close_raw"])

## Frozen execution benchmark: config.FILL_WINDOW_MIN['DMP_v3_2'] = 2, i.e. the batch fills
## at the mean OHLC4 of the TWO minutes AFTER the decision
## (next_close = (ob.shift(-1) + ob.shift(-2)) / 2).
FILL_WINDOW_MIN = 2

## Column order of the OHLCV frames, and of the numpy row the replay path carries instead
## (utils/market_data_replay.py builds its per-minute tuples in exactly this order).
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

## Kernel dispatch by side. Each strategy object IS one side, so this is a per-cell lookup,
## not a per-minute branch on data.
_LL_KERNEL = {"LONG": cryptoasset_dmpv32_long_iact_ll,
              "SHORT": cryptoasset_dmpv32_short_iact_ll}


class LiveStrategy(DirectionalMomentumSignalGenerator):
    """DMP_v3_2 Directional Momentum — live per-minute driver.

    SINGLE ASSET, EIGHT CELLS: 4 TFs x {LONG, SHORT}. The two sides of a TF are INDEPENDENT
    MODELS — separate parent_stratid (stride 2: LONG at 10000001+2i, SHORT at that +1),
    separate strategy object, separate kernel, separate order tag, separate broker row.
    Measured on the goldens they are both in position simultaneously 72.8% of minutes, so
    the process routinely emits an opposing pair on the same instrument; netting is the
    execution layer's business, not ours.

    Nothing from the reversal2 / QR1 second-entry lineage is present: no res2, no
    profit_target, no order_tag2, no `parent_id + 2` diagnostic row. DMP's "second stream"
    is the other SIDE, which is a different parent_stratid entirely.
    """

    def __init__(self, base_coin1, param_num, debug, inheritor="LIVE", start_time=None,
                 hist_replay=None, hist_replay_dir=None) -> None:
        super().__init__(base_coin1, param_num, debug, inheritor, start_time, hist_replay_dir)

        ##
        self.hist_replay = hist_replay

        ## DMP has no profit_booked / stop_loss state, so no pb_sl pickle persistence.
        self.exec_signal_dict = {}   # broker-bound execution signals
        self.db_signal_dict = {}     # DB-bound diagnostic signals
        self.logging_dict = {}
        self.entry_price_dict = {}   # parent_id -> reported traded price
        self.fill_window_dict = {}   # parent_id -> in-flight fill window (see below)
        self.df1_raw = None

        ## ---- replay fast-path state (see _replay_fetch / _run_replay) ---------------
        ## Outside the live-only logging, nothing downstream of self.df1 reads more than 4
        ## scalars and one timestamp. In replay we therefore carry the ffill state as a
        ## plain numpy row instead of rebuilding a DataFrame every minute: the pandas
        ## version (1-row frame construction + concat + fillna + .iloc[-1] on a FIVE-row
        ## frame) measured ~3.9 ms per call on the pair sleeves, all of it fixed per-call
        ## overhead rather than anything that scales with the data. Single asset means ONE
        ## fetch per minute here, against the pair sleeves' two.
        ## self.df1 stays maintained in LIVE mode, where it feeds the per-minute tail(2)
        ## log and the startup timestamp assert in __main__.
        self._ffill1 = None          # np.array([open, high, low, close, volume]), ffilled
        self._last_ts1 = None        # stands in for self.df1.index[-1]

        ## Per-SLEEVE, per-ASSET output directory. entry_data_dir is shared by every sleeve,
        ## and parent_stratids are per-asset LOCAL — every asset reuses 10000001..10000008 —
        ## so a flat entry_data_dir/{parent_id}.csv would have two assets overwriting each
        ## other. The two SIDES of a cell do not collide: they have different ids.
        ##
        ## Created HERE rather than in log_dump_data on purpose: that whole block is wrapped
        ## in a try/except that logs and swallows, so a missing directory there would mean
        ## silently writing nothing. Failing at construction is loud.
        self.entry_dir = f"entry_data_dir/{SLEEVE}/{self.coin1}"
        os.makedirs(self.entry_dir, exist_ok=True)

        if self.hist_replay:
            ## prepare_init_data has already trimmed df1 to last_processed_ts and ffilled
            ## it, so the last row is exactly the ffill baseline the original
            ## `pd.concat([...]).fillna(method='ffill').iloc[-1]` would have started from.
            assert list(self.df1.columns) == OHLCV_COLS, \
                f"replay fast path assumes column order {OHLCV_COLS}, got {list(self.df1.columns)}"
            self._ffill1 = self.df1.to_numpy(dtype='float64')[-1].copy()
            self._last_ts1 = self.df1.index[-1].tz_localize(None) if self.df1.index.tz else self.df1.index[-1]

    ##
    def _initlization_checks(self, parent_id, price):
        if parent_id not in self.logging_dict:
            self.logging_dict[parent_id] = {}

        ##
        if parent_id not in self.entry_price_dict:
            self._set_reported_prices_raw(parent_id, price)

        if parent_id not in self.db_signal_dict:
            self.exec_signal_dict[parent_id] = {}
            self.db_signal_dict[parent_id] = {}

    # ------------------------------------------------------------------------------
    # Reported fill price — a progressive window over the frozen benchmark.
    #
    # The execution engine consumes `price`, so it must be an ACHIEVABLE price. The
    # benchmark is the mean OHLC4 of the two minutes AFTER the decision, which has not
    # happened yet at the decision minute. So publish an estimate and refine it as the
    # window elapses:
    #
    #   T    (transition fires)  ->  ohlc4(T)                       estimate
    #   T+1                      ->  ohlc4(T+1)                     first real window minute
    #   T+2                      ->  mean(ohlc4(T+1), ohlc4(T+2))   the frozen benchmark
    #   T+3 onward               ->  frozen at the T+2 value
    #
    # This lives purely in the REPORTING layer, after the kernel, and it is safe to do so:
    # neither DMP kernel reads next_close or tradeprice inside any condition — their
    # in-trade state is entry_median / entry_atr / the extreme. (Contrast QR1_v4, whose
    # kernel reads a fill price inside the scale-in trigger, which is exactly why that
    # sleeve needs a sentinel backfill BEFORE the next kernel call and can only ever be
    # knife-edge-close to its batch. DMP can be exact.)
    # ------------------------------------------------------------------------------
    def _set_reported_prices(self, parent_id, price_no_tc, slippage, buying):
        """Apply slippage with the kernel's signs and publish.

        `buying` is the direction of the FILL, which depends on both the model's side and
        whether this is an entry or an exit — mirroring the two kernels exactly:
            LONG  entry  next_close*(1+slip)     LONG  exit  next_close*(1-slip)
            SHORT entry  next_close*(1-slip)     SHORT exit  next_close*(1+slip)
        """
        px = price_no_tc * (1 + slippage) if buying else price_no_tc * (1 - slippage)
        self._set_reported_prices_raw(parent_id, px)

    def _set_reported_prices_raw(self, parent_id, price):
        self.entry_price_dict[parent_id] = {'traded_price': price}

    def _open_fill_window(self, parent_id, buying, ohlc4, slippage):
        """A position just opened or closed this minute. Publish the estimate and start the
        window. Opening OVERWRITES any in-flight window, so a re-entry during an exit's
        window — or an exit during an entry's window — resolves in favour of the newer
        event."""
        self.fill_window_dict[parent_id] = {"n": 0, "buying": buying, "p": None}
        self._set_reported_prices(parent_id, ohlc4, slippage, buying)

    def _advance_fill_window(self, parent_id, ohlc4, slippage):
        """Advance an in-flight window by one minute. Returns True if one was live."""
        w = self.fill_window_dict.get(parent_id)
        if w is None:
            return False

        w["n"] += 1
        if w["n"] == 1:
            # First minute of the real fill window.
            w["p"] = ohlc4
            self._set_reported_prices(parent_id, ohlc4, slippage, w["buying"])
        else:
            # Window complete: the frozen benchmark, then freeze.
            self._set_reported_prices(parent_id, (w["p"] + ohlc4) / FILL_WINDOW_MIN,
                                      slippage, w["buying"])
            del self.fill_window_dict[parent_id]
        return True

    def _get_update_pb_sl_dict(self, parent_id, output_numba_cls: TradeOutputDirectionalMomentumLastOutputCls):
        """DMP has no profit_booked state — always returns 0 so the broker contract's
        `is_pb_sl` field stays a stable scalar."""
        return 0

    def _get_asset_ohlc4(self, data: INPUT_DATA_TUPLE):
        return (data.coin1_open_ffill + data.coin1_high_ffill
                + data.coin1_low_ffill + data.coin1_close_ffill) / 4

    ##
    def run_simulation(self, curr_time, data: INPUT_DATA_TUPLE):
        """Per-minute live driver: advance each cell's aggregator + features via `update()`,
        call that cell's side of the DMP_v3_2 per-bar kernel, emit the order tag on a state
        transition, refine the reported fill price, and queue the broker / DB rows."""

        ## The asset's own bar. No ratio is synthesized — the pair sleeves build one here;
        ## DMP just passes the instrument through.
        ##
        ## The RAW fields are fed to update(), not the ffilled ones, because the aggregator
        ## already owns the fill: its update() substitutes `prev_minutely_ohlcv` whenever any
        ## of O/H/L/C is NaN, which IS the forward fill, and that is what the batch's
        ## `mc[ff].ffill()` reproduces. Handing it pre-filled values would work identically
        ## but would hide a missing minute from the one place that logs it.
        asset_open = data.coin1_open_raw
        asset_high = data.coin1_high_raw
        asset_low = data.coin1_low_raw
        asset_close = data.coin1_close_raw

        d_ml = config_object.deque_maxlen
        ## Costs come from the RUNTIME config (framework convention), not the frozen bundle,
        ## so ops can change them without touching params.json. DMP is a breakout. Costs
        ## never affect the target position — only the reported fill price — because
        ## txn_cost and slippage appear in no kernel CONDITION.
        txn_cost = config_object.fixed_cost_dict["directional_momentum"]
        ## Single asset: this symbol's slippage, with no leg averaging (the pair sleeves
        ## charge mean(leg1, leg2)).
        slippage = config_object.slippage_hlc3_dict[self.coin1]

        asset_ohlc4 = self._get_asset_ohlc4(data)

        for _, tup in self.parameter_dict.items():
            tup: DIRECTIONAL_MOMENTUM_PARAM_TUPLE
            parent_id = tup.parent_stratid

            numba_input_tup: DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE = self.trading_model_dict[parent_id].update(
                curr_time, open_=asset_open, high=asset_high, low=asset_low,
                close=asset_close, volume=0,
            )
            input_numba_cls: TradeOutputDirectionalMomentumLastOutputCls = self.trading_model_dict[parent_id].numba_cls

            self._initlization_checks(parent_id=parent_id, price=data.coin1_close_ffill)

            ## This cell's side picks BOTH the kernel and which of the two minute extremes /
            ## band lines it reads. The numba input tuple is uniform across sides (it carries
            ## both) so one dump schema serves the sleeve; the selection happens here.
            is_long = (tup.side == "LONG")
            minutely_extreme = (numba_input_tup.minutely_low if is_long
                                else numba_input_tup.minutely_high)
            band_line = (numba_input_tup.upper_line if is_long
                         else numba_input_tup.lower_line)

            ## DMP_v3_2 per-bar kernel. Positional, grouped exactly as the signature:
            ##   13 per-bar  (next_close = THIS minute's realized fill — no lookahead; it is
            ##                written into tradeprice and never read by any test)
            ##    7 per-cell scalars off the frozen param tuple (dde_mult already carries
            ##                M_LONG=4.0 / M_SHORT=1.5; there is no GRACE — B1 has one, DMP
            ##                does not)
            ##    9 carried state from the previous minute
            output = _LL_KERNEL[tup.side](
                asset_ohlc4, data.coin1_close_ffill, numba_input_tup.p1,
                minutely_extreme, numba_input_tup.asset_close, numba_input_tup.median_line,
                band_line, numba_input_tup.atr_eq, numba_input_tup.long_vol,
                numba_input_tup.z_median,
                txn_cost, slippage, numba_input_tup.allocation,

                float(tup.T1), float(tup.TT), float(tup.dde_mult),
                float(tup.K), float(tup.X), float(tup.Z), float(tup.EZ),

                input_numba_cls.trade_allocation_lo,
                input_numba_cls.signal_on,
                input_numba_cls.can_take_new_trade,
                input_numba_cls.entry_median_lo,
                input_numba_cls.entry_atr_lo,
                input_numba_cls.extreme_price_lo,
                input_numba_cls.entry_long_vol_lo,
                input_numba_cls.bars_in_trade_lo,
                input_numba_cls.developed_lo,
            )

            output: TradeOutputDirectionalMomentumLastOutput
            output_numba_cls = TradeOutputDirectionalMomentumLastOutputCls(**output._asdict())

            ## ---- order tag + reported price -------------------------------------
            ## Each object is a SINGLE-DIRECTION stream (res in {0, +1} for LONG, {0, -1}
            ## for SHORT), so a single 0/non-0 transition test covers both entry and exit
            ## and there is no flip case to handle — unlike sa_mft, which is one bidirectional
            ## stream and must test `curr != prev`.
            prev_signal = input_numba_cls.res_lo
            curr_signal = output_numba_cls.res_lo
            transitioned = (prev_signal == 0) != (curr_signal == 0)

            if transitioned:
                ## get_order_tag asserts isinstance(signal, int); res_lo comes back from the
                ## numba kernel as a float, so cast at the boundary.
                order_tag1 = get_order_tag(curr_time, parent_id, int(curr_signal))
                self.trading_model_dict[parent_id].update_order_tag1(order_tag1=order_tag1)
                ## Fill direction: LONG entry and SHORT exit BUY; LONG exit and SHORT entry
                ## SELL. Both kernels' slippage signs follow exactly this.
                is_entry = (curr_signal != 0)
                self._open_fill_window(parent_id, is_long == is_entry, asset_ohlc4, slippage)

            elif not self._advance_fill_window(parent_id, asset_ohlc4, slippage):
                ## No window in flight. Flat -> keep publishing the current market price;
                ## in position -> leave the settled entry price frozen.
                if curr_signal == 0:
                    self._set_reported_prices_raw(parent_id, asset_ohlc4)

            ## Persist next-bar state. No deepcopy: output_numba_cls was constructed fresh
            ## above from the kernel's namedtuple of floats, is never mutated afterwards, and
            ## is replaced by a new object next minute — so there is nothing to alias.
            self.trading_model_dict[parent_id].numba_cls = output_numba_cls

            ## Everything below is broker / DB bookkeeping. In replay it is dead weight:
            ## run_simulation returns below before anything reads exec_signal_dict or
            ## db_signal_dict, and the CSV the matching harness compares comes from
            ## log_dump_data -> get_logging_dict, which reads the trading model directly.
            ## Skipping it also skips populate_db_signal_dict's broker-contract asserts,
            ## which only the live path then exercises.
            if self.hist_replay:
                continue

            ## DMP has no profit-target / stop-loss OUTPUTS (the hard stop is a kernel
            ## branch, not a resting order), so those legacy DB payload fields stay None.
            is_pb_sl = self._get_update_pb_sl_dict(parent_id, output_numba_cls)
            pft_take = None
            stop_loss1 = None

            curr_signal_1 = int(curr_signal)
            price_e1 = self.entry_price_dict[parent_id]['traded_price']
            order_tag1 = self.trading_model_dict[parent_id].order_tag1

            msg_ = (f"{parent_id}:\t {tup.side} \t Signal: {curr_signal_1} \t "
                    f"Price: {price_e1} \t asset_close: {numba_input_tup.asset_close} \t "
                    f"band: {band_line} \t zmed: {numba_input_tup.z_median} \t "
                    f"alloc: {numba_input_tup.allocation} \t {curr_time}")
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            send_curr_time = curr_time.replace(tzinfo=None)
            signal_time = dt.datetime.utcnow()
            signal_id = int(send_curr_time.timestamp())

            ## SINGLE-ASSET broker schema, following the sa_directional precedent this sleeve
            ## replaces in the 1xxxxxxx band: trading_symbol1 only, one `price`, and none of
            ## the pair sleeves' price_l1/price_l2 or second_entry_* columns — there are no
            ## legs and no second entry. Flagged for the execution side to confirm.
            self.populate_db_signal_dict(parent_id=parent_id, signal_time=signal_time,
                                         signal_floor_time=send_curr_time, signal_id=signal_id,
                                         signal=curr_signal_1, price=price_e1,
                                         pft_take=pft_take, stop_loss=stop_loss1,
                                         strategy_type=StrategyType.BREAKOUT,
                                         trading_symbol1=self.coin1,
                                         is_pb_sl=is_pb_sl, is_execution_signal=True,
                                         order_tag=order_tag1)

            self.populate_db_signal_dict(parent_id=parent_id, signal_time=signal_time,
                                         signal_floor_time=send_curr_time, signal_id=signal_id,
                                         signal=curr_signal_1, price=price_e1,
                                         ## DMP publishes no case, but live_signals.case_num is
                                         ## NOT NULL -- 0 here matches what the batch generator
                                         ## and the backtest driver already write.
                                         case_num=0,
                                         ## Real int from submodel_parameters on the live path;
                                         ## the bundle path leaves it None but returns above,
                                         ## at `if self.hist_replay: continue`, before this runs.
                                         exec_type=1, ## TODO: Fix this
                                         is_pb_sl=is_pb_sl, is_execution_signal=False,
                                         order_tag=order_tag1)

        ##
        if self.hist_replay:
            return

        if len(self.db_signal_dict[parent_id]["signal"]) < (d_ml - 1):
            return

        ##
        df_ = self.get_exec_signals_dataframe_fast(exec_signal_dict=self.exec_signal_dict)

        asyncio.run(self.send_signal_broker(df_))

    ####
    def populate_db_signal_dict(self, parent_id, signal_time, signal_floor_time, signal_id,
                                signal, price, pft_take=None, stop_loss=None,
                                strategy_type=None, trading_symbol1=None, is_pb_sl=None,
                                case_num=None, exec_type=None,
                                is_execution_signal: bool = None, order_tag=None):

        d_ml = config_object.deque_maxlen

        assert isinstance(is_execution_signal, bool), f"is_execution_signal should be bool, got {type(is_execution_signal)}"

        if is_execution_signal:
            non_nan_check_ls = [signal_time, signal_floor_time, parent_id, signal_id, signal,
                                price, strategy_type, trading_symbol1, is_pb_sl]
            assert all([(x != None) for x in non_nan_check_ls]), f"All values should be provided for execution signal, got {non_nan_check_ls}"

            nan_check_ls = [pft_take, stop_loss]
            assert all([(x is None) for x in nan_check_ls]), f"All values should be None for execution signal, got {nan_check_ls}"

            self.exec_signal_dict[parent_id].setdefault("signal_time", deque(maxlen=d_ml)).append(signal_time)
            self.exec_signal_dict[parent_id].setdefault("signal_floor_time", deque(maxlen=d_ml)).append(signal_floor_time)
            self.exec_signal_dict[parent_id].setdefault("parent_trading_model", deque(maxlen=d_ml)).append(parent_id)
            self.exec_signal_dict[parent_id].setdefault("signal_id", deque(maxlen=d_ml)).append(signal_id)
            self.exec_signal_dict[parent_id].setdefault("signal", deque(maxlen=d_ml)).append(signal)
            self.exec_signal_dict[parent_id].setdefault("price", deque(maxlen=d_ml)).append(price)
            self.exec_signal_dict[parent_id].setdefault("pft_take", deque(maxlen=d_ml)).append(pft_take)
            self.exec_signal_dict[parent_id].setdefault("stop_loss", deque(maxlen=d_ml)).append(stop_loss)
            self.exec_signal_dict[parent_id].setdefault("strategy_type", deque(maxlen=d_ml)).append(strategy_type)
            self.exec_signal_dict[parent_id].setdefault("trading_symbol1", deque(maxlen=d_ml)).append(trading_symbol1)
            self.exec_signal_dict[parent_id].setdefault("is_pb_sl", deque(maxlen=d_ml)).append(is_pb_sl)
            self.exec_signal_dict[parent_id].setdefault("init_order_tag", deque(maxlen=d_ml)).append(order_tag)

        else:
            self.db_signal_dict[parent_id].setdefault("signal_time", deque(maxlen=d_ml)).append(signal_time)
            self.db_signal_dict[parent_id].setdefault("signal_floor_time", deque(maxlen=d_ml)).append(signal_floor_time)
            self.db_signal_dict[parent_id].setdefault("parent_trading_model", deque(maxlen=d_ml)).append(parent_id)
            self.db_signal_dict[parent_id].setdefault("signal_id", deque(maxlen=d_ml)).append(signal_id)
            self.db_signal_dict[parent_id].setdefault("signal", deque(maxlen=d_ml)).append(signal)
            self.db_signal_dict[parent_id].setdefault("price", deque(maxlen=d_ml)).append(price)
            ## Base.dump_signals reads BOTH of these by name out of this dict, and
            ## live_signals.case_num / .execution_type are NOT NULL. Omitting them does not
            ## fail the INSERT -- it raises KeyError before the statement is even built, and
            ## log_dump_data's except swallows it, so the process looks healthy while nothing
            ## after the warm-up backfill is ever persisted.
            self.db_signal_dict[parent_id].setdefault("case_num", deque(maxlen=d_ml)).append(case_num)
            self.db_signal_dict[parent_id].setdefault("exec_type", deque(maxlen=d_ml)).append(exec_type)
            self.db_signal_dict[parent_id].setdefault("is_pb_sl", deque(maxlen=d_ml)).append(is_pb_sl)
            self.db_signal_dict[parent_id].setdefault("order_tag", deque(maxlen=d_ml)).append(order_tag)

    ###
    def log_dump_data(self, curr_time):
        ##
        try:
            for _, tup in self.parameter_dict.items():
                ## Debug start
                if tup.parent_stratid not in self.trading_model_dict:
                    continue

                ##
                self.logging_dict[tup.parent_stratid].setdefault("indx", []).append(curr_time)
                self.trading_model_dict[tup.parent_stratid].get_logging_dict(self.logging_dict[tup.parent_stratid])

            ##
            dump_flag = True
            if self.hist_replay:
                dump_flag = False
                if curr_time.hour == 15 and curr_time.minute == 29:
                    dump_flag = True

            if dump_flag:
                for parent_id in self.logging_dict:
                    file_path = f"{self.entry_dir}/{parent_id}.csv"
                    file_exists = os.path.exists(file_path)
                    is_header = (not file_exists)

                    pd.DataFrame(self.logging_dict[parent_id]).to_csv(file_path, mode='a', header=is_header, index=False)
                    self.logging_dict[parent_id] = {}

            ##
            if self.hist_replay:
                return

            try:
                # dump db signal dict
                self.dump_signals(signal_dict=self.db_signal_dict)

            except Exception as e:
                tb_ = traceback.format_exc()
                msg_ = f"Error: {e} \n {tb_}"
                config_object.masterlog.get_logger(self.logger_name, "error")(msg_)

        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"Error in log_dump_data: {e} \n {tb_}"
            config_object.masterlog.get_logger(self.logger_name, "error")(msg_)

    # ------------------------------------------------------------------------------
    # Replay fast path.
    #
    # The LIVE path below is unchanged. This pair of methods reproduces it exactly for the
    # 8 scalars the strategy consumes, without the pandas plumbing:
    #
    #   utils/base.py::get_hist_replay_data wraps a 5-float tuple in a DataFrame via
    #   pd.DataFrame([tuple]) -> set_index -> replace(pd.NaT, np.nan) -> astype('float64')
    #   -> tz_localize, measured at ~3.9 ms PER CALL on the pair sleeves and the single
    #   largest line item in their replay budget. It lives in base.py, which every sleeve
    #   shares, so the replacement stays local to this sleeve.
    #
    #   `replace(pd.NaT, np.nan)` and `astype('float64')` are both no-ops here:
    #   MarketDataReplay.__parse_df builds its tuples from `df[cols].to_numpy()` over
    #   float64 columns, so the values are already np.float64 and NaT cannot occur.
    #
    #   The "MISSING TIMESTAMP" check in update_data is likewise dead in replay — the frame
    #   is built with index = curr_time, so `curr_time - index[-1]` is always 0.
    # ------------------------------------------------------------------------------
    def _replay_fetch(self, curr_time, coin):
        """One minute of OHLCV for `coin` as a 5-float numpy row, in OHLCV_COLS order."""
        if curr_time.tzinfo is not None:
            curr_time = curr_time.replace(tzinfo=None)
        return np.asarray(self.market_data_replay.get_hist_replay_data(curr_time, coin),
                          dtype='float64')

    def _run_replay(self, curr_time):
        """Replay twin of the data-fetch half of `run()`. Returns the same INPUT_DATA_TUPLE."""
        raw1 = self._replay_fetch(curr_time, self.coin1)
        ts = curr_time.replace(tzinfo=None) if curr_time.tzinfo is not None else curr_time

        ## `pd.concat([self.df1, new_row]).fillna(method='ffill').iloc[-1]` reduces to
        ## exactly this: per column, take the new value unless it is NaN, in which case carry
        ## the previous one forward. The feed is loaded with ffill=False
        ## (directional_momentum_signal_generator.py), so missing minutes really do arrive as
        ## NaN and this is doing work. The `ts >` guard mirrors update_data's
        ## `if df_temp_coin.index[-1] <= self.df1.index[-1]: return` — a non-advancing bar
        ## must not move the ffill state.
        if ts > self._last_ts1:
            self._ffill1 = np.where(np.isnan(raw1), self._ffill1, raw1)
            self._last_ts1 = ts

        f1 = self._ffill1
        return INPUT_DATA_TUPLE(
            coin1_open_ffill=f1[0], coin1_high_ffill=f1[1],
            coin1_low_ffill=f1[2], coin1_close_ffill=f1[3],
            coin1_open_raw=raw1[0], coin1_high_raw=raw1[1],
            coin1_low_raw=raw1[2], coin1_close_raw=raw1[3],
        )

    ##
    async def update_data(self, curr_time, coin):

        df_temp_coin = await self.get_ohlcv_data(curr_time, coin)

        if (curr_time - df_temp_coin.index[-1]) > dt.timedelta(minutes=1):
            msg_ = f"MISSING TIMESTAMP. Data not available for {curr_time}"
            config_object.masterlog.get_logger(self.logger_name, "error")(msg_)

        if coin == self.coin1:
            if df_temp_coin.index[-1] <= self.df1.index[-1]:
                return df_temp_coin

            else:
                self.df1 = pd.concat([self.df1, df_temp_coin])
                self.df1 = self.df1.fillna(method='ffill')
                return df_temp_coin
        else:
            raise ValueError(f"Invalid coin: {coin} (this sleeve is single-asset, expected {self.coin1})")

    def get_data_db_ltp(self, curr_time, coin):
        ## Get data from database
        conn, cursor = connect_postgre(db_user="signal_generation")

        data_query_ = f"select * from ohlcv_data where symbol = '{coin}' and start_time <= '{curr_time}' order by start_time DESC limit 1;"
        cursor.execute(data_query_)
        reply = cursor.fetchall()
        reply = reply[0]
        reply = (curr_time, curr_time, reply['open'], reply['high'], reply['low'], reply['close'], reply['volume'])

        ##
        conn.close(); cursor.close()

        return reply

    ##
    def run(self, curr_time):
        try:
            ## Replay takes the numpy fast path and returns; everything below this block is
            ## the LIVE path and is untouched.
            if self.hist_replay:
                self.run_simulation(curr_time, self._run_replay(curr_time))
                return

            ##
            st_time_st = time.time()
            self.df1_raw = asyncio.run(self.update_data(curr_time, self.coin1))

            # ##
            coin1_open_ffill, coin1_high_ffill, coin1_low_ffill, coin1_close_ffill, _ = self.df1.iloc[-1][OHLCV_COLS]

            ####
            coin1_open_raw, coin1_high_raw, coin1_low_raw, coin1_close_raw, _ = self.df1_raw.iloc[-1][OHLCV_COLS]

            ##
            input_data_tup = INPUT_DATA_TUPLE(
                coin1_open_ffill=coin1_open_ffill, coin1_open_raw=coin1_open_raw,
                coin1_high_ffill=coin1_high_ffill, coin1_high_raw=coin1_high_raw,
                coin1_low_ffill=coin1_low_ffill, coin1_low_raw=coin1_low_raw,
                coin1_close_ffill=coin1_close_ffill, coin1_close_raw=coin1_close_raw)

            ##
            self.run_simulation(curr_time, input_data_tup)

            msg_ = f"Time taken: {time.time() - st_time_st} seconds \t {curr_time}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            ##
            self.df1 = self.df1.iloc[-5:]

            msg_ = "------" + f"{self.coin1}" + "------\n"
            msg_ += self.df1.tail(2).to_string()
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        except Exception as e:
            ## NOTE: pair_lead_lag / sa_* drop into `ipdb.set_trace()` here, which hangs a
            ## live process on stdin waiting for a terminal that isn't there. Log loudly and
            ## re-raise so the caller's handler decides.
            tb = traceback.format_exc()
            key_ = f"{self.coin1}_{self.param_num}"
            msg_ = f"{key_}:\t Error: {e} \t {curr_time} \n {tb}"
            print(msg_, flush=True)
            config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
            raise


if __name__ == "__main__":
    """
    DMP_v3_2 Directional Momentum. One process hosts one (or several) ASSETS; each asset
    runs 8 cells — TF {5, 15, 30, 60} x {LONG, SHORT} — at parent_stratid
    10000001 + 2*tf_index (+1 for the short side).
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--coin1', nargs="+", help='asset name', required=True)
    parser.add_argument('--store_output', type=int, help='whether to store the output or not', required=True, choices=[0, 1])
    parser.add_argument('--param_num', type=str, help='param_num', required=True)
    parser.add_argument('--curr_time', type=str, help='UTC datetime format: %Y-%m-%d %H:%M:%S. 2024-09-01 22:00:00', required=True)
    parser.add_argument('--hist_replay', type=int, help='replaying hist data', required=False, choices=[0, 1], default=0)
    parser.add_argument('--hist_replay_dir', type=str, help='directory to load data for hist replay', required=False, default=None)

    args = parser.parse_args()

    """
    python directional_momentum/live.py --coin1 BTCUSDT --store_output 1 --param_num 0 \
        --curr_time "2021-06-30 23:59:00" --hist_replay 1 --hist_replay_dir ./backtest_codes/data_dm
    """

    ##
    start_time = dt.datetime.strptime(args.curr_time, "%Y-%m-%d %H:%M:%S")

    store_data = False
    if args.store_output == 1:
        store_data = True

    ##
    # BASE_CHECKS()

    ##
    dt_list = []
    COIN_OBJ_DICT = {}
    for coin1 in args.coin1:
        key_ = f"{coin1}_{args.param_num}"
        obj = LiveStrategy(base_coin1=coin1, param_num=args.param_num, debug=store_data,
                           start_time=start_time, hist_replay=args.hist_replay,
                           hist_replay_dir=args.hist_replay_dir)
        COIN_OBJ_DICT[key_] = obj
        dt_list.append(obj)

    ##
    # curr_time_check
    assert all(dt_.df1.index[-1] == dt_list[0].df1.index[-1] for dt_ in dt_list), f"Not all coins have same current time: {dt_list}"

    ##
    # Anchor live's first incremental minute to the bar AFTER batch init's last processed
    # bar. get_numba_parameters trims the last 2 minutes (needed for the 2-bar forward fill
    # convention), so df1.index[-1] is 2 bars AHEAD of where the kernel actually finished.
    # Reading the trim-aware last_processed_ts from any one trading model gives the correct
    # anchor — it is identical across all 8 cells of an asset (same trim, same arrays).
    _any_obj = COIN_OBJ_DICT[key_]
    _any_tm = next(iter(_any_obj.trading_model_dict.values()))
    curr_time = _any_tm.last_processed_ts + dt.timedelta(minutes=1)
    curr_time = curr_time.replace(second=0, microsecond=0, tzinfo=_any_obj.base_tz)

    while True:
        try:
            now_dt = dt.datetime.now().replace(second=0, microsecond=0)
            if curr_time != COIN_OBJ_DICT[key_].base_tz.localize(now_dt):
                if not args.hist_replay:
                    time.sleep(1)

                for key_ in COIN_OBJ_DICT:
                    try:
                        COIN_OBJ_DICT[key_].run(curr_time)

                    except Exception as e:
                        tb_ = traceback.format_exc()
                        msg_ = f"Error: {e} \n {tb_}"
                        config_object.masterlog.get_logger(f"signal_gen_{COIN_OBJ_DICT[key_].process_name}", "critical")(msg_)

                ###
                for key_ in COIN_OBJ_DICT:
                    COIN_OBJ_DICT[key_].log_dump_data(curr_time)

                ##
                curr_time += dt.timedelta(minutes=1)

            if not args.hist_replay:
                sleep_time = (60 - (time.time() % 60))
                sleep_time = min(sleep_time, 1)
                time.sleep(sleep_time)

        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"Error: {e} \n {tb_}"
            config_object.masterlog.get_logger(f"signal_gen_{COIN_OBJ_DICT[key_].process_name}", "critical")(msg_)
            time.sleep(1)
