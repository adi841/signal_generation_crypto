
import pandas as pd
import numpy as np
import datetime as dt
from functools import lru_cache
import bottleneck as bn
import copy
import pickle
from collections import namedtuple
import json
import math
import traceback
import signal
import sys
from pytz import timezone

from config.config_read import config_object
from last_line_utils.directional_momentum_utils.directional_momentum_last_line_utils import DIRECTIONAL_MOMENTUM_PARENT_ID_STRAT_OBJ
from last_line_utils.directional_momentum_utils.directional_momentum_utils import DIRECTIONAL_MOMENTUM_PARAM_TUPLE, build_directional_momentum_parameter_dict, build_directional_momentum_parameter_dict_from_db
from last_line_utils.directional_momentum_utils.generate_signal_directional_momentum import GetDirectionalMomentumSignal

from .base import Base

## Prod import triggers a `git fetch origin` + a client lookup at module load
## (config_utils runs check_sync_with_origin() / check_db_host() at import time) —
## skip in local/test mode. Mirrors pair_momentum/pair_relative_value.
import os
if os.environ.get("CLIENT_ID"):
    from shared_codes.utils.config_utils import connect_postgre
else:
    def connect_postgre(db_user=None):
        return None, None

from utils.utils import get_inst_status_df

import warnings
warnings.filterwarnings("ignore")

import line_profiler
import atexit


class DirectionalMomentumSignalGenerator(Base):
    """DMP_v3_2 — Directional Momentum. One process per ASSET, 8 cells.

    SINGLE-ASSET sleeve: there is no coin2, no ratio and therefore no comb_df. One
    process drives all 8 streams of one asset — 4 TFs x {LONG, SHORT} — off a single
    OHLCV frame. The two sides of a TF are independent models (own parent_stratid,
    strategy object, kernel and order tag); they simply share this frame and the
    warm-up pass.
    """

    def __init__(self, base_coin1, param_num, debug, inheritor, start_time: dt.datetime, hist_replay_dir=None) -> None:

        ## ffill=False, as pair_momentum — NOT sa_mft/sa_directional's True.
        ##
        ## The reference DOES forward-fill the minute series (vendored_b1_dmp.prep:126
        ## `mc[cf] = mc[cf].ffill().bfill()` over minutely_high/low/close/ohlc_based), so the
        ## fill has to happen somewhere. Doing it explicitly in live.py rather than inside
        ## Base/MarketDataReplay buys two things:
        ##   * ONE code path. With ffill=True the fill would happen in
        ##     Base.merge_data_inst_status for the live socket and in
        ##     MarketDataReplay.__parse_df for replay — two implementations that can drift.
        ##     With False both paths hand us raw NaNs and live.py's ffill state machine is
        ##     the single source of truth.
        ##   * live.py keeps BOTH the ffilled and the raw value per minute in its
        ##     INPUT_DATA_TUPLE. The kernel consumes the ffilled one (matching the
        ##     reference); the raw one is what tells us the minute was actually missing,
        ##     which the fill-window reporting needs. ffill=True would erase that.
        super().__init__(ffill=False)

        assert inheritor in ['LIVE', 'BACKTEST'], "inheritor must be live or backtest"

        ## Single-asset config key: coin_param["directional_momentum"]["BTC"] =
        ## {"coin_1": ...}. Guard against being handed a PAIR key ("BTC_AVAX") — that
        ## entry resolves coin_1 to a real DMP asset, so without this the process would
        ## silently trade BTCUSDT while every log line and output path said "BTC_AVAX".
        key_ = f"{base_coin1}"
        param_entry = config_object.coin_param["directional_momentum"][key_]
        assert param_entry.get('coin_2') is None, (
            f"{key_} is a PAIR entry in coin_param (coin_2={param_entry.get('coin_2')}). "
            f"directional_momentum is single-asset — pass the asset key, e.g. 'BTC'.")

        self.base_coin1 = base_coin1
        self.param_entry = param_entry
        self.coin1 = param_entry['coin_1']
        self.start_parent_id = param_entry['start_parent_id']
        self.debug = debug
        self.param_num = param_num
        self.trading_model_dict = {}
        self.inheritor = inheritor # live or backtest
        self.hist_replay_dir = hist_replay_dir

        self.process_name = "{}_{}_{}".format(base_coin1, self.param_num, self.inheritor)
        self.conn, self.cursor = connect_postgre(db_user="signal_generation")

        self.get_strat_parameters(base_coin1)

        self.identity_name = f"{self.process_name}_directional_momentum_signal_gen"
        self.logger_name = f"directional_momentum_signal_gen_{self.param_num}_{base_coin1}"

        self.prepare_init_data(start_time, base_coin1)
        if self.hist_replay_dir:
            ## ONE symbol — the single-asset delta vs the pair sleeves' [coin1, coin2].
            self.market_data_replay.load_hist_replay_data(self.hist_replay_dir, [self.coin1], ffill=self.ffill)

        else:
            self.prepare_socket()

        ## Initialize market status
        last_indx = self.df1.index[-1]

    ##
    def get_strat_parameters(self, asset):
        """Build this process's cell set: DMP_v3_2 Directional Momentum, one parent_stratid
        per (asset, TF, side) stream — 4 TFs (5/15/30/60) x {LONG, SHORT} = 8 cells, per-asset
        local IDs 10000001..10000008 (stride 2: LONG at start+2*tf_index, SHORT at that +1).
        Every parameter is sourced from the frozen production snapshot
        (directional_momentum_production/params.json) via the pure helper, so the same logic
        is unit-testable / matchable without the live Base machinery.

        `asset` is the BASE name from argparse; the builder needs the resolved exchange
        symbol, so pass self.coin1 (set from config_object.coin_param) rather than the
        argument.

        hist_replay: frozen-bundle builder with per-asset LOCAL ids (keeps the recorded
        matching fixtures valid). LIVE: parameters AND parent ids come from
        submodel_parameters (is_live = 1) — the DB is the single source of truth; the
        live host does not need the frozen bundle, and artifact paths come from the
        client config's sleeve_config.
        """
        if self.hist_replay_dir:
            self.parameter_dict = build_directional_momentum_parameter_dict(self.coin1)
        else:
            db_rows = self.get_db_strat_params("directional_momentum")
            self.parameter_dict = build_directional_momentum_parameter_dict_from_db(
                db_rows, self.coin1,
                sleeve_paths=config_object.sleeve_config["directional_momentum"])

    def prepare_init_data(self, start_time, base_coin1):

        # ###
        if not self.hist_replay_dir:
            self.df1 = self.get_init_data(coin=self.coin1, start_time=start_time, ffill=self.ffill)

        else:
            self.load_init_hist_data()

        ##
        inst_status_df = get_inst_status_df()

        ## The warm-up frame the batch path and initialize() consume. The reference builds
        ## every one of its inputs off `raw[['Open','High','Low','Close']].ffill().bfill()`
        ## (candle resample, vendored_b1_dmp.prep:106) and `mc[cf].ffill().bfill()` (the
        ## minute fields, :126) — both from the same source frame — so one filled copy
        ## reproduces both.
        ##
        ## Kept SEPARATE from self.df1 (which stays raw) for the same reason pair_momentum
        ## keeps comb_df separate from df1/df2: live.py seeds its raw-tail state off self.df1
        ## and needs the NaNs intact to know which minutes were genuinely missing.
        self.warm_df = self.df1.copy()
        self.warm_df = self.warm_df.fillna(method='ffill')
        self.warm_df = self.warm_df.fillna(method='bfill')

        ###
        init_signal_gen = GetDirectionalMomentumSignal(tup=None, df1=self.df1, warm_df=self.warm_df, curr_time=start_time, process_name=self.process_name)

        ## TESTING HOOK: when DIRECTIONAL_MOMENTUM_DUMP_PID is set, enable debug so
        ## create_debug_df writes
        ## ./numpy_pandas_matching/directional_momentum/{coin1}/{parent_stratid}_numpy.parquet
        ## for every cell. create_debug_df makes its own sleeve/asset leaf directories, so
        ## only the flat parent is created here.
        import os as _os
        if _os.environ.get("DIRECTIONAL_MOMENTUM_DUMP_PID"):
            init_signal_gen.debug = True
            _os.makedirs("./numpy_pandas_matching", exist_ok=True)

        for parent_id, tup in self.parameter_dict.items():
            init_signal_gen._update_tup(tup)

            self.trading_model_dict[parent_id] = DIRECTIONAL_MOMENTUM_PARENT_ID_STRAT_OBJ(tup, logger_name=f"signal_gen_{self.process_name}")
            self.trading_model_dict[parent_id].initialize(self.warm_df, inst_status_df, start_time, self.process_name, signal_generator=init_signal_gen)

        self.trading_model_dict[parent_id].clear_data_cache()

        ## Trim self.df1/warm_df to the kernel's last-processed bar. Without this, live's
        ## first 2 incremental minutes (= the bars the batch path's trim-2 dropped, because
        ## next_close needs t+1 and t+2) see self.df1.iloc[-1] two bars AHEAD of
        ## last_processed_ts, and the kernel's same_close / close_ffill inputs read from the
        ## wrong minute. last_processed_ts is consistent across all 8 cells (same trim, same
        ## arrays) so reading it from any one trading model is safe.
        _last_proc_ts = self.trading_model_dict[parent_id].last_processed_ts
        self.df1 = self.df1.loc[:_last_proc_ts]
        self.warm_df = self.warm_df.loc[:_last_proc_ts]

        ## TESTING HOOK: exit before the DB/socket-bound post-init blocks below.
        if _os.environ.get("DIRECTIONAL_MOMENTUM_DUMP_PID"):
            import sys as _sys
            print(f"[DIRECTIONAL_MOMENTUM_DUMP] per-cell parquets written to "
                  f"./numpy_pandas_matching/directional_momentum/{self.coin1}/", flush=True)
            _sys.exit(0)

        ##
        if len(init_signal_gen.dump_pandas_ls) and self.hist_replay_dir is None:
            hist_signal_df = pd.concat(init_signal_gen.dump_pandas_ls)
            hist_signal_df.index = hist_signal_df.index.tz_localize("UTC")
            hist_signal_df.index.name = "signal_floor_time"
            hist_signal_df = hist_signal_df.reset_index()
            hist_signal_df["signal_time"] = dt.datetime.now().replace(tzinfo=timezone("UTC"))
            self.dump_hist_signal_df(hist_signal_df)

        ###
        self.df1 = self.df1.fillna(method='ffill')

    ##
    def load_init_hist_data(self):
        ## Warm-up _input files come from the SAME directory as the replay feed.
        ##
        ## Resolution order env -> hist_replay_dir -> the shared default. The shared
        ## ./backtest_codes/data is used by every sleeve, so a concurrent run (or manual
        ## work) there can swap the warm-up window out from under an in-flight test — that
        ## actually happened between the pair_momentum and pair_relative_value harnesses on
        ## 2026-08-02. Honouring hist_replay_dir makes each replay run self-contained with no
        ## env var needed, and carries no live-mode risk: load_init_hist_data() is only
        ## reached from the replay branch of prepare_init_data, so hist_replay_dir is never
        ## None here. The env var keeps the escape hatch the sibling sleeves use
        ## (PAIR_MOMENTUM_INIT_DIR, PAIR_RELATIVE_VALUE_INIT_DIR, SA_INIT_PARQUET).
        import os as _os
        init_dir = (_os.environ.get("DIRECTIONAL_MOMENTUM_INIT_DIR")
                    or self.hist_replay_dir or "./backtest_codes/data")

        self.df1 = pd.read_parquet(f"{init_dir}/{self.coin1}_input.parquet")
        if self.df1.index.tz is None:
            self.df1.index = self.df1.index.tz_localize("UTC")

    ## NOTE: no get_hist_replay_parent_ids(). The sibling sleeves' dead copies (nothing
    ## ever called them) were deleted when 4xxxxxxx stopped being the legacy pair_breakout
    ## band and became pair_momentum's parameter band. The parent ids for this sleeve come
    ## from build_directional_momentum_parameter_dict and nowhere else.
