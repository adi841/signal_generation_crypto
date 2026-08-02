
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
from last_line_utils.pair_lead_lag_utils.pair_lead_lag_last_line_utils import PAIR_LEAD_LAG_PARENT_ID_STRAT_OBJ
from last_line_utils.pair_lead_lag_utils.pair_lead_lag_utils import PAIR_LEAD_LAG_PARAM_TUPLE, build_pair_lead_lag_parameter_dict, build_pair_lead_lag_parameter_dict_from_db
from last_line_utils.pair_lead_lag_utils.generate_signal_pair_lead_lag import GetPairsLeadLagSignal

from .base import Base

## Prod import triggers a `git fetch origin` + a client lookup at module load
## (config_utils runs check_sync_with_origin() / check_db_host() at import time) —
## skip in local/test mode. Mirrors sa_directional/live.py and sa_mft/live.py.
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

#
import os

# file_name = os.path.basename(__file__)
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
# atexit.register(profile.dump_stats, file_name.replace("py", "txt"))

class PairLeadLagSignalGenerator(Base):
    
    def __init__(self, base_coin1, base_coin2, param_num, debug, inheritor, start_time: dt.datetime, hist_replay_dir=None) -> None:
        
        super().__init__(ffill=False)

        assert inheritor in ['LIVE', 'BACKTEST'], "inheritor must be live or backtest"
        
        key_ = f"{base_coin1}_{base_coin2}"
        self.base_coin1 = base_coin1
        self.base_coin2 = base_coin2
        ## Per-sleeve unit entry: coin resolution, socket lane and this unit's GLOBAL
        ## parent-id base (cumulative band packing — must match submodel_parameters).
        self.param_entry = config_object.coin_param["pair_lead_lag"][key_]
        self.coin1 = self.param_entry['coin_1']
        self.coin2 = self.param_entry['coin_2']
        self.start_parent_id = self.param_entry['start_parent_id']
        self.debug = debug
        self.param_num = param_num
        self.trading_model_dict = {}
        self.inheritor = inheritor # live or backtest
        self.hist_replay_dir = hist_replay_dir

        self.process_name = "{}_{}_{}_{}".format(base_coin1, base_coin2, self.param_num, self.inheritor)
        self.conn, self.cursor = connect_postgre(db_user="signal_generation")
        
        self.get_strat_parameters(base_coin1, base_coin2)

        self.identity_name = f"{self.process_name}_pairs_signal_gen_tradefi"
        self.logger_name = f"pair_lead_lag_signal_gen_{self.param_num}_{base_coin1}_{base_coin2}"

        self.prepare_init_data(start_time, base_coin1, base_coin2)
        if self.hist_replay_dir:
            self.market_data_replay.load_hist_replay_data(self.hist_replay_dir, [self.coin1, self.coin2], ffill=self.ffill)
        
        else:
            self.prepare_socket()
        
        ## Initialize market status
        last_indx = self.df1.index[-1]
                
    ##
    def get_strat_parameters(self, coin1, coin2):
        """Build this process's cell set: QR31_v2 Pairs Lead-Lag, one parent_stratid
        per (pair, TF) sleeve — all 5 TFs (15/30/60/120/240), per-pair local IDs
        50000001 + 2*tf_index (+1 is the leg-2 slot; moved off the sleeve's historic
        2xxxxxxx band — the old B1_v8_v9 IDs there are dead). Every parameter is
        sourced from the
        frozen production snapshot (pair_lead_lag_production/params.json) via the
        pure helper, so the same logic is unit-testable / matchable without the live
        Base machinery.

        `coin1`/`coin2` are the BASE names from argparse (e.g. "BTC"/"SOL"); the
        builder needs the resolved exchange symbols, so pass self.coin1/self.coin2
        (set from config_object.coin_param["pair_lead_lag"]) rather than the arguments.

        hist_replay: frozen-bundle builder with per-unit LOCAL ids (keeps the recorded
        matching fixtures valid). LIVE: parameters AND parent ids come from
        submodel_parameters (is_live = 1) — the DB is the single source of truth; the
        live host does not need the frozen bundle, and artifact paths come from the
        client config's sleeve_config.
        """
        if self.hist_replay_dir:
            self.parameter_dict = build_pair_lead_lag_parameter_dict(self.coin1, self.coin2)
            ## Replay never reaches the DB dumps, and the recorded matching fixtures
            ## were produced with exec_type=None — keep that surface unchanged.
            self.exec_type_dict = {}
        else:
            db_rows = self.get_db_strat_params("pair_lead_lag")
            self.parameter_dict = build_pair_lead_lag_parameter_dict_from_db(
                db_rows, self.coin1, self.coin2,
                sleeve_paths=config_object.sleeve_config["pair_lead_lag"])
            ## live_signals.execution_type is NOT NULL and PAIR_LEAD_LAG_PARAM_TUPLE
            ## carries no exec_type field — keep the DB rows' value per parent id for
            ## the warm-up dump and the per-minute live_signals writes.
            self.exec_type_dict = {pid: int(mp["exec_type"]) for pid, mp in db_rows.items()}
        
    def prepare_init_data(self, start_time, base_coin1, base_coin2):
        
        # end_date = dt.datetime.now().replace(second=0, microsecond=0) - dt.timedelta(minutes=config_object.init_timedelta)
        # end_date = start_time
        # start_date = end_date - dt.timedelta(days=config_object.lookback_days)
        
        # ###
        if not self.hist_replay_dir:
            self.df1 = self.get_init_data(coin=self.coin1, start_time=start_time, ffill=self.ffill)
            self.df2 = self.get_init_data(coin=self.coin2, start_time=start_time, ffill=self.ffill)        

        else:
            self.load_init_hist_data()

        ##
        inst_status_df = get_inst_status_df()

        ##
        # ## Remove data between 17:00:00 and 17:59:00
        # self.df1 = self.df1.drop(self.df1.between_time('17:00:00', '17:59:00').index)
        # self.df2 = self.df2.drop(self.df2.between_time('17:00:00', '17:59:00').index)

        # ## CONVERT BACK TO UTC
        # self.df1.index = self.df1.index.tz_convert("UTC")
        # self.df2.index = self.df2.index.tz_convert("UTC")

        ##
        self.comb_df = self.df1.merge(self.df2, left_index=True, right_index=True, suffixes=('_1', '_2'), how='outer')

        self.comb_df['open'] = self.comb_df['open_1']/self.comb_df['open_2']   
        self.comb_df['close'] = self.comb_df['close_1']/self.comb_df['close_2']
        self.comb_df['volume'] = self.comb_df['volume_1']+self.comb_df['volume_2']
        self.comb_df['high'] = np.max(self.comb_df[["open", "close"]], axis=1)
        self.comb_df['low'] = np.min(self.comb_df[["open", "close"]], axis=1)

        ###
        self.comb_df = self.comb_df.fillna(method='ffill')
        self.comb_df = self.comb_df.fillna(method='bfill')
        
        ###
        init_signal_gen = GetPairsLeadLagSignal(tup=None, df1=self.df1, df2=self.df2, comb_df=self.comb_df, curr_time=start_time, process_name=self.process_name)

        ## TESTING HOOK: when PAIR_LEAD_LAG_DUMP_PID is set, enable debug so create_debug_df
        ## writes ./numpy_pandas_matching/pair_lead_lag/{coin1}_{coin2}/{parent_stratid}_numpy.parquet
        ## for every bucket (create_debug_df makes the subdirectory itself).
        import os as _os
        if _os.environ.get("PAIR_LEAD_LAG_DUMP_PID"):
            init_signal_gen.debug = True
            _os.makedirs("./numpy_pandas_matching", exist_ok=True)

        for parent_id, tup in self.parameter_dict.items():
            init_signal_gen._update_tup(tup)

            self.trading_model_dict[parent_id] = PAIR_LEAD_LAG_PARENT_ID_STRAT_OBJ(tup, logger_name=f"signal_gen_{self.process_name}")
            self.trading_model_dict[parent_id].initialize(self.df1, self.df2, self.comb_df, inst_status_df, start_time, self.process_name, signal_generator=init_signal_gen)

        self.trading_model_dict[parent_id].clear_data_cache()

        ## Trim self.df1/df2/comb_df to the kernel's last-processed bar. Without
        ## this, live's first 2 incremental minutes (= the bars batch's trim-2
        ## dropped) see self.df*.iloc[-1] = init.index[-1] (2 bars AHEAD of
        ## last_processed_ts), and the kernel's `same_close*` / `close_ffill`
        ## inputs read from the wrong minute. last_processed_ts is consistent
        ## across all buckets (same trim, same arrays) so reading from any one
        ## trading_model is safe.
        _last_proc_ts = self.trading_model_dict[parent_id].last_processed_ts
        self.df1 = self.df1.loc[:_last_proc_ts]
        self.df2 = self.df2.loc[:_last_proc_ts]
        self.comb_df = self.comb_df.loc[:_last_proc_ts]

        ## TESTING HOOK: exit before the DB/socket-bound post-init blocks below.
        if _os.environ.get("PAIR_LEAD_LAG_DUMP_PID"):
            import sys as _sys
            print(f"[PAIR_LEAD_LAG_DUMP] per-bucket parquets written to "
                  f"./numpy_pandas_matching/pair_lead_lag/{self.coin1}_{self.coin2}/", flush=True)
            _sys.exit(0)

        ##
        if len(init_signal_gen.dump_pandas_ls) and self.hist_replay_dir is None:
            hist_signal_df = pd.concat(init_signal_gen.dump_pandas_ls)
            hist_signal_df.index = hist_signal_df.index.tz_localize("UTC")
            hist_signal_df.index.name = "signal_floor_time"
            hist_signal_df = hist_signal_df.reset_index()
            hist_signal_df["signal_time"] = dt.datetime.now().replace(tzinfo=timezone("UTC"))
            ## live_signals.case_num / execution_type are NOT NULL. QR31 publishes no
            ## case (case_num=0, same convention as the backtest producer); the
            ## execution_type of each stream comes from its submodel_parameters row.
            hist_signal_df["case_num"] = 0
            hist_signal_df["execution_type"] = hist_signal_df["parent_trading_model"].map(self.exec_type_dict)
            assert not hist_signal_df["execution_type"].isna().any(), \
                "execution_type missing for some parent ids — check submodel_parameters exec_type"
            ## psycopg2 cannot adapt numpy integer scalars and itertuples() yields
            ## them — box the integer columns to Python ints (floats/str/Timestamp
            ## adapt natively). `signal` is float64 {0.0, 1.0} off the kernel and the
            ## column is smallint, so the int cast is exact.
            for _c in ("signal_id", "parent_trading_model", "signal",
                       "case_num", "execution_type", "is_tp_sl"):
                hist_signal_df[_c] = hist_signal_df[_c].astype("int64").astype(object)
            self.dump_hist_signal_df(hist_signal_df)

        ###
        self.df1 = self.df1.fillna(method='ffill')
        self.df2 = self.df2.fillna(method='ffill')
    
    ##
    def load_init_hist_data(self):
        ## Warm-up _input files come from the SAME directory as the replay feed.
        ## Mirrors pair_momentum: hardcoding ./backtest_codes/data/ (shared by every
        ## sleeve) let a concurrent run swap the warm-up window out from under an
        ## in-flight test. Honouring hist_replay_dir is safe with no live-mode risk:
        ## load_init_hist_data() is only reached from the replay branch of
        ## prepare_init_data, so hist_replay_dir is never None here. The env var keeps
        ## the escape hatch the sibling sleeves use.
        import os as _os
        init_dir = (_os.environ.get("PAIR_LEAD_LAG_INIT_DIR")
                    or self.hist_replay_dir or "./backtest_codes/data")

        self.df1 = pd.read_parquet(f"{init_dir}/{self.coin1}_input.parquet")
        if self.df1.index.tz is None:
            self.df1.index = self.df1.index.tz_localize("UTC")

        self.df2 = pd.read_parquet(f"{init_dir}/{self.coin2}_input.parquet")
        if self.df2.index.tz is None:
            self.df2.index = self.df2.index.tz_localize("UTC")
        

    ## NOTE: no get_hist_replay_parent_ids(). The dead copies the sibling sleeves carried
    ## (nothing ever called them) were deleted when 4xxxxxxx stopped being the legacy
    ## pair_breakout band and became pair_momentum's parameter band. Replay subscribes by
    ## symbol (market_data_replay), never by parent id.

        

