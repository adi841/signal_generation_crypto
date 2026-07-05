
import os

if not os.environ.get("CLIENT_ID"):
    # ---- Local / test mode: skip S3/DB, load minimal config from local JSON. ----
    import json

    LOCAL_JSON = os.environ.get(
        "LOCAL_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "local_config.json"),
    )
    with open(LOCAL_JSON) as _f:
        _cfg = json.load(_f)

    class _NoopLogger:
        def get_logger(self, name, level):
            # Dev mode (local_config.json). Silently dropping errors/criticals
            # makes diagnosis impossible; surface them on stderr so the tail of
            # live.py's redirected log shows them.
            if level in ("error", "critical"):
                import sys as _sys
                def _emit(msg):
                    print(f"[{level}] {name}: {msg}", file=_sys.stderr, flush=True)
                return _emit
            def _emit(msg): pass
            return _emit

    class configObject:
        def __init__(self, c):
            self.time_zone          = c["time_zone"]
            self.offset             = c["offset"]
            self.deque_maxlen       = c.get("deque_maxlen", 1440)
            ## per-sleeve unit tables: coin_param[<sleeve>][<unit_key>] — same shape as
            ## the client config's signal_gen_config. Hard-indexed: a missing key is a
            ## config bug and must raise, not default.
            self.coin_param         = c["coin_param"]
            self.sleeve_config      = c["sleeve_config"]
            self.fixed_cost_dict    = c.get("fixed_cost_dict", {})
            self.slippage_hlc3_dict = c.get("slippage_hlc3_dict", {})
            self.masterlog          = _NoopLogger()
            self.db_pool            = None

    config_object = configObject(_cfg)

else:
    # ---- Prod mode (unchanged) ----
    import hjson
    import asyncpg
    import asyncio

    from shared_codes.utils.config_utils import get_current_branch_env, get_s3_config, modelLogger, get_client_asset_class
    from shared_codes.utils.config_utils import get_db_params


    async def create_db_pool():
        ##
        db_params = get_db_params()
        db_pool = await asyncpg.create_pool(min_size=db_params.min_conn, max_size=db_params.max_conn, max_inactive_connection_lifetime=db_params.max_inactive_connection_lifetime, max_queries=db_params.max_queries,
                                            host=db_params.host, database=db_params.database, user=db_params.user, password=db_params.password, port=db_params.port)

        return db_pool


    class configObject(object):

        def __init__(self) -> None:

            ##
            self.client_id = os.environ["CLIENT_ID"]
            self.client_name, self.asset_class = get_client_asset_class(self.client_id)

            config_name = f"{self.client_name}_som_config"
            hjson_data, _ = get_s3_config(config_name)
            self.parent_config = hjson.loads(hjson_data)
            self.config = self.parent_config['signal_gen_config']

            ####
            self.zmq_data = self.config['zmq_data']
            self.lookback_days = self.config['lookback_days']
            self.init_timedelta = self.config['init_timedelta']
            self.deque_maxlen = self.config['deque_maxlen']
            self.recover_time_delta = self.config['recover_time_delta']
            self.time_zone = self.config['time_zone']
            self.offset = self.config['offset']
            self.max_data_retry = self.config["max_data_retry"]
            self.data_retry_interval = self.config["data_retry_interval"]

            ## Parameter config — the legacy sub-keys (bb_lookback_candle, rsi_tp_mult,
            ## atr_param, market_signal_ma, signal_timeperiod_mapping) were removed with
            ## the legacy strategies; the new sleeves consume only costs + slippage.
            self.param_config = self.config['parameter_config']
            self.slippage_hlc3_dict = self.param_config['slippage_hlc3_dict']
            self.fixed_cost_dict = self.param_config['fixed_cost']

            ## Per-sleeve unit tables (coin_param[<sleeve>][<unit_key>]) — the legacy
            ## pair_breakout/pair_reversal blocks were removed with the new sleeves.
            self.coin_param = self.config['coin_param']
            self.sleeve_config = self.config['sleeve_config']

            ## signal_mismatch_config load removed: no code reads it and magha's
            ## config does not carry the key.

            run_dest = f"signal_gen_som_{self.client_name}"
            self.masterlog = modelLogger(run_dest)
            self.db_pool = asyncio.run(create_db_pool())

            ## Backtest params
            hjson_global_setting, _ = get_s3_config("global_settings")
            self.global_setting = hjson.loads(hjson_global_setting)
            self.backtest_duration_minutes = self.global_setting["misc_global_static_config"]['backtest_duration_minutes']
            self.backtest_sleep_minutes = self.global_setting["misc_global_static_config"]['backtest_sleep_minutes']
            self.socket_config = self.global_setting["socket_config"]

            ##
            hjson_global_variables, _ = get_s3_config("global_variables")
            self.global_variables = hjson.loads(hjson_global_variables)
            self.global_variables = self.global_variables["CRYPTO"]

            ##
            hjson_data, _ = get_s3_config("binance_config", None)
            binance_dict = hjson.loads(hjson_data)
            self.symbol_mapping = dict(binance_dict["symbol_mapping"])

            if get_current_branch_env() == "main":
                assert self.init_timedelta <= 10, "Init timedelta must be less than 10"

    ##
    config_object = configObject()
