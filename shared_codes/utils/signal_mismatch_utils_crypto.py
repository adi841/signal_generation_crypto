"""
This module provides utility functions to help with BACKTEST vs LIVE trading signal mismatches.
"""

import pandas as pd
from functools import partial


# def get_market_timing_series(underlying_ls, start_datetime, end_datetime):
#     """
#     Get the market timing series for the given underlying between start_datetime and end_datetime
#     """
#     start_datetime = pd.to_datetime(start_datetime) - pd.Timedelta(days=5)
#     end_datetime = pd.to_datetime(end_datetime) + pd.Timedelta(days=5)
#     ts_ = pd.date_range(start=start_datetime, end=end_datetime, freq='1min', tz='UTC')
#     df_status = pd.DataFrame(columns=underlying_ls, index=ts_)

#     for underlying in underlying_ls:
#         df_status[underlying] = df_status.index.map(lambda x: is_market_open(underlying, x))
    
#     ##
#     df_status.index = df_status.index.tz_localize(None)  # Remove timezone information
#     return df_status

###
def get_underlying_start_end_timestamp(trade_datetime, required_market_open_minutes, reverse=False):
    """
    Get the start and end timestamp of the underlying for the given trade_datetime.
    """

    if reverse:
        start_dt = trade_datetime - pd.Timedelta(minutes=required_market_open_minutes)
        end_dt = trade_datetime
    else:
        start_dt = trade_datetime
        end_dt = trade_datetime + pd.Timedelta(minutes=required_market_open_minutes)

    return start_dt, end_dt


####
def match_backtest_vs_live_execution_v2(backtest_df, live_exec_df, param_dict, entry_parent_id_to_primary_parent_id_map, base_symbol_mapping, processed_dict, 
                                        required_market_open_minutes, mismatch_type, config_object, bk_df_max_indx):

    # if len(backtest_df) == 0:
    #     return live_exec_df

    # if len(live_exec_df) == 0:
    #     print("No live execution data found.")
    #     return backtest_df # Fix this 

    ## Matching bk df with live execution df
    msg_ = f"Matching bk vs live trades for {mismatch_type}"
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)

    assert mismatch_type in ["EXACT_MISMATCH", "NON_EXACT_MISMATCH"], "Mismatch type must be either EXACT_MISMATCH or NON_EXACT_MISMATCH"

    mismatch_df = pd.DataFrame()
    match_df = pd.DataFrame()
    for tup in backtest_df.itertuples():
        if hasattr(tup, "underlying"):
            underlying = tup.underlying
        else:
            underlying = base_symbol_mapping[tup.asset]

        ts_ = tup.Index
        signal_diff = tup.signal_diff
        parent_id = tup.parent_id
        signal = int(tup.signal)
        param_dict_parent_id_bk = entry_parent_id_to_primary_parent_id_map.get(parent_id, parent_id)
        asset_str = tup.asset_str
        order_tag = tup.order_tag
        raw_parent_id = tup.raw_parent_id

        ## Reverse is not needed because the trade_datetime is already in the past.
        # start_dt, end_dt = get_underlying_start_end_timestamp(ts_, reverse=True, required_market_open_minutes=required_market_open_minutes)
        start_dt, end_dt = get_underlying_start_end_timestamp(ts_, required_market_open_minutes=required_market_open_minutes)

        if abs(signal) == 0:
            if tup.case >= 10_000:  # take profit
                entry_exit = "take_profit"
            elif tup.case <= -800:  # stop loss
                entry_exit = "stop_loss"
            else:  # normal exit
                entry_exit = "normal_exit"
        else:
            entry_exit = "entry"

        ##
        if start_dt is None or end_dt is None:
            raise ValueError(f"Skipping {underlying} for timestamp {ts_} as market is not open.")
            continue

        ##
        if len(live_exec_df) > 0:
            exec_df = live_exec_df[(live_exec_df.index >= start_dt) & (live_exec_df.index <= end_dt) & (live_exec_df["parent_id"] == parent_id)]
        else:
            exec_df = pd.DataFrame()

        ##
        parent_id_str = str(parent_id)
        backtest_mismatch_dict = processed_dict["BACKTEST_LIVE_MISMATCH"][mismatch_type]["BACKTEST_MISMATCH"]
        
        if parent_id_str in backtest_mismatch_dict:
            processed_dict_ts = backtest_mismatch_dict[parent_id_str]["timestamp"]
            if pd.Timestamp(processed_dict_ts) >= ts_.floor("s"):
                msg_ = f"{mismatch_type}:BK Skipping {parent_id} for timestamp {ts_} as it is already processed."
                config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)
                continue

            else:
                msg_ = f"{mismatch_type}:BK Processing {parent_id} for timestamp {ts_} as it is not processed yet."
                config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)
                str_ts = ts_.floor("s").strftime("%Y-%m-%d %H:%M:%S")
                backtest_mismatch_dict[parent_id_str] = {"timestamp": str_ts, "signal": signal, "asset": asset_str}

        else:
            str_ts = ts_.floor("s").strftime("%Y-%m-%d %H:%M:%S")
            backtest_mismatch_dict[parent_id_str] = {"timestamp": str_ts, "signal": signal, "asset": asset_str}

        if (len(exec_df) > 0) and (order_tag in exec_df["order_tag"].tolist()):
            pass

        else:
            tmp = live_exec_df[live_exec_df["parent_id"] == parent_id]
            tail_tmp = tmp.tail(3).to_string()
            msg_ = f"{mismatch_type}: No live execution data found for {parent_id} for timestamp {ts_}. Start time: {start_dt}, End time: {end_dt}. Tail of live exec df: \n{tail_tmp}"
            config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)

            ##
            is_second_entry = True if raw_parent_id != parent_id else False
            mismatch_df = pd.concat([mismatch_df, pd.DataFrame({"timestamp": ts_, "parent_id": parent_id, "signal": signal, "asset": asset_str, "param_dict": str(param_dict[param_dict_parent_id_bk]), "entry_exit": entry_exit, 
                                                            "live/bk": "bk", "order_tag": order_tag, "raw_parent_id": raw_parent_id, "is_second_entry": is_second_entry}, index=[0])], ignore_index=True)

    ## Matching live messages with backtest df
    for tup in live_exec_df.itertuples():
        parent_id = tup.parent_id
        if hasattr(tup, "underlying"):
            underlying = tup.underlying
        else:
            underlying = base_symbol_mapping[tup.asset]
            
        ts_ = tup.Index
        entry_exit = tup.entry_type
        asset_str = tup.asset_str
        order_tag = tup.order_tag
        param_dict_parent_id_live = entry_parent_id_to_primary_parent_id_map.get(parent_id, parent_id)
        exec_purpose = tup.exec_purpose

        start_dt, end_dt = get_underlying_start_end_timestamp(ts_, reverse=True, required_market_open_minutes=required_market_open_minutes)
        start_dt2, end_dt2 = get_underlying_start_end_timestamp(ts_, required_market_open_minutes=required_market_open_minutes)

        ##
        if start_dt is None or end_dt is None:
            raise ValueError(f"Skipping {underlying} for timestamp {ts_} as market is not open.")
        
        ##
        start_dt = min(start_dt, start_dt2)
        end_dt = max(end_dt, end_dt2)
    
        ##
        parent_id_str = str(parent_id)
        live_mismatch_dict = processed_dict["BACKTEST_LIVE_MISMATCH"][mismatch_type]["LIVE_MISMATCH"]
        
        if parent_id_str in live_mismatch_dict:
            processed_dict_ts = live_mismatch_dict[parent_id_str]["timestamp"]
            if pd.Timestamp(processed_dict_ts) >= ts_.floor("s"):
                msg_ = f"{mismatch_type}:LV Skipping {parent_id} for timestamp {ts_} as it is already processed."
                config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)
                continue

            else:
                msg_ = f"{mismatch_type}:LV Processing {parent_id} for timestamp {ts_} as it is not processed yet."
                config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)
                str_ts = ts_.floor("s").strftime("%Y-%m-%d %H:%M:%S")
                live_mismatch_dict[parent_id_str] = {"timestamp": str_ts, "signal": tup.signal, "asset": asset_str}
            
        else:
            str_ts = ts_.floor("s").strftime("%Y-%m-%d %H:%M:%S")
            live_mismatch_dict[parent_id_str] = {"timestamp": str_ts, "signal": tup.signal, "asset": asset_str}
        
        ##
        if len(backtest_df) > 0:
            bk_df = backtest_df[(backtest_df.index >= start_dt) & (backtest_df.index <= end_dt) & (backtest_df["parent_id"] == parent_id)]
        else:
            bk_df = pd.DataFrame()

        if len(bk_df) > 0 and order_tag in bk_df["order_tag"].tolist():
            match_df = pd.concat([match_df, pd.DataFrame({"timestamp": ts_, "parent_id": parent_id, "signal": tup.signal, "asset": asset_str, "param_dict": str(param_dict[param_dict_parent_id_live]), 
            "entry_exit": entry_exit, "live/bk": "live", "exec_purpose": tup.exec_purpose}, index=[0])], ignore_index=True)

        else:
            if ts_ > bk_df_max_indx:
                msg_ = f"Skipping {tup} for timestamp {ts_} as it is after the last index of the backtest df. bk_df_max_indx: {bk_df_max_indx}"
                config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)
                continue

            is_second_entry = True if "2nd_entry" in exec_purpose else False

            ##
            mismatch_df = pd.concat([mismatch_df, pd.DataFrame({"timestamp": ts_, "parent_id": parent_id, "signal": tup.signal, "asset": asset_str, "param_dict": str(param_dict[parent_id]), "entry_exit": entry_exit, 
                                                                "live/bk": "live", "order_tag": order_tag, "is_second_entry": is_second_entry}, index=[0])], ignore_index=True)
        
    ##
    return mismatch_df, match_df, processed_dict



###
def get_all_live_messages(start_datetime, end_datetime, use_grouping_table, st_days_td=2, conn=None, cursor=None, signal_mismatch_config: dict=None, config_object=None):
    """
    st_days_td: int, number of days to look back for execution logs.
    
    Get all messages from the database between start_datetime and end_datetime. 

    If a quoting execution is started before the start_datetime, it will be ignored.
        Examples:
            1. Single Asset Breakout: Entry and normal exit
            2. Single Asset Reversal: Entry and normal exit (but not the second entry)
            3. Pair Breakout: Entry, TP, SL, normal exit (since we are quoting in all of them)
            4. Pair Reversal: Entry, TP, Second Entry, SL, normal exit (since we are quoting in all of them)
    """

    assert use_grouping_table in [True, False], "use_grouping_table must be either True or False"

    exec_logs_start_time = pd.to_datetime(start_datetime) - pd.Timedelta(days=st_days_td)
    exec_logs_end_time = pd.to_datetime(end_datetime)

    msg_ = f"Getting all live messages from {exec_logs_start_time} to {exec_logs_end_time}"
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)

    all_live_message_df = pd.DataFrame()

    # query = f"""
    #     SELECT timestamp, child_trading_model, symbol, total_quantity, executed_quantity, price , avg_price, remaining_quantity, side, signal_id , order_state, order_id, exec_purpose  
    #     FROM execution_logs
    #     WHERE timestamp >= '{exec_logs_start_time}' AND timestamp <= '{exec_logs_end_time}' AND child_trading_model > 0
    #     ORDER BY timestamp ASC;
    # """

    # cursor.execute(query)
    # rows = cursor.fetchall()

    # all_live_message_df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    # all_live_message_df["timestamp"] = pd.to_datetime(all_live_message_df["timestamp"])
    # all_live_message_df.set_index("timestamp", inplace=True)
    # if all_live_message_df.index.tz is not None:
    #     all_live_message_df.index = all_live_message_df.index.tz_localize(None)
    
    ## CLEAN THE `ALL_LIVE_MESSAGE_DF` DATAFRAME
    """
    For all the quoting messages, we will ignore the parents_ids-trades that started quoting before the start_datetime.
    
    1. single asset:
        - TP/SL/SE, we will only keep the filled messages. 
        - For Entry/EXIT, we will keep the message when the quoting began. 
    
    2. Pairs: (Since we are quoting in all types of signals)
        - TP/SL/SE, we will keep the timestamp when the quoting began. 
        - For Entry/EXIT, we will keep the message when the quoting began
    """

    ###
    # Exec purpose of trades for which we are quoting
    all_quoting_exec_purpose = get_all_quoting_exec_purpose(signal_mismatch_config=signal_mismatch_config)

    ##
    # quoting_exec_logs = all_live_message_df[all_live_message_df["exec_purpose"].isin(all_quoting_exec_purpose)].copy(deep=True)
    ## We have to take only filled orders IDs and then use the first entry of each filled order ID.
    # filled_order_ids = quoting_exec_logs[quoting_exec_logs["executed_quantity"].abs() != 0]["order_id"].unique()
    # quoting_exec_logs = quoting_exec_logs[quoting_exec_logs["order_id"].isin(filled_order_ids)]

    # quoting_exec_logs["tmp_str1"] = quoting_exec_logs["child_trading_model"].astype(str) + "_" + quoting_exec_logs["exec_purpose"].astype(str) + "_" + quoting_exec_logs["signal_id"].astype(str)
    # quoting_exec_logs = quoting_exec_logs.sort_index()
    # quoting_exec_logs = quoting_exec_logs.drop_duplicates(subset=["tmp_str1"], keep="first")

    ##
    ## non-quoting exec logs
    # non_quoting_exec_logs = all_live_message_df[~all_live_message_df["exec_purpose"].isin(all_quoting_exec_purpose)].copy(deep=True)
    # non_quoting_exec_logs = non_quoting_exec_logs[non_quoting_exec_logs["executed_quantity"].abs() != 0]
    # non_quoting_exec_logs["tmp_str1"] = non_quoting_exec_logs["child_trading_model"].astype(str) + "_" + non_quoting_exec_logs["exec_purpose"].astype(str) + "_" + non_quoting_exec_logs["signal_id"].astype(str)
    # non_quoting_exec_logs = non_quoting_exec_logs.sort_index()
    # non_quoting_exec_logs = non_quoting_exec_logs.drop_duplicates(subset=["tmp_str1"], keep="last")

    live_unique_exec_logs = pd.DataFrame()
    # live_unique_exec_logs = pd.concat([quoting_exec_logs, non_quoting_exec_logs], axis=0)
    # live_unique_exec_logs = live_unique_exec_logs.sort_index()
    # live_unique_exec_logs = live_unique_exec_logs[(live_unique_exec_logs.index >= start_datetime) & (live_unique_exec_logs.index <= end_datetime)]

    ###
    if use_grouping_table:
        query = f"""
            SELECT * FROM execution_report_grouping
            WHERE timestamp >= '{start_datetime}' AND timestamp <= '{end_datetime}' AND avg_exec_price != 0
            ORDER BY timestamp ASC;
        """

    else:
        query = f"""
            SELECT * FROM execution_report
            WHERE timestamp >= '{start_datetime}' AND timestamp <= '{end_datetime}' AND signal_price != 0
            ORDER BY timestamp ASC;
        """

    msg_ = f"Getting executed df from {start_datetime} to {end_datetime}. Grouping table: {use_grouping_table}"
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)

    cursor.execute(query)
    executed_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    executed_df["timestamp"] = pd.to_datetime(executed_df["timestamp"])
    executed_df.set_index("timestamp", inplace=True)
    if executed_df.index.tz is not None:
        executed_df.index = executed_df.index.tz_localize(None)
    
    ##
    return live_unique_exec_logs, executed_df


##
def get_all_quoting_exec_purpose(signal_mismatch_config: dict) -> set:
    """
    Get all the quoting exec purpose from the config object.
    """
    exec_purpose_mapping = get_exec_purpose_mapping(signal_mismatch_config)
    all_quoting_exec_purpose = set()
    for key, value in exec_purpose_mapping.items():
        if value["is_quoting"]:
            all_quoting_exec_purpose.add(value["exec_purpose"])
    
    ###
    return all_quoting_exec_purpose


###
def get_exec_purpose_mapping(signal_mismatch_config: dict) -> dict:
    """
    Get the exec purpose mapping from the config object.
    """
    exec_purpose_mapping = {}
    for asset_type, asset_type_dict in signal_mismatch_config.items():
        if asset_type not in ["single_asset", "pairs"]:
            continue
        for strategy_name, strategy_name_dict in asset_type_dict.items():
            for execution_type, execution_type_dict in strategy_name_dict.items():
                exec_purpose_mapping[execution_type_dict["exec_purpose"]] = execution_type_dict

    ##
    return exec_purpose_mapping


def get_is_exact(param_dict, signal_mismatch_config: dict, param_dict_parent_id, trade_type=None, exec_purpose=None):
    ## Adding a check. Both trade_type and exec_purpose cannot be None.
    assert (trade_type is not None) or (exec_purpose is not None), "Either trade_type or exec_purpose must be provided."

    if exec_purpose is not None:
        exec_purpose_mapping = get_exec_purpose_mapping(signal_mismatch_config=signal_mismatch_config)
        return exec_purpose_mapping[exec_purpose]["match_type"]

    ###
    if param_dict[param_dict_parent_id]["asset_type"] == "pairs":
        if param_dict[param_dict_parent_id]["strategy_name"] in ["breakout1", "breakout2", "breakout3", "breakout4", "qbreakout1", "qbreakout2", "qbreakout3", "qbreakout4"]:
            is_exact = signal_mismatch_config["pairs"]["breakout"][trade_type]["match_type"]
        elif param_dict[param_dict_parent_id]["strategy_name"] in ["reversal1", "reversal2", "qreversal1", "qreversal2"]:
            is_exact = signal_mismatch_config["pairs"]["reversal"][trade_type]["match_type"]
        else:
            raise ValueError(f"Invalid strategy name {param_dict[param_dict_parent_id]['strategy_name']} for parent_id {param_dict_parent_id}")
    
    elif param_dict[param_dict_parent_id]["asset_type"] == "single":
        if param_dict[param_dict_parent_id]["strategy_name"] in ["breakout1", "breakout2", "breakout3", "breakout4", "qbreakout1", "qbreakout2", "qbreakout3", "qbreakout4"]:
            is_exact = signal_mismatch_config["single_asset"]["breakout"][trade_type]["match_type"]
        elif param_dict[param_dict_parent_id]["strategy_name"] in ["reversal1", "reversal2", "qreversal1", "qreversal2"]:
            is_exact = signal_mismatch_config["single_asset"]["reversal"][trade_type]["match_type"]
        else:
            raise ValueError(f"Invalid strategy name {param_dict[param_dict_parent_id]['strategy_name']} for parent_id {param_dict_parent_id}")
    
    ##
    return is_exact


###
def get_exact_non_exact_live_exec(all_live_message_df, executed_df, parent_child_dict, param_dict, base_symbol_mapping, signal_mismatch_config, config_object):
    msg_ = f"Getting exact and non exact live exec"
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)

    _child_parent_dict = {v[0]: k for k, v in parent_child_dict.items()}

    ####
    exec_purpose_mapping = get_exec_purpose_mapping(signal_mismatch_config=signal_mismatch_config)
    exact_df = pd.DataFrame()
    non_exact_df = pd.DataFrame()
    for tup in executed_df.itertuples():
        symbol_ = tup.symbol
        end_ts = tup.Index
        signal_id = tup.signal_id
        # qty_sign = 1 if tup.side.lower() == "buy" else -1
        exec_purpose = tup.exec_purpose
        asset_str = tup.asset_str
        order_tag = tup.order_tag

        parent_id = _child_parent_dict[tup.child_trading_model]

        ##
        entry_exit = "entry" if exec_purpose_mapping[exec_purpose]["is_entry"] else "exit"
        is_exact = exec_purpose_mapping[exec_purpose]["match_type"]

        start_indx = tup.Index
        if exec_purpose_mapping[exec_purpose]["is_entry"]:
            if tup.side.lower() == "buy":
                signal = 1
            else:
                signal = -1
        else:
            signal = 0
        
        ##
        tmp_df = pd.DataFrame({"timestamp": start_indx, "signal": signal, "parent_id": parent_id, "entry_type": entry_exit, "signal_id": signal_id, "asset": symbol_, "exec_purpose": exec_purpose, 
                            "asset_str": asset_str, "order_tag": order_tag}, index=[0])

        if is_exact:
            exact_df = pd.concat([exact_df, tmp_df], ignore_index=True)
        else:
            non_exact_df = pd.concat([non_exact_df, tmp_df], ignore_index=True)

    ###
    if len(exact_df):
        exact_df = exact_df.set_index("timestamp")
        exact_df = exact_df.sort_index()    

    if len(non_exact_df):
        non_exact_df = non_exact_df.set_index("timestamp")
        non_exact_df = non_exact_df.sort_index()
    
    ##
    executed_df["parent_id"] = executed_df["child_trading_model"].map(lambda x: _child_parent_dict[x])
    return exact_df, non_exact_df, executed_df


def get_asset_str_bk(parent_id, param_dict, param_parent_id_dict):
    param_parent_id = param_parent_id_dict.get(parent_id, parent_id)
    if param_dict[param_parent_id]["asset_type"] == "pairs":
        return param_dict[param_parent_id]["coin1"] + "_" + param_dict[param_parent_id]["coin2"]
    elif param_dict[param_parent_id]["asset_type"] == "single":
        return param_dict[param_parent_id]["coin"]

def get_asset_str_live(child_id, param_dict, param_parent_id_dict, parent_child_dict):
        _child_parent_dict = {v[0]: k for k, v in parent_child_dict.items()}
        parent_id = _child_parent_dict[child_id]
        return get_asset_str_bk(parent_id=parent_id, param_dict=param_dict, param_parent_id_dict=param_parent_id_dict)

#####
def match_backtest_signals_live_execution_crypto(bk_trades_df, param_dict, start_datetime, end_datetime, processed_dict, parent_child_dict, base_symbol_mapping,
                                          signal_mismatch_config, conn, cursor, required_market_open_minutes, config_object, use_grouping_table):
    ####
    """
    config_object: for logging purposes. `config_object.masterlog.get_logger("signal_match_live_bk", "info")(msg_)`

    Parent to Parent mapping. For example, in case of pairs or sa reversal, 1 strategy may have multiple parent ids, therefore we need to map them.
        Cases:
            1. SA Breakout: No change
            2. SA Reversal: For entry, we need to use both the parent ids, but for exit, we need to use only one parent id.
            3. Pair breakout: We only need to use one parent id, we can safely ignore (ID + 1).
            4. Pair reversal: For entry, we need to use 2 parent ids (ID and ID + 2). For exit, we need to use only one parent id (ID).
    """

    assert use_grouping_table in [True, False], "use_grouping_table must be either True or False"

    ##
    if "BACKTEST_LIVE_MISMATCH" not in processed_dict:
        processed_dict["BACKTEST_LIVE_MISMATCH"] = {
            "EXACT_MISMATCH": {
                "BACKTEST_MISMATCH": {},
                "LIVE_MISMATCH": {}
            },
            "NON_EXACT_MISMATCH": {
                "BACKTEST_MISMATCH": {},
                "LIVE_MISMATCH": {}
            }
        }

    ###
    parent_ids_ls = [] # These are the parent ids that we need to use. 
    entry_parent_id_to_primary_parent_id_map = {} # This is the mapping second entry parent id to first entry parent id.
    reversal_parent_id_set = set()
    for parent_id, dict_ in param_dict.items():
        if dict_["asset_type"] == "pairs":
            if dict_["strategy_name"] in ["breakout1", "breakout2", "breakout3", "breakout4", "qbreakout1", "qbreakout2", "qbreakout3", "qbreakout4"]:
                parent_ids_ls.append(parent_id)

            elif dict_["strategy_name"] in ["reversal1", "reversal2", "qreversal1", "qreversal2"]:
                parent_ids_ls.append(parent_id)
                parent_ids_ls.append(parent_id + 2)
                reversal_parent_id_set.add(parent_id + 2)
                entry_parent_id_to_primary_parent_id_map[parent_id + 2] = parent_id

        elif dict_["asset_type"] == "single":
            if dict_["strategy_name"] in ["breakout1", "breakout2", "breakout3", "breakout4", "qbreakout1", "qbreakout2", "qbreakout3", "qbreakout4"]:
                parent_ids_ls.append(parent_id)

            elif dict_["strategy_name"] in ["reversal1", "reversal2", "qreversal1", "qreversal2"]:
                parent_ids_ls.append(parent_id)
                parent_ids_ls.append(parent_id + 1)
                reversal_parent_id_set.add(parent_id + 1)
                entry_parent_id_to_primary_parent_id_map[parent_id + 1] = parent_id

    ###
    if len(bk_trades_df) > 0:
        bk_trades_df = bk_trades_df[bk_trades_df["parent_id"].isin(parent_ids_ls)]
        bk_trades_df["asset_str"] = bk_trades_df["parent_id"].map(lambda x: get_asset_str_bk(parent_id=x, param_dict=param_dict, param_parent_id_dict=entry_parent_id_to_primary_parent_id_map))

    exact_rows = []
    non_exact_rows = []
    for tup in bk_trades_df.itertuples():
        ##
        parent_id = tup.parent_id
        signal = tup.signal

        ##
        partial_is_exact = partial(get_is_exact, param_dict=param_dict, signal_mismatch_config=signal_mismatch_config, param_dict_parent_id=parent_id)

        ##
        param_dict_parent_id = entry_parent_id_to_primary_parent_id_map.get(parent_id, parent_id)
        if abs(signal) > 0: # entry signal:
            if parent_id in reversal_parent_id_set:
                is_exact = partial_is_exact(param_dict_parent_id=param_dict_parent_id, trade_type="second_entry")
            else:
                is_exact = partial_is_exact(param_dict_parent_id=param_dict_parent_id, trade_type="entry")
        
        elif signal == 0: # exit signal
            case_ = tup.case
            if case_ >= signal_mismatch_config["case_config"]["pft_take"] : # 10_000: # take profit
                is_exact = partial_is_exact(param_dict_parent_id=param_dict_parent_id, trade_type="pft_take")
            elif case_ <= signal_mismatch_config["case_config"]["stop_loss"]: # stop loss
                is_exact = partial_is_exact(param_dict_parent_id=param_dict_parent_id, trade_type="stop_loss")
            else: # normal exit
                is_exact = partial_is_exact(param_dict_parent_id=param_dict_parent_id, trade_type="normal_exit")
            
        else:
            raise ValueError(f"Invalid signal {signal} for parent_id {parent_id}")
        
        if is_exact:
            exact_rows.append(tup._asdict())
        else:
            non_exact_rows.append(tup._asdict())
    
    ###
    exact_df = pd.DataFrame()
    if len(exact_rows) > 0:
        exact_df = pd.DataFrame(exact_rows)
        exact_df["raw_parent_id"] = exact_df["parent_id"]
        exact_df["parent_id"] = exact_df["parent_id"].map(lambda x: entry_parent_id_to_primary_parent_id_map.get(x, x))  # Map to primary parent id. To avoid duplicate entries for exits.
        ## I want to drop duplicates based on (index, signal, parent_id) tuple. Like the combination of these 3 should be unique.
        exact_df = exact_df[~exact_df.duplicated(subset=["Index", "signal", "parent_id"], keep="last")]
        exact_df.set_index("Index", inplace=True)
    
    ##
    non_exact_df = pd.DataFrame()
    if len(non_exact_rows) > 0:
        non_exact_df = pd.DataFrame(non_exact_rows)
        non_exact_df["raw_parent_id"] = non_exact_df["parent_id"]
        non_exact_df["parent_id"] = non_exact_df["parent_id"].map(lambda x: entry_parent_id_to_primary_parent_id_map.get(x, x))  # Map to primary parent id. To avoid duplicate entries for exits.
        non_exact_df = non_exact_df[~non_exact_df.duplicated(subset=["Index", "signal", "parent_id"], keep="last")]
        non_exact_df.set_index("Index", inplace=True)

    ##
    # all_parent_ids = sorted([x for x in bk_trades_df["parent_id"].unique()])

    ###
    all_live_message_df, executed_df = get_all_live_messages(start_datetime=start_datetime, end_datetime=end_datetime, conn=conn, cursor=cursor, signal_mismatch_config=signal_mismatch_config, config_object=config_object, use_grouping_table=use_grouping_table)

    ##
    # all_live_message_df["symbol"] = all_live_message_df["symbol"].map(lambda x: micro_contracts_mapping[x] if x in micro_contracts_mapping else x)
    # all_live_message_df["asset_str"] = all_live_message_df["child_trading_model"].map(lambda x: get_asset_str_live(child_id=x, param_dict=param_dict, param_parent_id_dict=entry_parent_id_to_primary_parent_id_map, parent_child_dict=parent_child_dict))

    # executed_df["security_symbol"] = executed_df["security_symbol"].map(lambda x: micro_contracts_mapping[x] if x in micro_contracts_mapping else x)
    executed_df["symbol"] = executed_df["security_symbol"]
    executed_df["asset_str"] = executed_df["child_trading_model"].map(lambda x: get_asset_str_live(child_id=x, param_dict=param_dict, param_parent_id_dict=entry_parent_id_to_primary_parent_id_map, parent_child_dict=parent_child_dict))

    ###
    live_exact_df, live_non_exact_df, executed_df = get_exact_non_exact_live_exec(all_live_message_df=all_live_message_df, executed_df=executed_df, parent_child_dict=parent_child_dict, param_dict=param_dict, 
                                                                     base_symbol_mapping=base_symbol_mapping, signal_mismatch_config=signal_mismatch_config, config_object=config_object)
    
    ###
    # symbol_ls = bk_trades_df["asset"].unique().tolist()
    underlying_symbol_ls = set()
    parent_id_ls = bk_trades_df["parent_id"].unique().tolist() if len(bk_trades_df) > 0 else []
    for parent_id in parent_id_ls:
        param_dict_parent_id = entry_parent_id_to_primary_parent_id_map.get(parent_id, parent_id)
        if param_dict[param_dict_parent_id]["asset_type"] == "pairs":
            underlying_symbol_ls.add(param_dict[param_dict_parent_id]["coin1"])
            underlying_symbol_ls.add(param_dict[param_dict_parent_id]["coin2"])

        elif param_dict[param_dict_parent_id]["asset_type"] == "single":
            underlying_symbol_ls.add(param_dict[param_dict_parent_id]["coin"])

    ###
    exact_df, exact_df_max_indx = sort_df_by_index(exact_df)
    non_exact_df, non_exact_df_max_indx = sort_df_by_index(non_exact_df)
    live_exact_df, live_exact_df_max_indx = sort_df_by_index(live_exact_df)
    live_non_exact_df, live_non_exact_df_max_indx = sort_df_by_index(live_non_exact_df)

    ## Logging last indx for all the dfs
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(f"Last index for exact_df: {exact_df_max_indx}")
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(f"Last index for non_exact_df: {non_exact_df_max_indx}")
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(f"Last index for live_exact_df: {live_exact_df_max_indx}")
    config_object.masterlog.get_logger("signal_match_live_bk", "info")(f"Last index for live_non_exact_df: {live_non_exact_df_max_indx}")

    exact_mismatch_df, exact_match_df, processed_dict = match_backtest_vs_live_execution_v2(backtest_df=exact_df, live_exec_df=live_exact_df,
                                                                    param_dict=param_dict, entry_parent_id_to_primary_parent_id_map=entry_parent_id_to_primary_parent_id_map, 
                                                                    base_symbol_mapping=base_symbol_mapping, processed_dict=processed_dict, required_market_open_minutes=required_market_open_minutes, 
                                                                    mismatch_type="EXACT_MISMATCH", config_object=config_object, bk_df_max_indx=exact_df_max_indx)
    
    non_exact_mismatch_df, non_exact_match_df, processed_dict = match_backtest_vs_live_execution_v2(backtest_df=non_exact_df, live_exec_df=live_non_exact_df,
                                                                param_dict=param_dict, entry_parent_id_to_primary_parent_id_map=entry_parent_id_to_primary_parent_id_map, 
                                                                base_symbol_mapping=base_symbol_mapping, processed_dict=processed_dict, required_market_open_minutes=required_market_open_minutes,
                                                                mismatch_type="NON_EXACT_MISMATCH", config_object=config_object, bk_df_max_indx=non_exact_df_max_indx)

    ###
    return exact_mismatch_df, exact_match_df, non_exact_mismatch_df, non_exact_match_df, processed_dict, executed_df


def sort_df_by_index(df):
    max_indx = pd.Timestamp("1970-01-01 00:00:00")
    if len(df) > 0:
        df = df.sort_index()
        max_indx = df.index.max()
    
    ##
    return df, max_indx