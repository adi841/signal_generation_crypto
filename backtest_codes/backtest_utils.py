

import json
import pandas as pd
from psycopg2.extras import execute_values

import pickle
import datetime as dt
import numpy as np
import pandas as pd
import traceback
import sys
import asyncio
import time
import copy
import slack
from queue import Queue
from pathlib import Path
import os
import sys
from os.path import dirname, abspath

file_path = dirname(abspath(__file__))
while True:
    if file_path.endswith("signal_generation_crypto"):
        break
    
    file_path = dirname(file_path)

##
sys.path.append(file_path)

## Prod import triggers a `git fetch origin` + a client lookup at module load
## (config_utils runs check_sync_with_origin() / check_db_host() at import time) —
## skip in local/test mode, otherwise importing this module is impossible off a live
## host. Mirrors the guard every sleeve already carries (e.g.
## utils/pair_relative_value_signal_generator.py). The DB-touching helpers here
## (get_parameters / upsert_data / DataClassBacktest.init_data) are unreachable in
## local hist_replay runs, which read parquet instead.
if os.environ.get("CLIENT_ID"):
    from shared_codes.utils.config_utils import connect_postgre
else:
    def connect_postgre(db_user=None, direct_conn=False, database=None):
        raise RuntimeError(
            "connect_postgre is stubbed: CLIENT_ID is unset (local/test mode). "
            "A DB-backed code path was reached that should not run locally.")

from config.config_read import config_object
from shared_codes.utils.zmq_utils import get_dealer_socket
from utils.base import Base

def get_parameters(start_int: int, end_int: int, database_name: str|None):
    conn, cursor = connect_postgre(db_user="signal_generation", database=database_name)

    ##
    is_live_json = json.dumps({"is_live": 1})

    ##
    query_ = """SELECT * FROM submodel_parameters WHERE parent_trading_model >= {} and parent_trading_model < {} and model_parameters @> '{}'::jsonb;"""
    query_ = query_.format(start_int, end_int, is_live_json)

    ##
    cursor.execute(query_)
    data = cursor.fetchall()

    ##
    parameter_dict = {}
    for data_dict in data:
        parameter_dict[data_dict['parent_trading_model']] = data_dict['model_parameters']

    ##
    conn.close(); cursor.close()
    return parameter_dict

###
def process_trades_df(df: pd.DataFrame, trades_start_datetime: pd.Timestamp, trades_end_datetime: pd.Timestamp, asset: str):
    """
    Main Objective is handle the corner case when the trades are already in position at the `trades_start_datetime`.
    """
    df_trades = df.copy(deep=True)

    df_trades.loc[:, "asset"] = asset
    df_trades["signal_diff"] = df_trades["signal"].diff().fillna(0)
    df_trades = df_trades[df_trades["signal_diff"]!=0]
    df_trades = df_trades.loc[:trades_end_datetime] # Fixing the end date

    #####
    if len(df_trades):
        df_trades_tmp = df_trades.loc[trades_start_datetime:].copy(deep=True)
        if len(df_trades_tmp):
            signal_1 = df_trades_tmp.iloc[0]["signal"]
            if signal_1 == 0: # Means that trades was already in position since the `trades_start_datetime`. So we ignore it and take the next signal.
                indx_loc = df_trades.index.get_loc(df_trades_tmp.index[0]) + 1
                df_trades = df_trades.iloc[indx_loc:]

            else:
                df_trades = df_trades.loc[trades_start_datetime:]
            
            return df_trades




def get_trading_info(strat_ls: list):
    conn, cursor = connect_postgre(db_user="signal_generation")

    strat_tup = tuple(strat_ls)
    min_, max_ = min(strat_ls), max(strat_ls)
    max_ += 5 # to include the last value (pairs, reversals)
    # query_ = """select parent_trading_model, child_trading_model, client_id from child_id_info where parent_trading_model in {};""".format(strat_tup)
    query_ = f"""select parent_trading_model, child_trading_model, client_id from child_id_info where parent_trading_model >= {min_} and parent_trading_model <= {max_};"""
    cursor.execute(query_)
    data_parent_child = cursor.fetchall()

    child_client_dict = {}
    parent_child_dict = {}
    all_child_list = []
    for data_dict in data_parent_child:
        # client_child_dict.setdefault(data_dict['client_id'], []).append(data_dict['child_trading_model'])
        child_client_dict[data_dict['child_trading_model']] = data_dict['client_id']
        parent_child_dict.setdefault(data_dict['parent_trading_model'], []).append(data_dict['child_trading_model'])
        all_child_list.append(data_dict['child_trading_model'])

    ##
    all_child_tup = tuple(all_child_list)
    query_ = f"""
        WITH RankedModels AS (
            SELECT
                timestamp,
                child_trading_model,
                weight_mantissa,
                weight_exponent,
                quantity,
                ROW_NUMBER() OVER (PARTITION BY child_trading_model ORDER BY timestamp DESC) AS rn
            FROM 
                model_weight
            WHERE
                child_trading_model in {all_child_tup}
        )
        SELECT 
            timestamp,
            child_trading_model,
            weight_mantissa,
            weight_exponent,
            quantity
        FROM 
            RankedModels
        WHERE 
            rn = 1;
    """

    cursor.execute(query_)
    data_lots = cursor.fetchall()

    ##
    lots_dict = {}
    for data_dict in data_lots:
        assert data_dict['weight_exponent'] is not None, "Weight Exponent is not None"
        assert data_dict['weight_mantissa'] is not None, "Weight Mantissa is not None"
        lots_dict[data_dict['child_trading_model']] =  data_dict['weight_mantissa'] * (10** data_dict['weight_exponent'])
    
    ## client info
    query_ = "select * from clients;"
    cursor.execute(query_)
    client_info = cursor.fetchall()
    client_info_dict = {}
    for data in client_info:
        client_info_dict[data['client_id']] = data['name'] + "_" + data['asset_class']

    ##
    conn.close(); cursor.close()
    return child_client_dict, parent_child_dict, lots_dict, client_info_dict

def upsert_data(data_df: pd.DataFrame):
    """
    Insert data into the table or update it if a conflict occurs.
    """
    conn, cursor = connect_postgre(db_user="signal_generation")
    
    query = """
    INSERT INTO backtest_signals (signal_floor_time, parent_trading_model, signal_id, signal, case_num, price, execution_type, order_tag)
    VALUES %s
    ON CONFLICT (parent_trading_model, signal_floor_time)
    DO UPDATE SET 
        signal = EXCLUDED.signal,
        price = EXCLUDED.price,
        execution_type = EXCLUDED.execution_type,
        order_tag = EXCLUDED.order_tag;
    """

    data_tuples = [
            (
                row['Timestamp'],
                row['parent_trading_model'],
                row['signal_id'],
                row['signal'],
                row['case'],
                row['tradeprice1'],
                row['execution_type'],
                row['order_tag1']
            )
            for index, row in data_df.iterrows()
        ]
    
    ##
    execute_values(cursor, query, data_tuples)
    conn.commit()

    ##
    conn.close(); cursor.close()


def args_checks(curr_time, hist_replay, trades_only, trades_start_datetime, trades_end_datetime, data_file_path, trades_output_path, csv_output_path, csv_output_start_datetime, csv_output_end_datetime, \
    csv_output_row_count, cores, hist_replay_dir, init_hist_replay_dir, ffill_data, timestamp_signal_output):

    ## curr_time checks
    if curr_time is not None:
        try:
            curr_time = dt.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"{curr_time}: Incorrect data format, should be YYYY-MM-DD HH:MM:SS")

    if hist_replay:
        assert curr_time is not None, "curr_time must be set when hist_replay is 1"
        assert hist_replay_dir is not None, "hist_replay_dir must be set when hist_replay is 1"
        assert init_hist_replay_dir is not None, "init_hist_replay_dir must be set when hist_replay is 1"
        assert data_file_path is None, "data_file_path must be None when hist_replay is 1"
    
    if trades_only == 0:
        assert curr_time is not None, "curr_time must be set when trades_only is 0"
    
    ##
    if trades_only == 1:
        assert trades_output_path is not None, "trades_output_path must be set when trades_only is 1"
        assert data_file_path is not None, "data_file_path must be set when trades_only is 1"
        assert hist_replay == 0, "hist_replay must be 0 when trades_only is 1"
        assert timestamp_signal_output == 0, "timestamp_signal_output must be 0 when trades_only is 1"
    
    if trades_start_datetime or trades_end_datetime:
        assert trades_only == 1, "trades_start_datetime and trades_end_datetime are only used when trades_only is 1"
        assert trades_start_datetime is not None, "trades_start_datetime must be set when trades_end_datetime is set"
        assert trades_end_datetime is not None, "trades_end_datetime must be set when trades_start_datetime is set"
        try:
            dt.datetime.strptime(trades_start_datetime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"{trades_start_datetime}: Incorrect data format for trades_start_datetime, should be YYYY-MM-DD HH:MM:SS")

        try:
            dt.datetime.strptime(trades_end_datetime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"{trades_end_datetime}: Incorrect data format for trades_end_datetime, should be YYYY-MM-DD HH:MM:SS")

        assert trades_start_datetime < trades_end_datetime, "trades_start_datetime must be less than trades_end_datetime"

    ## Create path for `trades_output_path` or `csv_output_path`
    if trades_output_path:
        trades_output_path = Path(trades_output_path)
        if not trades_output_path.exists():
            trades_output_path.mkdir(parents=True, exist_ok=True)
    
    if csv_output_path:
        csv_output_path = Path(csv_output_path)
        if not csv_output_path.exists():
            csv_output_path.mkdir(parents=True, exist_ok=True)

    if csv_output_start_datetime or csv_output_end_datetime:
        assert csv_output_start_datetime is not None, "csv_output_start_datetime must be set when csv_output_end_datetime is set"
        assert csv_output_end_datetime is not None, "csv_output_end_datetime must be set when csv_output_start_datetime is set"
        assert csv_output_row_count is None, "csv_output_row_count must be None when csv_output_start_datetime and csv_output_end_datetime are set"
        try:
            dt.datetime.strptime(csv_output_start_datetime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"{csv_output_start_datetime}: Incorrect data format for csv_output_start_datetime, should be YYYY-MM-DD HH:MM:SS")

        try:
            dt.datetime.strptime(csv_output_end_datetime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"{csv_output_end_datetime}: Incorrect data format for csv_output_end_datetime, should be YYYY-MM-DD HH:MM:SS")

        assert csv_output_start_datetime < csv_output_end_datetime, "csv_output_start_datetime must be less than csv_output_end_datetime"
    
    if csv_output_row_count:
        assert int(csv_output_row_count) > 0, "csv_output_row_count must be greater than 0"
        assert csv_output_start_datetime is None, "csv_output_start_datetime must be None when csv_output_row_count is set"
        assert csv_output_end_datetime is None, "csv_output_end_datetime must be None when csv_output_row_count is set"
    
    assert cores > 0, "cores must be greater than 0"
    assert cores <= (os.cpu_count() - 1), "cores must be less than or equal to the number of available cores"


##
class DataClassBacktest(Base):
    def __init__(self, symbol_mapping: list, process_name: str, identity_name: str, data_queue, curr_time: dt.datetime, hist_replay: int=0, 
                hist_replay_dir: str=None, init_hist_replay_dir: str=None, ffill_data: int=-9):

        super().__init__(ffill=bool(ffill_data), connect_or_socket=False)

        assert hist_replay in [0, 1], "Invalid hist_replay value"

        self.symbol_mapping = symbol_mapping
        self.process_name = process_name
        self.identity_name = identity_name
        self.data_queue = data_queue
        self.hist_replay = hist_replay
        self.hist_replay_dir = hist_replay_dir
        self.init_hist_replay_dir = init_hist_replay_dir

        self.curr_time = dt.datetime.now().replace(second=0, microsecond=0)
        if curr_time is not None:
            if isinstance(curr_time, str):
                self.curr_time = dt.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')
            else:
                self.curr_time = curr_time
        
        ##
        self.curr_time = self.base_tz.localize(self.curr_time)

        ##
        self.LAST_START_TIME_DICT = {x:curr_time for x in self.symbol_mapping.values()}
        self.SYMBOL_DATA_DICT = {}
        self.PREV_SYMBOL_STATUS_DICT = {x:1 for x in self.symbol_mapping.values()}

        self.prepare_socket()
        if self.hist_replay:
            self.hist_replay_init_data()
            self.market_data_replay.load_hist_replay_data(self.hist_replay_dir, symbol_list=symbol_mapping.keys(), ffill=self.ffill)
        else:
            self.init_data(self.curr_time)

    ##
    def init_data(self, end_date):
        conn, cursor = connect_postgre(db_user="signal_generation")
        for base_sym, symbol in self.symbol_mapping.items():
            print(f"Getting data for {base_sym}:{symbol}")
            df = self.get_init_data(symbol, end_date, ffill=self.ffill)

            self.SYMBOL_DATA_DICT[base_sym] = df.copy(deep=True)

        ##
        DATA_DICT = copy.deepcopy(self.SYMBOL_DATA_DICT)
        self.data_queue.put(DATA_DICT)
        
        ##
        conn.close(); cursor.close()
    
    def hist_replay_init_data(self):
        """
        if file.endswith(".parquet"):
            symbol = file.split(".")[0]
            if symbol in self.symbol_mapping.keys():
        """
        for base_sym, symbol in self.symbol_mapping.items():
            df = pd.read_parquet(os.path.join(self.init_hist_replay_dir, f"{symbol}_input.parquet"))
            df.index = df.index.tz_localize(self.base_tz)
            self.SYMBOL_DATA_DICT[symbol] = df

    def get_data(self, curr_time, coin, base_coin):
        data = asyncio.run(self.get_ohlcv_data(curr_time, coin=coin))
        return data

    def run(self):
        # for ts in [dt.datetime(2024, 9, 16, 5, 10, 0)]:
        # self.curr_time = self.base_tz.localize(ts)
        self.curr_time = min([df.index[-1] for df in self.SYMBOL_DATA_DICT.values()])
        self.curr_time = self.curr_time.replace(second=0, microsecond=0)
        self.symbol_data_rows = {x: [] for x in self.symbol_mapping.keys()}
        while 1:
            try:
                # now_dt = dt.datetime.now().replace(second=0, microsecond=0).replace(tzinfo=self.base_tz)
                now_dt = self.base_tz.localize(dt.datetime.now().replace(second=0, microsecond=0))
                if self.curr_time != now_dt:
                    if not self.hist_replay:
                        time.sleep(3)
                    try:
                        for base_sym, symbol in self.symbol_mapping.items():

                            ##
                            msg_ = f"{symbol} {self.curr_time}"
                            config_object.masterlog.get_logger(f"backtest_util_{self.process_name}", "debug")(msg_)

                            ## Update the data
                            data = self.get_data(self.curr_time, symbol, base_sym)

                            if data.index[-1] <= self.SYMBOL_DATA_DICT[base_sym].index[-1]:
                                msg_ = f"Data is already present for {symbol}... \n {data} \n\n  {self.SYMBOL_DATA_DICT[base_sym].tail(5)}"
                                config_object.masterlog.get_logger(f"backtest_util_{self.process_name}", "info")(msg_)
                                continue

                            ##
                            if self.hist_replay:
                                self.symbol_data_rows[base_sym].append(data)

                            else:
                                self.SYMBOL_DATA_DICT[base_sym] = pd.concat((self.SYMBOL_DATA_DICT[base_sym], data))
                                # self.SYMBOL_DATA_DICT[base_sym] = self.SYMBOL_DATA_DICT[base_sym].iloc[1:]
                                ## ffill open, high, low, close
                                if self.ffill:
                                    self.SYMBOL_DATA_DICT[base_sym][['open', 'high', 'low', 'close']] = self.SYMBOL_DATA_DICT[base_sym][['open', 'high', 'low', 'close']].ffill()
                                    self.SYMBOL_DATA_DICT[base_sym]['volume'] = self.SYMBOL_DATA_DICT[base_sym]['volume'].fillna(0)

                        ##
                        self.curr_time += dt.timedelta(minutes=1)

                        if self.hist_replay:
                            if self.curr_time.hour == 15 and self.curr_time.minute == 30 and self.curr_time.day % 4 == 0:
                                for base_sym, symbol in self.symbol_mapping.items():
                                    if self.symbol_data_rows[base_sym]:
                                        data = pd.concat(self.symbol_data_rows[base_sym])
                                        self.SYMBOL_DATA_DICT[base_sym] = pd.concat((self.SYMBOL_DATA_DICT[base_sym], data))
                                        self.SYMBOL_DATA_DICT[base_sym] = self.SYMBOL_DATA_DICT[base_sym].sort_index()
                                        if self.ffill:
                                            self.SYMBOL_DATA_DICT[base_sym][['open', 'high', 'low', 'close']] = self.SYMBOL_DATA_DICT[base_sym][['open', 'high', 'low', 'close']].ffill()
                                            self.SYMBOL_DATA_DICT[base_sym]['volume'] = self.SYMBOL_DATA_DICT[base_sym]['volume'].fillna(0)

                                        ## Empty the symbol data rows
                                        self.symbol_data_rows[base_sym] = []

                                print(f"Putting data in queue... {self.curr_time}")
                                DATA_DICT = copy.deepcopy(self.SYMBOL_DATA_DICT)
                                if self.data_queue.qsize() < 10:
                                    self.data_queue.put(DATA_DICT)
                                else:
                                    self.data_queue.get()
                                    self.data_queue.put(DATA_DICT)
                        
                        else:
                            DATA_DICT = copy.deepcopy(self.SYMBOL_DATA_DICT)
                            self.data_queue.put(DATA_DICT)

                    except Exception as e:
                        tb_ = traceback.format_exc()
                        msg_ = f"Error in signal generation...  {e}\n{tb_}"
                        config_object.masterlog.get_logger(f"backtest_util_{self.process_name}", "critical")(msg_)
                        self.shutdown_and_exit(None, None)

                ##
                if not self.hist_replay:
                    time.sleep(1)

            except Exception as e:
                tb_ = traceback.format_exc()
                msg_ = f"Error in signal generation...  {e}\n{tb_}"
                config_object.masterlog.get_logger(f"backtest_util_{self.process_name}", "critical")(msg_)


if __name__ == "__main__":
    from queue import Queue
    curr_time = dt.datetime(2024, 9, 6, 9, 35, 0)
    aa = DataClassBacktest(symbol_mapping={'PAZ24': 'PAZ24'}, process_name="test", identity_name="test", data_queue=Queue(), curr_time=curr_time)
    aa.run()
    