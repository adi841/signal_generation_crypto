
import pickle
import datetime as dt
import numpy as np
import pandas as pd
import math
import traceback
import sys
import signal
import zmq
import zmq.asyncio
import asyncio
from pytz import timezone
import time
from collections import deque

from config.config_read import config_object
from shared_codes.utils.zmq_utils import get_dealer_socket
from shared_codes.utils.config_utils import connect_postgre
from shared_codes.utils.dataclass_utils import OHLCVRequest, OHLCVResponse
from .market_data_replay import MarketDataReplay


class Base:
    def __init__(self, ffill=None, connect_or_socket=True):
        self.identity_name = None
        self.logger_name = "signal_gen" # This will be overridden in the child class

        signal.signal(signal.SIGINT, self.shutdown_and_exit)
        # signal.signal(signal.SIGTERM, self.shutdown_and_exit)
        signal.signal(signal.SIGQUIT, self.shutdown_and_exit)

        ##
        self.conn, self.cursor = connect_postgre(db_user="signal_generation")
        self.base_tz = timezone('UTC')
        self.target_tz = timezone(config_object.time_zone)
        self.data_dict = dict() # Store the discarded data (Like init_timedelta)

        self.market_data_replay = MarketDataReplay()

        self.ffill = ffill
        self.connect_or_socket = connect_or_socket
        self.var_checks()

    def var_checks(self):
        assert self.ffill in [True, False], f"Invalid ffill value: {self.ffill}"

    def shutdown_and_exit(self, signum, frame):
        ##
        config_object.masterlog.get_logger(self.logger_name, "info")("Shutting down signal generator...")
        config_object.masterlog.get_logger(self.logger_name, "info")("Closing socket...")

        msg_ = f"Received signal: {signum}. \n"
        msg_ += f"Stack Trace: {traceback.format_stack()}"
        config_object.masterlog.get_logger(self.logger_name, "error")(msg_)

        if hasattr(self, "data_socket"):
            self.data_socket.setsockopt(zmq.LINGER, 0)
            self.data_socket.close()

        if hasattr(self, "broker_socket"):
            self.broker_socket.setsockopt(zmq.LINGER, 0)
            self.broker_socket.close()

        if hasattr(self, "context"):
            self.context.term()

        sys.exit(0)
    
    def get_init_data(self, coin, start_time, ffill):
        ## Add a check that start_time is a floor minute, that is, start_time.second == 0 and start_time.microsecond == 0
        assert start_time.second == 0 and start_time.microsecond == 0, f"start_time should be a floor minute. Got {start_time}"

        end_date = start_time
        start_date = end_date - dt.timedelta(days=config_object.lookback_days)

        if start_date.tzinfo is None:
            end_date, start_date = self.base_tz.localize(end_date), self.base_tz.localize(start_date)
        elif start_date.tzinfo != self.base_tz:
            start_date = start_date.astimezone(self.base_tz)
            end_date = end_date.astimezone(self.base_tz)
        
        ##
        final_df = self.load_intraday_data(coin, start_date, end_date)
        df = self.merge_data_inst_status(final_df, start_date, end_date, ffill=ffill)

        ##
        last_indx = df.index[-1]
        for i in range(-1, -len(df), -1):
            msg_ = f"Indx: {df.index[i]}"
            config_object.masterlog.get_logger(self.logger_name, "debug")(msg_)
            
            if (((last_indx - df.index[i]).total_seconds()/60) >= config_object.init_timedelta) and (df.index[i].minute % 5) in [2,3]:
                end_indx = df.index[i]
                break
        
        ##
        return final_df

    def merge_data_inst_status(self, df, start_date, end_date, ffill):
        assert start_date.second == 0 and start_date.microsecond == 0, f"start_date should be a floor minute. Got {start_date}"
        assert end_date.second == 0 and end_date.microsecond == 0, f"end_date should be a floor minute. Got {end_date}"

        ## Create a index from start_date to end_date with 1 minute interval
        index = pd.date_range(start=start_date, end=end_date, freq='1min')
        df = df.reindex(index)
        df = df.sort_index()
        if ffill:
            df = df.ffill()
        
        return df

    # Helper function to query 1-min bars from "ohlcv_data"
    def load_intraday_data(self, symbol, start_dt, end_dt):
        """
        Load 1-minute bars (UTC) from ohlcv_data for given symbol and [start_dt, end_dt).
        Return as a pandas.DataFrame with a DateTimeIndex (UTC).
        """
        query = f"""
            SELECT 
                start_time, open, high, low, close, volume FROM ohlcv_data WHERE symbol = %s AND start_time >= %s AND start_time <= %s ORDER BY start_time
        """

        self.cursor.execute(query, (symbol, start_dt, end_dt))
        rows = self.cursor.fetchall()
        if not rows:
            return pd.DataFrame([], columns=['open','high','low','close','volume'])

        df = pd.DataFrame(rows, columns=['start_time', 'open', 'high', 'low', 'close', 'volume'])
        # Convert start_time to datetime index
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df.set_index('start_time', inplace=True)
        return df

    def get_db_strat_params(self, strategy_name):
        """Fetch this unit's live rows from `submodel_parameters` — the SINGLE SOURCE OF
        TRUTH for parent ids AND kernel parameters in live mode (the rows were dumped
        with the full parameter set, so the live host does not need the frozen bundle).

        LIVE mode only — hist_replay uses the frozen-bundle builders with per-unit
        LOCAL ids instead. Returns {parent_trading_model: model_parameters(dict)} for
        the unit's is_live = 1 rows.

        The config entry's start_parent_id is asserted against the DB base id as loud
        drift detection — the config declares, the DB decides.
        """
        assert self.cursor is not None, \
            "live mode requires a DB connection — connect_postgre is stubbed in this checkout"

        if getattr(self, "base_coin2", None):
            self.cursor.execute(
                """SELECT parent_trading_model, model_parameters FROM submodel_parameters
                   WHERE model_parameters->>'strategy_name' = %s
                     AND model_parameters->>'coin1' = %s
                     AND model_parameters->>'coin2' = %s
                     AND (model_parameters->>'is_live')::int = 1""",
                (strategy_name, self.coin1, self.coin2))
        else:
            self.cursor.execute(
                """SELECT parent_trading_model, model_parameters FROM submodel_parameters
                   WHERE model_parameters->>'strategy_name' = %s
                     AND model_parameters->>'coin' = %s
                     AND (model_parameters->>'is_live')::int = 1""",
                (strategy_name, self.coin1))

        rows = self.cursor.fetchall()
        assert len(rows) > 0, f"no live submodel_parameters rows for {strategy_name} {self.coin1}"
        db_rows = {row['parent_trading_model']: row['model_parameters'] for row in rows}
        assert min(db_rows) == self.start_parent_id, \
            (f"config start_parent_id {self.start_parent_id} != DB base id {min(db_rows)} "
             f"for {strategy_name} {self.coin1} — update the client config")
        return db_rows

    def prepare_socket(self, prepare=0):
        if prepare:
            self.identity_name = f"{self.process_name}_"
            self.identity_name += str(time.time_ns()) 
        
        else:
            self.identity_name += str(time.time_ns())[-4:]

        self.context = zmq.asyncio.Context()

        zmq_data = 'ipc:///tmp/databento'
        if not self.connect_or_socket:
            pass
        
        elif hasattr(self, "base_coin1"):
            ## The signal generator resolved this unit's coin_param entry at construction
            ## (self.param_entry = config_object.coin_param[<sleeve>][<unit_key>]); its
            ## socket_key picks the trades_receiver lane in global_settings.socket_config.
            ## Hard-indexed: a live process without a socket lane is a config bug.
            socket_key = str(self.param_entry["socket_key"])
            ipc_broker_socket = config_object.socket_config[socket_key]["trades_receiver"]

        else:
            raise ValueError(f"Base coin(s) not set for {self.identity_name}")
        
        ##
        msg_ = f"Connecting to data socket: {config_object.zmq_data}"
        config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        # self.data_socket, self.context = get_dealer_socket(self.identity_name, self.context, zmq_data)
        self.data_socket, self.context = get_dealer_socket(self.identity_name, self.context, config_object.zmq_data)

        if self.connect_or_socket:
            msg_ = f"Connecting to signal receiver socket: {ipc_broker_socket}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

            self.broker_socket, self.context = get_dealer_socket(self.identity_name, self.context, ipc_broker_socket)
            self.broker_socket.setsockopt(zmq.IMMEDIATE, 1)        

    def get_data_db(self, curr_time, coin) -> "OHLCVResponse":
        ## Get data from database
        msg_ = f"Getting data from database for {coin} at {curr_time}"
        config_object.masterlog.get_logger(self.logger_name, "warning")(msg_)

        conn, cursor = connect_postgre(db_user="signal_generation")

        data_query_ = f"select * from ohlcv_data where symbol = '{coin}' and start_time = '{curr_time}' order by start_time DESC limit 1;"
        cursor.execute(data_query_)
        reply = cursor.fetchall()
        reply = reply[0]
        # reply = (curr_time, curr_time, reply['open'], reply['high'], reply['low'], reply['close'], reply['volume'])
        reply = OHLCVResponse(coin=coin, start_time=curr_time, end_time=curr_time, open=reply['open'], high=reply['high'], low=reply['low'], close=reply['close'], volume=reply['volume'], is_complete=True, reconnect_socket=False)

        ##
        conn.close(); cursor.close()

        return reply

    def get_tz_str(self, tz):
        if tz == timezone("UTC"):
            return "UTC"
        
        elif tz == dt.timezone.utc:
            return "UTC"
        
        else:
            raise ValueError(f"Timezone {tz} not supported")

    def check_tz(self, tz1, tz2):
        return self.get_tz_str(tz1) == self.get_tz_str(tz2)
    
    def get_instrument_status(self, base_symbol, curr_ts):
        return 1
    
    def reconnect(self):
        self.data_socket.setsockopt(zmq.LINGER, 0)
        self.data_socket.close()

        if hasattr(self, "broker_socket"):
            self.broker_socket.setsockopt(zmq.LINGER, 0)
            self.broker_socket.close()

        self.context.term()
        self.prepare_socket(prepare=1)
    
    ##
    def get_hist_replay_data(self, curr_time, coin):
        if curr_time.tzinfo is not None:
            curr_time = curr_time.replace(tzinfo=None)

        open_, high, low, close, volume = self.market_data_replay.get_hist_replay_data(curr_time, coin)
        df_ = pd.DataFrame([(curr_time, open_, high, low, close, volume)], columns=['start_time', 'open', 'high', 'low', 'close', 'volume'])
        df_ = df_.set_index('start_time')
        df_ = df_.replace(pd.NaT, np.nan)
        df_ = df_.astype('float64')
        df_.index = df_.index.tz_localize(self.base_tz)
        assert self.check_tz(df_.index.tz, self.base_tz), f"Timezone of the current timestamp should be UTC. Got {df_.index.tzinfo}"

        return df_

    #@profile
    async def get_ohlcv_data(self, curr_time, coin):

        assert isinstance(curr_time, dt.datetime), f"curr_time should be datetime object. Got  {curr_time}: {type(curr_time)}"
        assert self.check_tz(curr_time.tzinfo, self.base_tz), f"Timezone of the current timestamp should be UTC. Got {curr_time.tzinfo}"
        
        if self.hist_replay:
            return self.get_hist_replay_data(curr_time, coin)

        dict_df = self.data_dict.get(coin, {}).pop(curr_time, None)
        if dict_df is not None:
            return pd.DataFrame([dict_df])

        ##
        retry_count = 0
        success = False
        while True:
            force_complete = retry_count > config_object.max_data_retry
            sent_time = dt.datetime.utcnow()
            pickle_query = OHLCVRequest(coin=coin, time=curr_time, force_complete=force_complete, sent_time=sent_time)
            msg_ = f"Getting data from websocket...\t{curr_time}\t{pickle_query}"
            config_object.masterlog.get_logger(self.logger_name, "debug")(msg_)

            payload = pickle.dumps(pickle_query)
            await self.data_socket.send_multipart([payload])

            try:
                reply = await self.data_socket.recv_multipart()
                reply: OHLCVResponse|None = pickle.loads(reply[0])
                if isinstance(reply, type(None)):
                    # `continue` will go to `finally` block
                    continue

                if reply.coin != coin:
                    msg_ = f"Coin: {reply.coin} != {coin}. Reconnecting..."
                    config_object.masterlog.get_logger(self.logger_name, "error")(msg_)
                    self.reconnect()
                    reply = None
                    continue

                if reply.reconnect_socket:
                    msg_ = f"Reconnect socket flag is True. Reconnecting..."
                    config_object.masterlog.get_logger(self.logger_name, "error")(msg_)
                    self.reconnect()
                    reply = None
                    continue
                
                msg_ = f"Data received from websocket...\t{curr_time}\t{pickle_query}\t{reply}"
                config_object.masterlog.get_logger(self.logger_name, "debug")(msg_)
                success = True
                break

            ## catch all zmq exceptions
            except (zmq.error.Again, zmq.error.ZMQError, zmq.error.ContextTerminated) as e:
                msg_ = f"Error in receiving data from websocket. Reconnecting...\t{curr_time}\t{pickle_query}\t{e}\n{traceback.format_exc()}"
                config_object.masterlog.get_logger(self.logger_name, "error")(msg_)
                self.reconnect()
                reply = None
                continue

            except Exception as e:
                msg_ = f"Error in receiving data from websocket...\t{curr_time}\t{pickle_query}\t{e}\n{traceback.format_exc()}"
                config_object.masterlog.get_logger(self.logger_name, "error")(msg_)
                reply = None
                continue

            finally:
                if not success:
                    retry_count += 1
                    await asyncio.sleep(config_object.data_retry_interval)
                
                if force_complete:
                    msg_ = f"Force complete flag is True. Breaking the loop..."
                    config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
                    break

        try:
            is_nan = True
            if reply is not None:
                is_nan = np.isnan(reply.close)

            ##
            if is_nan or reply is None: #and self.ffill == True:
                reply : OHLCVResponse = self.get_data_db(curr_time, coin)
        
        except Exception as e:
            try:
                msg_ = f"Data not received from websocket...  {curr_time}  {e} \n {traceback.format_exc()}"
                config_object.masterlog.get_logger(self.logger_name, "error")(msg_)
                
                ## Get data from database
                ## Get data from database
                # if self.ffill == True:
                #     reply = self.get_data_db(curr_time, coin)
                # else:
                #     reply = (curr_time, curr_time, np.nan, np.nan, np.nan, np.nan, np.nan)
                
                reply = OHLCVResponse(coin=coin, start_time=curr_time, end_time=curr_time, open=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=np.nan, is_complete=True, reconnect_socket=False)

            except:
                tb_ = traceback.format_exc()
                msg_ = f"Data not received from database...  {curr_time}\n" + tb_
                config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
                reply = OHLCVResponse(coin=coin, start_time=curr_time, end_time=curr_time, open=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=np.nan, is_complete=True, reconnect_socket=False)
                  
        ##
        if self.hist_replay:
            msg_ = f"Data received from websocket...  {curr_time}  {reply}"
            config_object.masterlog.get_logger(self.logger_name, "debug")(msg_)

        ##
        if math.isnan(reply.close):
            msg_ = f"Data is not available for the coin {coin} for timestamp {curr_time}"
            config_object.masterlog.get_logger(self.logger_name, "warning")(msg_)
        ###
        reply_ls = [reply.start_time, reply.end_time, reply.open, reply.high, reply.low, reply.close, reply.volume]
        df_ = pd.DataFrame([reply_ls], columns=['start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume'])
        df_ = df_.set_index('start_time')
        assert self.check_tz(df_.index.tz, self.base_tz), f"Timezone of the current timestamp should be UTC. Got {df_.index.tzinfo}"
        df_ = df_[['open', 'high', 'low', 'close', 'volume']]
        df_ = df_.replace(pd.NaT, np.nan)
        df_ = df_.astype('float64')

        return df_

    # async def process_data_send_data(self, db_signal_dict):
    #     """
    #     This function is not used anymore. Directly sending data to broker
    #     """
    #     try:
    #         msg_ = f"Sending data to broker"
    #         config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
            
    #         send_tasks = []
    #         for parent_stratId in db_signal_dict:
    #             tmp_df = pd.DataFrame(db_signal_dict[parent_stratId])
    #             send_tasks.append(self.send_signal_broker(tmp_df))

    #         ##
    #         await asyncio.gather(*send_tasks)

    #     except Exception as e:
    #         tb_ = traceback.format_exc()
    #         msg_ = f"Error in sending data to broker: {e} \n {tb_}"
    #         config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)

    def dump_hist_signal_df(self, hist_signal_df):
        conn, cursor = connect_postgre(db_user="signal_generation")

        # Convert DataFrame to list of tuples for executemany
        data = [
            (
                row.signal_time, row.signal_floor_time, row.signal_id,
                row.parent_trading_model, row.signal, row.case_num,
                row.price, row.execution_type, row.is_tp_sl, row.order_tag
            )
            for row in hist_signal_df.itertuples(index=False)
        ]

        query = """
            INSERT INTO live_signals ( signal_time, signal_floor_time, signal_id, parent_trading_model, signal, case_num, price, execution_type, is_tp_sl, order_tag) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (parent_trading_model, signal_floor_time, signal_id)
            DO UPDATE SET 
                signal = EXCLUDED.signal,
                case_num = EXCLUDED.case_num,
                price = EXCLUDED.price,
                execution_type = EXCLUDED.execution_type,
                is_tp_sl = EXCLUDED.is_tp_sl,
                order_tag = EXCLUDED.order_tag;
        """
        ## data might not be in the correct order
        cursor.executemany(query, data)
        conn.commit()
        conn.close()

    ##
    def dump_signals(self, signal_dict):
        conn, cursor = connect_postgre(db_user="signal_generation")

        data = []
        for parent_stratId in signal_dict:
            signals = signal_dict[parent_stratId]
            tup_ = (
                signals["signal_time"][-1], signals["signal_floor_time"][-1], signals["signal_id"][-1], parent_stratId, signals["signal"][-1], signals["case_num"][-1],
                signals["price"][-1], signals["exec_type"][-1], signals["is_pb_sl"][-1], signals["order_tag"][-1]
                )
            data.append(tup_)

        ## ON CONFLICT is NOT optional here, even though every minute normally writes a
        ## fresh (parent_trading_model, signal_id, signal_floor_time). A restarted process
        ## re-walks the minutes it already wrote — its anchor is the batch's
        ## last_processed_ts, not "wherever the DB got to" — so on ANY restart this insert
        ## hits the live_signals PK. Without the upsert that raises UniqueViolation, and
        ## because executemany sends the batch as one statement, the abort loses EVERY
        ## cell's row for that minute, not just the duplicate. The caller
        ## (log_dump_data) then logs and swallows it: the process keeps running and
        ## silently records nothing until it walks past the already-written minutes.
        ## Same clause dump_hist_signal_df above already uses.
        query = """
            INSERT INTO live_signals ( signal_time, signal_floor_time, signal_id, parent_trading_model, signal, case_num, price, execution_type, is_tp_sl, order_tag)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (parent_trading_model, signal_floor_time, signal_id)
            DO UPDATE SET
                signal = EXCLUDED.signal,
                case_num = EXCLUDED.case_num,
                price = EXCLUDED.price,
                execution_type = EXCLUDED.execution_type,
                is_tp_sl = EXCLUDED.is_tp_sl,
                order_tag = EXCLUDED.order_tag;
        """

        cursor.executemany(query, data)
        conn.commit()

        conn.close()    

    ##
    async def send_signal_broker(self, df):
        try:
            payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL) # Convert to bytes
            await self.broker_socket.send_multipart([memoryview(payload)], copy=False)

            # Wait for the acknowledgement
            msg_dealer = await self.broker_socket.recv()

            msg_ = f"Received message from broker: {msg_dealer}"
            config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
        
        # except zmq.Again:
        #     msg_ = f"Error in sending data to broker: {e}"
        #     config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)

        except zmq.error.Again as e:
            msg_ = f"Ack not received from Broker: {e}"
            config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)

        except zmq.error.ZMQError as e:
            msg_ = f"Error in sending data to broker: {e}"
            config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
        
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"Error in sending data to broker: {e} \n {tb_}"
            config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
        
        finally:
            ## Delete the df from memory
            del df
            del payload


    ##
    def get_exec_signals_dataframe_fast(self, exec_signal_dict: dict):
        """
        Fastest method to convert exec_signal_dict to a single DataFrame.
        Uses pre-allocated numpy arrays for maximum speed with 200+ parent IDs.
        """
        if not exec_signal_dict:
            return pd.DataFrame()
        
        # Get column names from the first parent_id (all should have same structure)
        first_parent_id = next(iter(exec_signal_dict))
        column_names = list(exec_signal_dict[first_parent_id].keys())
        
        # Calculate total rows needed (sum of all deque lengths)
        total_rows = 0
        for parent_id, parent_data in exec_signal_dict.items():
            if column_names and column_names[0] in parent_data:
                total_rows += len(parent_data[column_names[0]])
        
        if total_rows == 0:
            return pd.DataFrame(columns=column_names)
        
        # Pre-allocate numpy arrays for each column
        arrays = {}
        for col in column_names:
            # Use object dtype to handle mixed types efficiently
            arrays[col] = np.empty(total_rows, dtype=object)
        
        # Fill arrays using vectorized operations
        current_idx = 0
        for parent_id, parent_data in exec_signal_dict.items():
            if not parent_data or column_names[0] not in parent_data:
                continue
                
            # Get length for this parent_id
            parent_rows = len(parent_data[column_names[0]])
            end_idx = current_idx + parent_rows
            
            # Copy data for each column in chunks
            for col in column_names:
                if col in parent_data:
                    # Convert deque to list once, then to numpy array
                    arrays[col][current_idx:end_idx] = list(parent_data[col])
                else:
                    # Fill with None if column doesn't exist for this parent
                    arrays[col][current_idx:end_idx] = None
            
            current_idx = end_idx
        
        # Create DataFrame directly from dictionary of arrays
        return pd.DataFrame(arrays)
