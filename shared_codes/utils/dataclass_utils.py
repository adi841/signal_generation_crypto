
from dataclasses import dataclass
import pandas as pd
import datetime as dt
from enum import Enum

class SocketStatus(Enum):
    UNKNOWN = 0
    ACTIVE = 1
    INACTIVE = 2
    DEAD = 3
    DISCONNECTED = 4

class Source(Enum):
    SOCKET = 1
    QUEUE = 2
    REDIS = 3

class SocketPurpose(Enum):
    DATA_RECEIVER = 1
    TRADES_RECEIVER = 2
    PUBLISHER = 3
    PULL_SINK = 4
    TRADES_FORWARDER = 5
    BROKER_TRADES_DISTRIBUTOR = 6
    BROKER_TRADES_RECEIVER = 7
    BROKER_INSTRUCTION_RECEIVER = 8
    TRADES_EXECUTOR = 9
    SUBSCRIBER = 10
    PUSH = 11
    EXECUTION_INSTRUCTION_RECEIVER = 12
    ZMQ_SERVER_ROUTER = 13
    INST_STATUS_PUBLISHER = 14 # Publisher for the status of the instructions
    SA_SERVER_TRADE_RECEIVER = 15 # Trade receiver for single asset
    SA_SERVER_ORDER_BRIDGE = 16 # Sending/Placing orders from single asset server to order router

class SocketType(Enum):
    MONITOR = 1
    DEALER = 2
    ROUTER = 3
    PUB = 4
    SUB = 5
    REQ = 6
    REP = 7
    PUSH = 8
    PULL = 9
    PAIR = 10


class StrategyStatus(Enum):
    START = 1
    STOP = 2

class RouterResponse(Enum):
    ACK = 1
    PONG = 2


@dataclass
class SocketInfo:
    ROUTER_STATUS: SocketStatus
    LAST_PING : dt.datetime


class MsgType(Enum):
    OHLCV = 1
    SIGNAL = 2
    PUSH_SIGNAL_RECV = 3 # signal that we received the signal
    PUSH_SIGNAL_EXIT = 4 # signal that signal execution is done
    DUMP_ERROR_LOG = 5
    DUMP_STRATEGY_OBJECT = 6 # dump the strategy object to the database
    CANCEL_CHILD_TRADE_KEYS = 7 # delete keys from `trades_exec_coroutine_dict` and cancel coroutines
    INSTRUMENT_STATUS_DICT = 8 # update the instrument status dictionary
    DB_UPDATE = 9 # update the database. Used to pull data from the database in all the worker sockets
    ROLL_OVER = 10 # roll over the trading models. Used to communicate the rollover to the `worker_socket`

@dataclass
class PushSignal:
    type: MsgType
    child_trading_model: int
    status: str

    def __str__(self):
        return f"Child Trading Model: {self.child_trading_model}, Status: {self.status}"
    
    
@dataclass
class ChildTradingModel:
    parent_trading_model: int
    child_trading_model: int
    client_id: int
    state: StrategyStatus

@dataclass
class SymbolInfo:
    id_: int
    symbol: str
    lot_size: int
    tick_size: float
    min_order_size: float
    min_qty_inc: float
    underlying: str
    expiry_date: dt.datetime

@dataclass
class ChildIDWeights:
    child_trading_model: int
    weight: float
    qty: float


@dataclass
class TradingParams:
    symbol: str
    expected_qty: float
    child_trading_model: int
    signal_df: pd.Series
    init_exec_params: dict
    pft_exec_params: dict | None
    sl_exec_params: dict | None
    exit_exec_params: dict | None
    second_entry_exec_params: dict | None
    off_hours_exec_params: dict | None # Non regular trading hours execution parameters. 
    contract_info: SymbolInfo
    sent_time: dt.datetime = dt.datetime.utcnow()
    load_order_object: bool = False
    
    def __str__(self):
        ## Print all the parameters
        return f"Child Trading Model: {self.child_trading_model}, Symbol: {self.symbol}, Expected Qty: {self.expected_qty}, Signal DF: {self.signal_df}, " \
               f"Init Exec Params: {self.init_exec_params}, PFT Exec Params: {self.pft_exec_params}, SL Exec Params: {self.sl_exec_params}, Contract Info: {self.contract_info}"



@dataclass
class TradingParamsMleg:
    symbol1: str
    symbol2: str
    expected_qty1: float
    expected_qty2: float
    child_trading_model1: int
    child_trading_model2: int
    signal_df: pd.Series
    init_exec_params: dict
    pft_exec_params: dict | None
    sl_exec_params: dict | None
    exit_exec_params: dict | None
    second_entry_exec_params: dict | None
    off_hours_exec_params: dict | None # Non regular trading hours execution parameters.
    contract_info1: SymbolInfo
    contract_info2: SymbolInfo
    sent_time: dt.datetime = dt.datetime.utcnow()
    load_order_object: bool = False
    
    def __str__(self):
        ## Print all the parameters
        return f"C1: {self.child_trading_model1}, S1: {self.symbol1}, E1: {self.expected_qty1}, C2: {self.child_trading_model2}, S2: {self.symbol2}, E2: {self.expected_qty2}, Signal DF: {self.signal_df}, " \
               f"Init Exec Params: {self.init_exec_params}, PFT Exec Params: {self.pft_exec_params}, SL Exec Params: {self.sl_exec_params}, Contract Info1: {self.contract_info1}, Contract Info2: {self.contract_info2}"
    

@dataclass
class OHLCV:
    start_time: dt.datetime
    end_time: dt.datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class SendOrderInfo:
    child_trading_model: int
    symbol: str
    expected_qty: float
    sent_time: dt.datetime
    trading_params: TradingParams
    acked: bool = False # check if we got confirmation from the execution engine
    resend_count: int = 0 # number of times we have resent the order

    def __str__(self) -> str:
        return f"Child Trading Model: {self.child_trading_model}, Symbol: {self.symbol}, Expected Qty: {self.expected_qty}, Sent Time: {self.sent_time}, Trading Params: {self.trading_params}, Acked: {self.acked}, Resend Count: {self.resend_count}"

    
class TaskType(Enum):
    TIMER = 1
    SOCKET_RECV = 2
    TRADE_EXEC = 3
    
@dataclass
class DumpErrorLog:
    log_level: str
    source: str
    error_origin: str
    error: str
    type: MsgType = MsgType.DUMP_ERROR_LOG

    def __str__(self) -> str:
        return f"Log Level: {self.log_level}, Source: {self.source}, Error Origin: {self.error_origin}, Error: {self.error}"
    
@dataclass
class DumpStrategyObject:
    child_trading_model: int
    load_obj_db: bool
    execution_state : str
    strategy_object: bytes
    qty: float
    qty2: float = 0.0
    type: MsgType = MsgType.DUMP_STRATEGY_OBJECT
    
class InstructionType(Enum):
    UpdateDB = 1
    DeleteChildKeys = 2 # Delete keys from `sent_trades_dict` and `trades_exec_coroutine_dict`
    RollOver = 3 # Roll over the trading models


@dataclass
class InstructUpdateDB:
    type: InstructionType = InstructionType.UpdateDB

    def __str__(self) -> str:
        return "Update DB"

@dataclass
class InstructDeleteChildKeys:
    child_trading_models: list[int]
    type: InstructionType = InstructionType.DeleteChildKeys

    def __str__(self) -> str:
        str_ = ""
        for model in self.child_trading_models:
            str_ += f"{str(model)}, "
        
        return f"Child Trading Model: {str_}"


@dataclass
class InstructRollOver:
    rollover_map: dict[str, str] # {old_symbol: new_symbol}. {"ESU24": "ESZ24"}
    type: InstructionType = InstructionType.RollOver

    def __str__(self) -> str:
        return f"Rollover Map: {self.rollover_map}"

@dataclass
class ChildModelTradingInfo:
    last_process_time: dt.datetime
    last_signal_time: dt.datetime # When diff != 0 or when curr_signal = 0 after a signal (1/-1)
    last_signal: int


class StrategyType(Enum):
    BREAKOUT = 1
    REVERSAL = 2
    QUOTE_REVERSAL = 3


## This will used for query OHLCV data
@dataclass(frozen=True)
class OHLCVRequest:
    coin: str
    time: dt.datetime
    force_complete: bool
    sent_time: dt.datetime # to log the sent time of the request

@dataclass(frozen=True)
class OHLCVResponse:
    coin: str
    start_time: dt.datetime
    end_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool
    reconnect_socket: bool # If true, the dealer socket should be reconnected

