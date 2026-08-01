
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from collections import OrderedDict

def get_utc_now():
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)

##
def are_strings_similar(string1, string2, threshold=0.8):
    from difflib import SequenceMatcher

    if string1 is None or string2 is None:
        return False
    
    ##
    similarity_ratio = SequenceMatcher(None, string1, string2).ratio()
    return similarity_ratio > threshold


##
def check_nan(val):
    import math
    import numpy as np
    import pandas as pd
    
    if isinstance(val, float) and math.isnan(val):
        return True

    # Check for string 'nan' (case insensitive)
    if isinstance(val, str) and val.lower() == 'nan':
        return True

    # Check for numpy NaN
    if isinstance(val, (float, np.float32, np.float64)) and np.isnan(val):
        return True

    # Check for pandas NaN (both pd.NA and np.nan are NaN representations in pandas)
    if val is pd.NA:
        return True

    if pd.isna(val):
        return True
                    
    elif val == 0:
        return True
    
    elif isinstance(val, type(None)):
        return True
    
    elif isinstance(val, str):
        return True
    
    return False

##
def is_market_trading():
    now_dt = dt.datetime.utcnow()
    day = now_dt.weekday()

    if day == 5:
        return False

    elif day == 4 and now_dt.hour >= 22:
        return False

    elif day == 6 and now_dt.hour < 20:
        return False
    
    return True


def symbol_validity_check(subcribe_ls, ignore_symbol_set, logger_name):
    from shared_codes.utils.config_utils import connect_postgre
    from config import configObject

    ##
    conn, cursor = connect_postgre(db_user='execution_engine')
    symbol_tup = tuple(subcribe_ls)
    if len(symbol_tup) == 1:
        query_ = f"SELECT * FROM symbol_info WHERE symbol = '{symbol_tup[0]}';"
    else:
        query_ = f"SELECT * FROM symbol_info WHERE symbol in {symbol_tup};"

    cursor.execute(query_)
    data_ = cursor.fetchall()
    
    processed_symbols = []
    for row in data_:
        symbol = row["symbol"]
        processed_symbols.append(symbol)
        if row["expiry_date"] < dt.datetime.utcnow().date():
            ignore_symbol_set.add(symbol)

            msg_ = f"Symbol {symbol} removed from the dictionary"
            configObject.masterlog.get_logger(logger_name, "error")(msg_)
    
    ##
    msg_ = f"Processed symbols: {processed_symbols}"
    configObject.masterlog.get_logger(logger_name, "info")(msg_)

    ##
    if set(subcribe_ls).difference(set(processed_symbols)):
        symbol_no_db = set(subcribe_ls).difference(set(processed_symbols))
        ignore_symbol_set.update(symbol_no_db)

        msg_ = f"Symbols not found in the database: {set(subcribe_ls).difference(set(processed_symbols))}"
        configObject.masterlog.get_logger(logger_name, "error")(msg_)
    
    ##
    conn.close(); cursor.close()
    return ignore_symbol_set

def get_subscribe_symbols(base_symbols, days):
    """
    How many days in the future to check for expiry dates.

    2 weeks is a good estimate for most options.
    """
    from shared_codes.utils.config_utils import get_s3_config
    import pandas as pd
    from io import StringIO
    
    ##
    get_s3_config.cache_clear()
    symbol_rollover_str, _ = get_s3_config("rollover_info", use_branch=False)
    symbol_rollover_str = symbol_rollover_str.decode('utf-8')
    df_ = pd.read_csv(StringIO(symbol_rollover_str))
    df_["LAST_TRADE"] = pd.to_datetime(df_["LAST_TRADE"], format="%Y-%m-%d")
    df_["ROLLOVER_DATE"] = df_.apply(lambda x: x["LAST_TRADE"] - dt.timedelta(days=x["ROLL_DATE_TD"]), axis=1)
    
    if df_ is None:
        return []
    
    ##
    now_dt = dt.datetime.utcnow()
    
    ##
    subscribe_ls = []
    for sym in base_symbols:
        tmp_df = df_[df_["UNDERLYING"] == sym]
        tmp_df = tmp_df.sort_values(by="LAST_TRADE")
        tmp_df = tmp_df[tmp_df["ROLLOVER_DATE"] >= now_dt]
        
        for index, (_, row) in enumerate(tmp_df.iterrows()):
            if index == 0:
                subscribe_ls.append(row.GLOBEX)
                continue

            if tmp_df.iloc[index-1].ROLLOVER_DATE <= (now_dt + dt.timedelta(days=days)):
                subscribe_ls.append(row.GLOBEX)
            else:
                break

    ##
    return subscribe_ls


###
def scale_data(value, decimals=9) -> int:
    """
    Convert a numeric price to an integer scaled by 10^decimals.

    Steps:
      1) Convert to Decimal
      2) Round to 'decimals' places using half-up rounding
      3) Multiply by 10^decimals
      4) Return as int

    :param value: The input price. Can be int, float, or string.
    :param decimals: The number of decimal places to scale by (default=9).
    :return: The price scaled to 'decimals' places as an int.
    :raises ValueError: If the input cannot be parsed to a valid number.
    """
    try:
        # Convert to Decimal (str() conversion avoids float precision issues).
        d_value = Decimal(str(value))
    except Exception:
        raise ValueError(f"Invalid input for price: {value!r}")

    # Build the quantize factor, e.g., Decimal('0.000000001') for decimals=9
    quantize_factor = Decimal('1').scaleb(-decimals)  # 10^-decimals

    # Round to the specified number of decimals using ROUND_HALF_UP
    d_rounded = d_value.quantize(quantize_factor, rounding=ROUND_HALF_UP)

    # Scale by 10^decimals (move decimal point right 'decimals' positions)
    scaled = d_rounded * (Decimal('1').scaleb(decimals))

    return int(scaled)


def descale_data(value, decimals=9) -> Decimal:
    """
    Convert a scaled integer back to a Decimal by shifting the decimal
    point left by 'decimals' places, with no rounding.

    :param value: The scaled price (int, float, or str).
    :param decimals: How many decimal places the value was originally scaled by.
    :return: A Decimal of the value shifted left by 'decimals' places.
    :raises ValueError: If the input cannot be parsed to a valid number.
    """
    try:
        # Convert to Decimal (using str() avoids floating-point artifacts).
        d_value = Decimal(str(value))
    except Exception:
        raise ValueError(f"Invalid input for scaled price: {value!r}")

    # Shift the decimal place 'decimals' spots to the left (i.e., divide by 10^decimals).
    return d_value.scaleb(-decimals)

    
## MAX LEN DICT
class FixedSizeDict:
    """
    High-performance fixed-size dictionary with FIFO eviction policy.
    
    Optimized for frequent insertions and lookups with automatic size management.
    Uses __slots__ for memory efficiency and implements LRU behavior when keys are updated.
    """
    __slots__ = ['maxlen', '_d']
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self._d = OrderedDict()

    def __setitem__(self, k, v):
        if k in self._d:
            # Move existing key to end (LRU behavior)
            del self._d[k]
        elif len(self._d) >= self.maxlen:
            # Remove oldest item before adding new one
            self._d.popitem(last=False)
        self._d[k] = v

    def __getitem__(self, k):
        return self._d[k]
    
    def __contains__(self, k):
        return k in self._d
    
    def __len__(self):
        return len(self._d)
    
    def get(self, k, default=None):
        return self._d.get(k, default)
    
    def keys(self):
        return self._d.keys()
    
    def values(self):
        return self._d.values()
    
    def items(self):
        return self._d.items()
    
    def clear(self):
        self._d.clear()

    def __repr__(self):
        return f"FixedSizeDict({dict(self._d)})"


def get_max_open_files_limit() -> int:
    """
    Return the current process's soft limit for max open files (RLIMIT_NOFILE) as an integer.
    This corresponds to the soft value shown under "Max open files" in /proc/<pid>/limits.
    """
    import resource
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return int(soft_limit)


def _snap_to_increment(value, step):
    # step can be float/str/Decimal, including '1e-7'
    q = Decimal(str(value)).quantize(Decimal(str(step)), rounding=ROUND_HALF_UP)
    return float(q)  # keep return type the same as before


def get_executed_qty_crypto(trade_dollar_value, price, min_size, qty_increment, floor_qty):
    ##
    min_dollar_value = min_size * price
    if min_dollar_value > trade_dollar_value:
        return min_size, min_size*price
    
    qty_increment_dollar_value = qty_increment * price
    remaining_dollar_value = trade_dollar_value - min_dollar_value
    if floor_qty:
        executed_increment_qty = (remaining_dollar_value // qty_increment_dollar_value) * qty_increment
        total_executed_qty = executed_increment_qty + min_size
    else:
        executed_increment_qty = round(remaining_dollar_value / qty_increment_dollar_value, 0) * qty_increment
        total_executed_qty = executed_increment_qty + min_size

    return total_executed_qty, total_executed_qty*price


def get_asset1_asset2_quantities_crypto(trade_dollar_value, asset1_price, asset2_price, asset1_qty_increment, asset2_qty_increment, asset1_min_size, asset2_min_size):
    
    ##
    asset1_dollar_qty_increment = asset1_qty_increment * asset1_price
    asset2_dollar_qty_increment = asset2_qty_increment * asset2_price

    if asset1_dollar_qty_increment > asset2_dollar_qty_increment:
        total_executed_qty1, total_executed_dollar_value1 = get_executed_qty_crypto(trade_dollar_value, asset1_price, asset1_min_size, asset1_qty_increment, floor_qty=True)
        total_executed_qty2, total_executed_dollar_value2 = get_executed_qty_crypto(total_executed_dollar_value1, asset2_price, asset2_min_size, asset2_qty_increment, floor_qty=False)

    else:
        total_executed_qty2, total_executed_dollar_value2 = get_executed_qty_crypto(trade_dollar_value, asset2_price, asset2_min_size, asset2_qty_increment, floor_qty=True)
        total_executed_qty1, total_executed_dollar_value1 = get_executed_qty_crypto(total_executed_dollar_value2, asset1_price, asset1_min_size, asset1_qty_increment, floor_qty=False)

    ##
    total_executed_qty1 = _snap_to_increment(total_executed_qty1, asset1_qty_increment)
    total_executed_qty2 = _snap_to_increment(total_executed_qty2, asset2_qty_increment)

    return total_executed_qty1, total_executed_qty2


def order_tag_generator(ts: dt.datetime, parent_id: int, signal: int) -> str:
    order_tag = f"{ts.strftime('%Y%m%d%H%M%S')}_{parent_id}_{signal}"
    return order_tag

def get_datetime_ordertag(order_tag: str) -> dt.datetime:
    return dt.datetime.strptime(order_tag.split("_")[0], "%Y%m%d%H%M%S")









