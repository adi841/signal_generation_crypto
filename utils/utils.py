import numpy as np
import pandas as pd
import numba
import datetime as dt
import psycopg2
import psycopg2.extras as psycopg2_e
from numba import float64
import copy
from pytz import timezone
from functools import lru_cache
import traceback
from enum import Enum
import pytz

from config.config_read import config_object

class FEATURE_TYPE(Enum):
    FULLY_ADAPTIVE = 1
    PARTIALLY_ADAPTIVE = 2

class FEATURE_NAME(Enum):
    BB = 1
    KC = 2
    RSI = 3
    MARKET_SIGNAL = 4
    EMA_SIGNAL = 5
    SL_ATR = 6
    KAMA = 7
    CANDLE_ALLOC = 8
    MEDIAN = 9
    ATR = 10
    LONG_VOL = 11
    ATR3_PCT = 12
    Z_MEDIAN = 13
    VOL_SF_TILE = 14

class FEATURE_INFO:
    def __init__(self, feature_type, feature_name, **kwargs):
        self.feature_type = feature_type
        self.feature_name = feature_name

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, FEATURE_INFO):
            return NotImplemented  # or return False, depending on preference

        return self.__dict__ == other.__dict__

    def __hash__(self):
        # Convert (key, value) pairs into a tuple after sorting by key
        # so that the order doesn't affect the hash
        items = tuple(sorted(self.__dict__.items()))
        return hash(items)        


def get_resample_array(data_df, agg_time, curr_time, return_candle_length=False, return_df=False):
    
    df = data_df.copy(deep=True)
    
    newyork_tz = timezone(config_object.time_zone)
    df['Timestamp'] = df.index
    if df.index.tzinfo is None:  
        df['NYCindex'] = df['Timestamp'].dt.tz_localize('UTC').dt.tz_convert(newyork_tz)
    else:
        df['NYCindex'] = df['Timestamp'].dt.tz_convert(newyork_tz)

    df.set_index('NYCindex', inplace=True)
    df.index = df.index.tz_localize(None)

    df = resample_stock_data(df, agg_time, offset=config_object.offset)
    df = df.loc[~df.index.duplicated(), :]

    df = df[~pd.isna(df['open'])]

    df['high'].fillna(method = 'ffill', inplace=True)
    df['low'].fillna(method = 'ffill', inplace=True)
    df['close'].fillna(method = 'ffill', inplace=True)
    df['avg_price'] = (df['high'] + df['low'] + df['close'])/3

    if return_df:
        return df
    
    ###
    col_loc_dict = {col:indx for indx, col in enumerate(df.columns.tolist())}
    
    numpy_array = df.to_numpy()
    ts_array = df.index.values.astype('datetime64[ns]').astype(np.int64)

    if return_candle_length:
        candle_length = (np.abs(df['close'] - df['open']))/df['open'] * 100
        candle_length = candle_length.to_numpy()

        return numpy_array, ts_array, col_loc_dict, candle_length

    return numpy_array, ts_array, col_loc_dict


##
@numba.jit(nopython = True)
def TRAdjEMA(close, high, low, Periods, Pds, Mltp):
    # Initialize variables
    Mltp1 = 2 / (Periods + 1)
    TRAdjEMA = np.zeros_like(close, dtype=np.float64)

    # Compute TL, TH, and TR
    TL = np.minimum(low, np.roll(close, 1))
    TH = np.maximum(high, np.roll(close, 1))
    TR = np.abs(TH - TL)

    # Precompute rolling min and max for TR
    rolling_min_TR = np.zeros_like(TR)
    rolling_max_TR = np.zeros_like(TR)
    for i in range(len(TR)):
        if i < Pds:
            rolling_min_TR[i] = np.min(TR[:i+1])
            rolling_max_TR[i] = np.max(TR[:i+1])
        else:
            rolling_min_TR[i] = np.min(TR[i-Pds+1:i+1])
            rolling_max_TR[i] = np.max(TR[i-Pds+1:i+1])

    # Calculate TRAdj, Mltp2, and Rate for each period
    for i in range(len(close)):
        if (rolling_max_TR[i] - rolling_min_TR[i]) == 0:
            TRAdj = 0
        else:
            TRAdj = (TR[i] - rolling_min_TR[i]) / (rolling_max_TR[i] - rolling_min_TR[i])

        Mltp2 = TRAdj * Mltp
        Rate = Mltp1 * (1 + Mltp2)
        TRAdjEMA[i] = close[i] if i == 1 else TRAdjEMA[i-1] + Rate * (close[i] - TRAdjEMA[i-1])

    return TRAdjEMA


#
@numba.jit(nopython = True)
def FRAMA(high, low, source, fast_ma, slow_ma, ma_length):
    #source for now is close

    # Initialize output before the algorithm  
    Filt = np.empty(source.shape)

    # sequencially calculate all variables and the output  
    for i in range(0, source.shape[0]): 
        Filt[i] = source[i] 
        # If there's not enough data, Filt is the price - whitch it already is, so just skip  
        if i < 2 * ma_length:  
            continue  
        # take 2 ma_lengthes of the input  
        v1_high = high[i-2*ma_length:i - ma_length]  
        v1_low = low[i-2*ma_length:i - ma_length]  
        v2_high = high[i - ma_length:i]
        v2_low = low[i - ma_length:i]

        #v1_high = high[i-1*ma_length:i - int(ma_length/2)]  
        #v1_low = low[i-1*ma_length:i - int(ma_length/2)]  
        #v2_high = high[i - int(ma_length/2):i]
        #v2_low = low[i - int(ma_length/2):i]

        # for the 1st ma_length calculate N1  
        H1 = np.max(v1_high)  
        L1 = np.min(v1_low)  
        N1 = (H1 - L1) / ma_length

        # for the 2nd ma_length calculate N2  
        H2 = np.max(v2_high)  
        L2 = np.min(v2_low)  
        N2 = (H2 - L2) / ma_length

        # for both ma_lengthes calculate N3  
        H = max([H1, H2])  
        L = min([L1, L2])  
        N3 = (H - L) / (2 * ma_length)

        # calculate fractal dimension  
        Dimen = 0  
        if N1 > 0 and N2 > 0 and N3 > 0:  
            Dimen = (np.log(N1 + N2) - np.log(N3)) / np.log(2)

        # calculate lowpass filter factor  
        w = np.log(2/(slow_ma+1))
        alpha = np.exp(w*(Dimen-1))
        alpha = max([alpha, 0.01])  
        alpha = min([alpha, 1])

        oldN = (2-alpha) / alpha
        newN = (((slow_ma-fast_ma) * (oldN-1)) / (slow_ma-1)) + fast_ma
        newalpha = 2/(newN+1)
        newalpha = max([newalpha, 2/(slow_ma+1)])  
        newalpha = min([newalpha, 1])  

        # filter the input data  
        Filt[i] = newalpha * source[i] + (1 - newalpha) * Filt[i-1]  
        # if currentBar < 2*ma_length + 1: <--- i dont get what these 2 lines do  
        # Filt = source[i]
    return Filt

##
@numba.njit(nopython=True, nogil=True)
def _ewma_infinite_hist(arr_in, window):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    #alpha = 2 / float(window + 1)
    alpha = 1/ window
    ewma[0] = arr_in[0]
    ewma[1] = arr_in[1]
    for i in range(2, n):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma


##
@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def ewma_test(arr_in, window):
 
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    alpha = 1 / float(window)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

##

@numba.njit(nopython=True, nogil=True)
def ewm_mean_adjust_true(x, alpha):
    """
    Pandas-like EWM with adjust=True, ignore_na=False (NaN at that index => NaN output).
    Leading NaNs remain NaN; once values start, uses normalized historical weights.

    Similar to pandas.Series.ewm(alpha=1/period, adjust=True, ignore_na=False).mean() or pd.Series(up).ewm(alpha=1/period).mean()
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    beta = 1.0 - alpha

    # running weighted sum (numerator) and running weight (denominator)
    num = 0.0
    den = 0.0
    started = False

    for i in range(n):
        v = x[i]
        if np.isnan(v):
            out[i] = np.nan
            # propagate the decay of previous history so later points have proper normalization
            if started:
                num *= beta
                den = den * beta + 1.0
            continue

        if not started:
            started = True
            num = v
            den = 1.0
            out[i] = num / den
        else:
            num = num * beta + v
            den = den * beta + 1.0
            out[i] = num / den

    return out


##
def rsi_tradingview_numpy(close_diff, period: int = 14, round_rsi: bool = True):
    up_array = copy.deepcopy(close_diff)
    up_array[up_array < 0] = 0

    ##
    down_array = copy.deepcopy(close_diff)
    down_array[down_array > 0] = 0
    down_array *= -1

    ##
    up = ewm_mean_adjust_true(up_array, 1.0 / period)
    down = ewm_mean_adjust_true(down_array, 1.0 / period)

    rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))
    return np.round(rsi, 2) if round_rsi else rsi


def rsi_tradingview(close_diff, period: int = 14, round_rsi: bool = True):

    up = copy.deepcopy(close_diff)
    up[up < 0] = 0
    # up = _ewma_infinite_hist(up, period)#pd.Series.ewm(up, alpha=1/period).mean()
    up = pd.Series(up).ewm(alpha=1/period).mean()
    # ewma_test(up[1:], period)

    down = copy.deepcopy(close_diff)
    down[down > 0] = 0
    down *= -1
    # down = _ewma_infinite_hist(down, period)
    down = pd.Series(down).ewm(alpha=1/period).mean()

    rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))

    return np.round(rsi, 2) if round_rsi else rsi

@numba.njit
def fill_nans_inplace(arr, fill_value):
    """
    Fill NaNs in-place in a 1D NumPy array (float dtype) with fill_value.
    """
    for i in range(arr.size):
        if np.isnan(arr[i]):
            arr[i] = fill_value
    return arr



@numba.njit
def numba_loops_fill(arr):
    '''Numba decorator solution provided by shx2.'''
    if arr.ndim == 1:
        for i in range(1, arr.shape[0]):
            if np.isnan(arr[i]):
                arr[i] = arr[i-1]
        
        return arr
        
    out = np.full(arr.shape, np.nan)
    value = np.nan
    for row_idx in range(0, out.shape[0]):
        for col_idx in range(0, out.shape[1]):
            if np.isnan(arr[row_idx, col_idx]):
                out[row_idx, col_idx] = value
            
            else:
                value = arr[row_idx, col_idx]
                out[row_idx, col_idx] = value

    return out

##
def connect_postgre(config_object):
    conn = psycopg2.connect(user=config_object.user, password=config_object.password, host=config_object.host, port=config_object.port, database=config_object.database)
    cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
    
    return conn, cursor

def convert_ns_to_datetime(ns):
    # Convert the nanoseconds to seconds
    seconds = ns / 1e9
    # Convert the seconds to a datetime object
    dt_ = dt.datetime.fromtimestamp(seconds)
    return dt_

@numba.njit
def shift_np(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, arr[0]), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, arr[-1])))
                
                
@numba.njit(nopython=True, nogil=True)
def ewma_infinite_hist(arr_in, window, type_ = 'atr'):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    if type_ in ['atr']:
        alpha = 1/ window
    
    elif type_ == 'kc':
        # alpha = 1 / float(window + 1)
        alpha = 1 / window

    else:
        raise ValueError('type_ should be either atr or kc')
    
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    
    return ewma

@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    """
    DONT MAKE ANY CHANGES TO THIS FUNCTION.

    Implemented as df.ewm(span=timeperiod).mean()
    This function is used to calculate the EMA of a given array.
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@numba.njit(nopython=True, nogil=True)
def ewma_mean_span_no_adjust(arr_in, span):
    """
    Equivalent to pandas.Series.ewm(span=span, adjust=False).mean().
    Leading NaNs are skipped; first non-NaN seeds the recursion; NaNs after
    that propagate the prior value (i.e. EWM is not updated for NaN inputs).
    """
    n = arr_in.shape[0]
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / float(span + 1)
    started = False
    prev = 0.0
    for i in range(n):
        v = arr_in[i]
        if np.isnan(v):
            if not started:
                out[i] = np.nan
            else:
                out[i] = prev
            continue
        if not started:
            prev = v
            started = True
        else:
            prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


@numba.njit(nopython=True, nogil=True)
def ewma_std_span(arr_in, span):
    """
    Equivalent to pandas.Series.ewm(span=span, adjust=True).std()
    (bias-corrected weighted std, the pandas default).

    Online recursion over weighted accumulators:
        S_w   += 1            (after decay)
        S_wx  += x            (after decay)
        S_wxx += x*x          (after decay)
        S_ww  += 1            (after decay by (1-alpha)^2)
    var_biased   = S_wxx/S_w - (S_wx/S_w)^2
    correction   = 1 / (1 - S_ww / S_w^2)
    var_unbiased = var_biased * correction
    Leading NaNs propagate; the first observation returns NaN (pandas behavior).
    """
    n = arr_in.shape[0]
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / float(span + 1)
    beta = 1.0 - alpha
    beta_sq = beta * beta

    S_w = 0.0
    S_wx = 0.0
    S_wxx = 0.0
    S_ww = 0.0
    n_obs = 0

    for i in range(n):
        v = arr_in[i]
        if np.isnan(v):
            out[i] = np.nan
            continue

        S_w = S_w * beta + 1.0
        S_wx = S_wx * beta + v
        S_wxx = S_wxx * beta + v * v
        S_ww = S_ww * beta_sq + 1.0
        n_obs += 1

        if n_obs < 2:
            out[i] = np.nan
            continue

        mean = S_wx / S_w
        var_biased = S_wxx / S_w - mean * mean
        denom = 1.0 - S_ww / (S_w * S_w)
        if denom <= 0.0:
            out[i] = np.nan
            continue
        var_unbiased = var_biased / denom
        if var_unbiased < 0.0:
            var_unbiased = 0.0
        out[i] = np.sqrt(var_unbiased)

    return out

##
@numba.jit(nopython=True)
def calculate_ema_kc_adaptive(values, span):
    """Calculate Exponential Moving Average (EMA) using Numba."""

    ema = np.zeros(len(values))
    ema[0] = values[0]  # Initialize EMA with the first value
    
    for i in range(1, len(values)):
        alpha = 2 / (span[i] + 1)  # Calculate smoothing factor based on adaptive span
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    return ema



@numba.jit(nopython=True)
def calculate_rsi_numba(close, period):
    rsi_values = np.empty(len(close))
    rsi_values[:] = np.nan

    for i in range(period, len(close)):
        delta = close[i - period + 1:i + 1] - close[i - period:i]  # Calculate the difference array
        gain = np.sum(delta[delta > 0])  # Sum of positive changes (gains)
        loss = -np.sum(delta[delta < 0])  # Sum of negative changes (losses)

        avg_gain = gain / period
        avg_loss = loss / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_values[i] = rsi

    return rsi_values

@numba.jit(nopython=True)
def adaptive_rsi_numba(close, adaptive_length):
    adaptive_rsi = np.empty(len(close))
    adaptive_rsi[:] = np.nan

    for i in range(len(close)):
        period = adaptive_length[i]
        if i < period:
            adaptive_rsi[i] = np.nan
        else:
            rsi_values = calculate_rsi_numba(close[i - period:i + 1], period)
            adaptive_rsi[i] = rsi_values[-1]  # Get the last RSI value

    return adaptive_rsi


@numba.jit(nopython=True)
def calculate_bollinger_bands_sma(values, lengths, multiplier):
    """Calculate Bollinger Bands using SMA as the middle band."""

    upper_band = np.zeros(len(values))
    lower_band = np.zeros(len(values))
    middle_band = np.empty(len(values))
    middle_band[:] = np.nan  # Initialize middle band with NaN

    for i in range(len(values)):
        length = int(lengths[i])  # Get the adaptive length for each index
        if i >= length:  # Ensure there's enough data for the calculation
            # Calculate SMA for the current window
            middle_band[i] = np.mean(values[i - length + 1:i + 1])
            
            # Extract the window using the adaptive length
            window = values[i - length + 1:i + 1]
            std_dev = np.std(window)

            upper_band[i] = middle_band[i] + multiplier * std_dev
            lower_band[i] = middle_band[i] - multiplier * std_dev
        else:
            # Set NaN if the window cannot be formed
            upper_band[i] = np.nan
            lower_band[i] = np.nan


    return upper_band, lower_band



@numba.jit(nopython=True)
def calculate_adaptive_ema(values, adaptive_lengths):
    """Calculate EMA with adaptive lengths using Numba for performance."""
    ema = np.zeros(len(values))
    ema[0] = values[0]  # Initialize the EMA with the first value
    
    for i in range(1, len(values)):
        length = adaptive_lengths[i]
        alpha = 2 / (length + 1)  # Adaptive smoothing factor based on length
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@numba.jit(nopython=True)
def calculate_custom_indicators_numba(high, low, adaptive_length):
    """Calculate custom indicators HHSa and LLSa using Numba for performance."""
    
    HHSa = np.zeros(len(high))
    LLSa = np.zeros(len(low))
    
    for i in range(len(high)):
        length = adaptive_length[i]
        if i >= length:
            highest_high = np.max(high[i - length + 1:i + 1])
            lowest_high = np.min(high[i - length + 1:i + 1])
            HHH = (high[i] - lowest_high) / (highest_high - lowest_high) if high[i] > high[i - 1] else 0
            
            highest_low = np.max(low[i - length + 1:i + 1])
            lowest_low = np.min(low[i - length + 1:i + 1])
            LLL = (highest_low - low[i]) / (highest_low - lowest_low) if low[i] < low[i - 1] else 0
            
            HHSa[i] = HHH
            LLSa[i] = LLL
    
    # Calculate EMA of HHSa and LLSa with adaptive lengths
    HHSa_ema = calculate_adaptive_ema(HHSa, adaptive_length) * 100
    LLSa_ema = calculate_adaptive_ema(LLSa, adaptive_length) * 100
    
    return HHSa_ema, LLSa_ema


def numpy_left_join(coin_indx_array, signal_array, shift_ts, ffill):
    
    bband_indx = np.intersect1d(coin_indx_array, signal_array[:, 0])
    indx_loc = np.where(np.isin(coin_indx_array, bband_indx))[0]

    #####
    array = np.full((coin_indx_array.shape[0], signal_array.shape[1]), np.nan)
    array[indx_loc] = signal_array

    if ffill:
        array = numba_loops_fill(array)
    
    array = array[:, 1]
    if shift_ts > 0:
        array = shift_np(array, shift_ts)
        
    return array


def resample_stock_data(df, timedelta, offset=None):
    df = df.loc[~df.index.duplicated(), :]
    df = df.sort_index()
    
    aggregation_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume':'sum',
    }
    return df.resample(timedelta, offset=offset).agg(aggregation_dict)


def get_inst_status_df():
    """
    Create a df with 5 data points. Each data point is 10 minutes apart starting from 2010-01-01 00:00:00.
    2 and 4 data point should have is_trading as 0.
    """
    base_tz = pytz.timezone(config_object.time_zone)
    inst_status_df = pd.DataFrame([(dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=base_tz), 1), 
            (dt.datetime(2010, 1, 1, 0, 10, 0, tzinfo=base_tz), 1), 
            (dt.datetime(2010, 1, 1, 0, 20, 0, tzinfo=base_tz), 0), 
            (dt.datetime(2010, 1, 1, 0, 30, 0, tzinfo=base_tz), 0), 
            (dt.datetime(2010, 1, 1, 0, 40, 0, tzinfo=base_tz), 1)], 
            columns=['timestamp', 'is_trading'])

    ##
    return inst_status_df
        
def get_order_tag(signal_time: dt.datetime, parent_id, signal):
    assert isinstance(signal_time, dt.datetime)
    assert isinstance(parent_id, int)
    assert isinstance(signal, int)

    order_tag = f"{signal_time.strftime('%Y%m%d%H%M%S')}_{parent_id}_{signal}"
    return order_tag


@numba.jit(nopython=True)
def kama(prices, er_period=10, fast_period=2, slow_period=30):
    """
    Compute KAMA using Numba-accelerated loop.
    
    Parameters:
    - prices: np.ndarray of closing prices.
    - er_period: Period for Efficiency Ratio (default 10).
    - fast_period: Fast EMA period (default 2).
    - slow_period: Slow EMA period (default 30).
    
    Returns:
    - np.ndarray of KAMA values (same length as prices, NaNs at start).
    """
    n = len(prices)
    if n < er_period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)
    
    # Pre-allocate arrays
    er = np.full(n, 0.0, dtype=np.float64)
    sc = np.full(n, 0.0, dtype=np.float64)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(er_period, n):
        # ER: net change / total volatility
        net_change = abs(prices[i] - prices[i - er_period])
        total_change = 0.0
        for j in range(1, er_period + 1):
            total_change += abs(prices[i - j + 1] - prices[i - j])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0.0
        
        # SC
        sc[i] = (er[i] * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # Initialize KAMA
    kama[er_period - 1] = prices[er_period - 1]
    
    # Update KAMA
    for i in range(er_period, n):
        kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
    
    return kama


@numba.jit(nopython=True)
def ewma_var(logret, halflife):
    """
    Numba-safe EWMA variance.
    logret: 1D array of log returns
    halflife: in bars (e.g., 60 for 1h)
    """
    n = logret.shape[0]
    out = np.empty(n)

    # stable decay
    # lambda is decay per bar: exp(-ln(2)/HL)
    lam = np.exp(-0.6931471805599453 / halflife)

    var = 0.0
    initialized = False

    for i in range(n):
        r = logret[i]
        if np.isnan(r):
            out[i] = np.nan
            continue
        r2 = r * r
        if not initialized:
            var = r2              # initialize with first non-NaN return
            initialized = True
        else:
            var = lam * var + (1 - lam) * r2
        out[i] = var
    return out
