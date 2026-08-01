
import datetime as dt
from zoneinfo import ZoneInfo
import importlib, sys
import os

_MODULE_NAME = 'shared_codes.utils.cme_mkt_hours'
_last_reload: dt.datetime = dt.datetime.min
_last_mtime: float = 0.0
_module = None

def get_cme_mkt_module(max_age_min: int = 60, check_file_changes: bool = True):
    """
    Get the CME market hours module with smart reloading.
    
    Args:
        max_age_min: Maximum age in minutes before forced reload (fallback)
        check_file_changes: If True, check file modification time for changes
    
    Returns:
        The reloaded or cached module
    """
    global _last_reload, _last_mtime, _module
    
    should_reload = False
    now = dt.datetime.now(dt.timezone.utc)
    
    # Initial load
    if _module is None:
        should_reload = True
    
    # File change detection (primary method)
    elif check_file_changes:
        try:
            # Get the actual file path
            module_file = None
            if _MODULE_NAME in sys.modules:
                module_file = getattr(sys.modules[_MODULE_NAME], '__file__', None)
            
            if not module_file:
                # Construct expected path
                module_file = os.path.join(os.path.dirname(__file__), '..', '..', 'shared_codes', 'utils', 'cme_mkt_hours.py')
                module_file = os.path.abspath(module_file)
            
            if module_file and os.path.exists(module_file):
                current_mtime = os.path.getmtime(module_file)
                if current_mtime > _last_mtime:
                    print(f"Reloading CME market hours module due to file change at {module_file}")
                    should_reload = True
                    _last_mtime = current_mtime
        except (OSError, AttributeError):
            # Fallback to time-based if file checking fails
            pass
    
    # Time-based fallback (secondary method)
    if not should_reload and (now - _last_reload) > dt.timedelta(minutes=max_age_min):
        should_reload = True
    
    if should_reload:
        # Force fresh load
        sys.modules.pop(_MODULE_NAME, None)
        _module = importlib.import_module(_MODULE_NAME)
        _last_reload = now
        
        # Update mtime on successful reload
        if check_file_changes:
            try:
                module_file = getattr(_module, '__file__', None)
                if module_file and os.path.exists(module_file):
                    _last_mtime = os.path.getmtime(module_file)
            except (OSError, AttributeError):
                pass
    
    return _module

def datetime_to_float_hour(dt_: dt.datetime) -> float:
    """
    Convert a datetime object to a float representing the hour with fractions.
    For example:
      13:15 -> 13.25
      14:30 -> 14.5
      14:45 -> 14.75
    """
    hour = dt_.hour
    minute = dt_.minute
    fraction = minute / 60
    # Round to two decimal places (if desired) or leave as is
    return hour + fraction

def is_holiday_closed(product: str, dt_: dt.datetime) -> bool:
    mod = get_cme_mkt_module()
    REGULAR_HOURS = mod.REGULAR_HOURS
    HOLIDAYS      = mod.HOLIDAYS
    SETTLE_INFO   = mod.SETTLE_INFO    

    """
    Check if the given datetime falls into a holiday closure window for the specified product.
    Returns True if market is closed due to holiday, False otherwise.
    """
    date_str = dt_.strftime("%Y-%m-%d")
    if product not in HOLIDAYS:
        raise ValueError(f"No holiday schedule defined for product '{product}'")

    if product in HOLIDAYS and date_str in HOLIDAYS[product]:
        start, end = HOLIDAYS[product][date_str]
        current_hour = datetime_to_float_hour(dt_)
        return start <= current_hour < end
    return False

def is_within_regular_hours(product: str, dt_: dt.datetime) -> bool:
    mod = get_cme_mkt_module()
    REGULAR_HOURS = mod.REGULAR_HOURS
    HOLIDAYS      = mod.HOLIDAYS
    SETTLE_INFO   = mod.SETTLE_INFO

    """
    Check if the given datetime falls within the normal trading hours for the specified product.
    Returns True if open, False if closed.
    """
    if product not in REGULAR_HOURS:
        # If no schedule defined, assume closed
        raise ValueError(f"No regular hours defined for product '{product}'")
    
    day_name = dt_.strftime("%A")  # e.g. "Monday", "Tuesday", etc.
    if day_name not in REGULAR_HOURS[product]:
        # No trading defined for this day (e.g. Saturday)
        raise ValueError(f"No regular hours defined for {day_name} for product '{product}'")
    
    # Get the intervals for that day
    intervals = REGULAR_HOURS[product][day_name]
    current_hour = datetime_to_float_hour(dt_)
    
    # Intervals are defined in pairs
    # For example, (0,17,18,24) -> (0,17) and (18,24) are two trading sessions.
    # We'll iterate through these in steps of 2.
    try:
        for start, end in intervals:
            # Check if current_hour is within [start, end)
            # We'll treat the end as exclusive to be consistent with half-hour increments.
            if start <= current_hour < end:
                return True
    
    except:
        raise ValueError(f"Invalid regular hours format for product '{product}' on {day_name}")
    
    return False

def is_within_non_regular_hours(product: str, dt_: dt.datetime) -> bool:
    """
    Check if the given datetime falls within the non-regular trading hours for the specified product.
    Returns True if datetime is within non-regular hours, False if not. 

    Purpose is to check if market is trading within non-liquid hours
    """
    mod = get_cme_mkt_module()
    NON_RTH_HOURS = mod.NON_RTH_HOURS

    if product not in NON_RTH_HOURS:
        # If no schedule defined, assume closed
        raise ValueError(f"No regular hours defined for product '{product}'")
    
    day_name = dt_.strftime("%A")  # e.g. "Monday", "Tuesday", etc.
    if day_name not in NON_RTH_HOURS[product]:
        # No trading defined for this day (e.g. Saturday)
        raise ValueError(f"No regular hours defined for {day_name} for product '{product}'")
    
    # Get the intervals for that day
    intervals = NON_RTH_HOURS[product][day_name]
    current_hour = datetime_to_float_hour(dt_)
    
    # Non-regular hours are those outside the defined intervals
    try:
        for start, end in intervals:
            # Check if current_hour is within [start, end)
            if start <= current_hour < end:
                return True  # Within non-regular hours
    
    except:
        raise ValueError(f"Invalid regular hours format for product '{product}' on {day_name}")
    
    return False  # If we reach here, it means it's outside regular hours



def is_market_open(product: str, dt_: dt.datetime) -> bool:
    """
    Check if the market is open for the given product at the given datetime.
    aargs:
        product: str, the product to check (e.g. 'ES', 'NQ', etc.). Underlying product
        dt_: timezone-aware datetime object, the datetime to check
    """
    mod = get_cme_mkt_module()
    REGULAR_HOURS = mod.REGULAR_HOURS
    HOLIDAYS      = mod.HOLIDAYS
    SETTLE_INFO   = mod.SETTLE_INFO

    """
    Main function:
    Returns True if the market is open for the given product at the given ET datetime,
    False otherwise. Holiday closures take precedence over normal hours.
    """
    original_tz = dt_.tzinfo
    tz_str = SETTLE_INFO[product]['timezone']
    tz = ZoneInfo(tz_str)
    dt_local = dt_.astimezone(tz)

    # First check holiday
    if is_holiday_closed(product, dt_local):
        return False
    
    # If not holiday closed, check regular hours
    return is_within_regular_hours(product, dt_local)


def float_to_hour_minute(float_hour: float) -> tuple[int, int]:
    """
    Convert a float hour (e.g. 13.25) to hour and minute tuple (13, 15).
    """
    hour = int(float_hour)
    minute = int((float_hour - hour) * 60)
    return hour, minute

def get_last_market_open_time(product, dt_: dt.datetime, cnt=0) -> dt.datetime:
    mod = get_cme_mkt_module()
    REGULAR_HOURS = mod.REGULAR_HOURS
    HOLIDAYS      = mod.HOLIDAYS
    SETTLE_INFO   = mod.SETTLE_INFO

    if cnt > 7:
        return None
    
    original_tz = dt_.tzinfo
    tz_str = SETTLE_INFO[product]['timezone']
    tz = ZoneInfo(tz_str)
    dt_local = dt_.astimezone(tz)

    date_str = dt_local.strftime("%Y-%m-%d")
    if product not in HOLIDAYS:
        raise ValueError(f"No holiday schedule defined for product '{product}'")

    if product in HOLIDAYS and date_str in HOLIDAYS[product]:
        start, end = HOLIDAYS[product][date_str]
        current_hour = datetime_to_float_hour(dt_local)

        if current_hour > end:
            hour, min_ = float_to_hour_minute(end)
            dt_ret = dt.datetime(dt_local.year, dt_local.month, dt_local.day, hour, min_, 0, tzinfo=tz)
            return dt_ret.astimezone(original_tz)
        else:
            prev_date = dt_local.date() - dt.timedelta(days=1)
            prev_datetime = dt.datetime(prev_date.year, prev_date.month, prev_date.day, 23, 59, 0, tzinfo=tz)
            dt_ret = get_last_market_open_time(product, prev_datetime, cnt + 1)
            return dt_ret.astimezone(original_tz)


    day_name = dt_local.strftime("%A")  # e.g. "Monday", "Tuesday", etc.
    if day_name not in REGULAR_HOURS[product]:
        # No trading defined for this day (e.g. Saturday)
        raise ValueError(f"No regular hours defined for {day_name} for product '{product}'")
    
    # Get the intervals for that day
    intervals = REGULAR_HOURS[product][day_name]
    current_hour = datetime_to_float_hour(dt_local)
    for i in range(1, len(intervals)):
        start, end = intervals[i]
        if start <= current_hour < end:
            # If we are in the middle of a session, return the start of the session
            hour, min_ = float_to_hour_minute(start)
            dt_ = dt.datetime(dt_.year, dt_.month, dt_.day, hour, min_, 0, tzinfo=tz)
            return dt_.astimezone(original_tz)
    
    else: # if no session found, check the previous session
        prev_date = dt_.date() - dt.timedelta(days=1)
        prev_datetime = dt.datetime(prev_date.year, prev_date.month, prev_date.day, 23, 59, 59, tzinfo=tz)
        dt_ret = get_last_market_open_time(product, prev_datetime, cnt + 1)
        return dt_ret.astimezone(original_tz)



