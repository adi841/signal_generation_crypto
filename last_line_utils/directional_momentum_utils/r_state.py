"""Reader for the daily entry-sizing multiplier R (DMP_v3_2 directional_momentum).

R multiplies the allocation of NEW SHORT ENTRIES ONLY:
    LONG   tal = clip(VT / rv_alloc / annf, 0, 1)     * clip(K/z, 0, 1)
    SHORT  tal = clip(VT / rv_alloc / annf, 0, 1) * R * clip(K/z, 0, 1)
`allocation` appears in no kernel CONDITION, so R changes position SIZE and never entry or
exit timing. The LONG side never reads this file.

TWO DIFFERENCES FROM THE pair_momentum COPY OF THIS READER — the file body is identical,
the artifact it reads is not:
  * R is CAPPED AT 1.0 here (`R = min(1.0, 0.25 + 1.5*(1-pct))`), so the range is
    [0.25, 1.00] and a DMP target can never exceed 1.0. B1's R is uncapped to 1.75.
  * shorts-only and capped are the same decision: hot-state SHORT entries are late
    crash-chasing, so throttling them helps, while up-sizing calm LONG entries de-hedges
    the book — tested and rejected, it cost -0.24 OOS and deepened MaxDD by 1.8pp.

R is a CROSS-ASSET daily state (built from the 15-minute closes of all 11 traded assets)
that a single-asset process cannot derive, so it arrives from
`eod_scripts/directional_momentum/build_r_state.py` as a parquet of (date, R), already
capped, already lagged, and already stamped with the day each value APPLIES TO.

WHY NOT @lru_cache
    The other frozen artifacts (rv_alloc_bounds, sigma_is_ref, ...) are read once and cached
    for the life of the process, because they never change. This file is REWRITTEN NIGHTLY.
    An lru_cache would pin the first version read and silently size off a frozen R forever.
    Instead we stat() the file on every call -- effectively free -- and re-read only when
    mtime/size changes. That is also what lets a long-running live process pick up tonight's
    R without a restart: it polls, and the moment the EOD job lands a new file it is used.

WHY KEYED ON curr_time, NOT WALL-CLOCK
    hist_replay drives `curr_time` through historical dates at full speed. Anything asking
    "what is today?" would return the same answer for a whole replay and silently size every
    historical bar off the newest R. Keying on the bar's own timestamp makes replay and live
    the same code path with no branch.

The EOD builder writes via os.replace(), so a reader either sees the whole old file or the
whole new one -- never a partial parquet.
"""
import os

import numpy as np
import pandas as pd

from config.config_read import config_object

# path -> ((mtime_ns, size), Series indexed by tz-naive normalised dates)
_CACHE = {}
# path -> the (day, hour) we last emitted a staleness critical for, so an outage logs
# hourly rather than 1440 times a day per cell
_STALE_LOGGED = {}


def _load(path):
    """Return the R series, re-reading only when the file on disk has changed."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"r_state: {path} does not exist. Build it with "
            f"eod_scripts/directional_momentum/build_r_state.py --rebuild (once), then "
            f"nightly. Refusing to default R to 1.0 -- that would silently un-throttle "
            f"every SHORT entry (R lives in [0.25, 1.0], so the error is always an "
            f"OVER-size, up to 4x) while looking entirely plausible.")

    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]

    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df['date'])
    if idx.tz is not None:
        # The artifact is written tz-naive UTC; callers pass tz-aware UTC timestamps. A
        # naive/aware mismatch silently yields all-NaN on reindex, so normalise here once.
        idx = idx.tz_localize(None)
    s = pd.Series(df['R'].to_numpy(dtype='float64'), index=idx.normalize()).sort_index()
    _CACHE[path] = (key, s)
    return s


def _day(ts):
    """Bar timestamp -> tz-naive normalised UTC date."""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert('UTC').tz_localize(None)
    return t.normalize()


def _log(logger_name, level, msg):
    if logger_name:
        config_object.masterlog.get_logger(logger_name, level)(msg)


def get_r(path, curr_time, logger_name=None):
    """R for the bar at `curr_time`. Scalar; for the streaming per-minute path.

    Three regimes, matching PRODUCTION's
    `mm.shift(R_LAG_DAYS).reindex(dn).ffill().fillna(1.0)`
    (directional_momentum.py:58-59; the shift and the cap are baked into the artifact):
      exact hit          -> that value
      after the last row -> hold the last value, log CRITICAL (throttled hourly). The EOD
                            job is late or down; we keep trading on the most recent known
                            regime rather than stopping, but never silently.
      before the first   -> 1.0. This is the ONE sanctioned silent default in this sleeve,
                            because it is what the reference does and batch parity depends
                            on it. It cannot fire in production (the artifact starts
                            2020-06-27, long before any live warm-up).
    """
    s = _load(path)
    day = _day(curr_time)

    if len(s) == 0:
        raise ValueError(f"r_state: {path} is empty")

    if day in s.index:
        prev = _STALE_LOGGED.pop(path, None)
        if prev is not None:
            _log(logger_name, "info",
                 f"r_state: {path} is current again — using R for {day.date()}")
        return float(s.loc[day])

    if day < s.index[0]:
        return 1.0

    # day > last row: hold the last known value
    stale_days = int((day - s.index[-1]).days)
    slot = (day, pd.Timestamp(curr_time).hour)
    if _STALE_LOGGED.get(path) != slot:
        _STALE_LOGGED[path] = slot
        _log(logger_name, "critical",
             f"r_state: no row for {day.date()} in {path} (last row {s.index[-1].date()}, "
             f"{stale_days} day(s) stale) — holding R={float(s.iloc[-1]):.6f}. The EOD "
             f"builder has not run; entries are being sized off a stale market-vol regime.")
    return float(s.iloc[-1])


def r_multiplier_array(path, index, logger_name=None):
    """R aligned to a minute/candle index. Array; for the batch warm-up path.

    Transcribes PRODUCTION/engines/directional_momentum.py:58-59 exactly:
        mm = min(R_CAP, R_FLOOR + R_SLOPE*(1-pctv)).shift(R_LAG_DAYS) \
                 .reindex(dn).ffill().fillna(1.0)
    with the min/shift already applied by the EOD builder, leaving reindex/ffill/fillna.

    CALLER'S RESPONSIBILITY: apply the result to the SHORT side only. The LONG side of the
    same cell must multiply by nothing at all -- not by a defaulted 1.0 read from here,
    which would couple it to an artifact it has no business depending on.
    """
    s = _load(path)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = idx.normalize()

    rm = s.reindex(days).ffill()
    n_filled = int(rm.isna().sum())
    if n_filled:
        _log(logger_name, "info",
             f"r_state: {n_filled:,} bar(s) precede the first R row ({s.index[0].date()}) — "
             f"defaulting those to R=1.0, matching the reference.")
    if len(days) and days[-1] > s.index[-1]:
        _log(logger_name, "critical",
             f"r_state: warm-up runs to {days[-1].date()} but {path} ends "
             f"{s.index[-1].date()} — the tail is sized off a held, stale R.")
    return rm.fillna(1.0).to_numpy(dtype='float64')
