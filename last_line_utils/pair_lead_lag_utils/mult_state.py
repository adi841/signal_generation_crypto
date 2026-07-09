"""Reader for the daily R x I sizing multiplier (QR31_v2 pair_lead_lag).

mult multiplies the allocation of NEW ENTRIES only:
    tal = clip( clip((VT/annf)/sizing_vol, 0, 1) * mult, 0, 2 )
`allocation` appears in no kernel CONDITION, so mult changes position SIZE and never
entry/exit timing.

mult is a CROSS-MARKET daily state (10-symbol vol basket x the strategy's own entry
clock) that a single-pair process cannot derive; it arrives from
`eod_scripts/pair_lead_lag/build_mult_daily.py` as a parquet of (date, mult), indexed
by the day each value APPLIES TO (both state legs carry their own 1-day shift).

THE QR31 LAW DIFFERS FROM pair_momentum's R — NO ffill:
    PRODUCTION run_cell applies `mult.reindex(idx.normalize()).fillna(1.0)`.
A day missing from the artifact sizes at 1.0x, full stop. The reader reproduces that
exactly so replay and live match the batch bitwise; when the miss means the EOD job is
stale (day AFTER the last row) it logs CRITICAL — loudly 1.0, never silently held.
(pm's get_r holds the last value instead, because R's reference law ffills. Do not
"harmonise" the two: each mirrors its own frozen reference.)

Same infrastructure discipline as r_state.py: stat()-based re-read (the file is
rewritten nightly — no lru_cache), keyed on the BAR's timestamp (replay == live code
path), atomic-write partner on the builder side.
"""
import os

import numpy as np
import pandas as pd

from config.config_read import config_object

# path -> ((mtime_ns, size), Series indexed by tz-naive normalised dates)
_CACHE = {}
# path -> the (day, hour) we last emitted a staleness critical for
_STALE_LOGGED = {}


def _load(path):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"mult_state: {path} does not exist. Build it with "
            f"eod_scripts/pair_lead_lag/build_mult_daily.py (once), then nightly. "
            f"Refusing to default mult to 1.0 wholesale -- that would mis-size every "
            f"entry by up to 1.5x/0.3x while looking entirely plausible.")

    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]

    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df['date'])
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(df['mult'].to_numpy(dtype='float64'), index=idx.normalize()).sort_index()
    _CACHE[path] = (key, s)
    return s


def _day(ts):
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert('UTC').tz_localize(None)
    return t.normalize()


def _log(logger_name, level, msg):
    if logger_name:
        config_object.masterlog.get_logger(logger_name, level)(msg)


def get_mult(path, curr_time, logger_name=None):
    """mult for the bar at `curr_time`. Scalar; for the streaming per-minute path.

    Regimes, matching PRODUCTION's `mult.reindex(idx.normalize()).fillna(1.0)`:
      exact hit           -> that value
      before the first    -> 1.0, silent (the state's expanding(120) warm-up; the
                             reference does exactly this and batch parity depends on it)
      after the last row  -> 1.0 — SAME numeric as the reference law — but CRITICAL
                             (throttled hourly): the EOD job is late or down and
                             entries are sizing at neutral instead of the live regime.
    """
    s = _load(path)
    day = _day(curr_time)

    if len(s) == 0:
        raise ValueError(f"mult_state: {path} is empty")

    if day in s.index:
        prev = _STALE_LOGGED.pop(path, None)
        if prev is not None:
            _log(logger_name, "info",
                 f"mult_state: {path} is current again — using mult for {day.date()}")
        return float(s.loc[day])

    if day < s.index[0]:
        return 1.0

    stale_days = int((day - s.index[-1]).days)
    slot = (day, pd.Timestamp(curr_time).hour)
    if _STALE_LOGGED.get(path) != slot:
        _STALE_LOGGED[path] = slot
        _log(logger_name, "critical",
             f"mult_state: no row for {day.date()} in {path} (last row "
             f"{s.index[-1].date()}, {stale_days} day(s) stale) — sizing at mult=1.0 "
             f"per the reference law (NO ffill). The EOD builder has not run.")
    return 1.0


def mult_multiplier_array(path, index, logger_name=None):
    """mult aligned to a minute index. Array; for the batch warm-up path.

    Transcribes PRODUCTION/engines/pairs_leadlag.py::run_cell line 91 exactly:
        mult.reindex(idx.normalize()).fillna(1.0)
    """
    s = _load(path)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = idx.normalize()

    mm = s.reindex(days)
    n_before = int(mm.isna().sum())
    if len(days) and days[-1] > s.index[-1]:
        _log(logger_name, "critical",
             f"mult_state: warm-up runs to {days[-1].date()} but {path} ends "
             f"{s.index[-1].date()} — the tail sizes at mult=1.0 (reference law, "
             f"NO ffill); the EOD builder is behind the data.")
    elif n_before:
        _log(logger_name, "info",
             f"mult_state: {n_before:,} bar(s) precede the first mult row "
             f"({s.index[0].date()}) — defaulting to 1.0, matching the reference.")
    return mm.fillna(1.0).to_numpy(dtype='float64')
