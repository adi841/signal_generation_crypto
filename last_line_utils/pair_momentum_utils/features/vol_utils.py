"""
Streaming volatility family for B1_v8_v10. All three mirror batch expressions in
`get_numba_parameters`, and all three are the SAME primitive (`EWMSpanStdAdjust`,
i.e. pandas `ewm(span=...).std()` with adjust=True / bias-corrected) applied to the
candle log-return, differing only in span and post-processing:

    lr        = np.log(close).diff()
    long_vol  = lr.ewm(span=long_vol_ewm_span).std()                    -> LongVolCalc
    atr_eq    = close * lr.ewm(span=atr_ewm_span).std()                 -> AtrEqCalc
    rv_alloc  = clip(lr.ewm(span=rv_ewm_span).std() * rv_scale, lo, hi) -> RvAllocCalc

NOTE: `atr_eq` is NOT a Wilder ATR — it is an EWM std of log-returns scaled to price
units. Do not substitute the sa_directional ATR primitives.

Each class keeps its own `prev_close` and recomputes the log-return. That duplication
is deliberate: it keeps every calc independently testable and removes all cross-calc
plumbing from the driver.
"""
import math
from collections import deque

from ._primitives import EWMSpanStdAdjust


def _log_return(curr_close, prev_close):
    """log(c_t / c_{t-1}); NaN on the first bar (matches np.log(close).diff())."""
    if math.isnan(prev_close):
        return float("nan")
    return math.log(curr_close / prev_close)


class LongVolCalc:
    """`long_vol = log_r.ewm(span=span).std()` (adjust=True).

    The kernel's TRAIL VOL: it is the denominator of the develop metric
    `pan = ((spc-em)/em)/lv` and the unit of the DDE trail
    `(ppx-spc)/ppx >= xdde * min(lv, elv)`.
    """

    def __init__(self, span):
        self.span = int(span)
        self._std = EWMSpanStdAdjust(self.span)
        self.prev_close = float("nan")
        self.long_vol_deque = deque(maxlen=3)
        self.long_vol_last = float("nan")

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        log_r = _log_return(c, self.prev_close)
        self.prev_close = c
        self.long_vol_last = self._std.update(log_r)
        self.long_vol_deque.append(self.long_vol_last)

    @property
    def get_logging_dict(self):
        return {"long_vol_last": self.long_vol_last}


class AtrEqCalc:
    """`atr_eq = close * log_r.ewm(span=span).std()` (adjust=True), in PRICE units.

    The kernel's BAND VOL: the Keltner half-width (`upper = med + K*atr_eq`), the
    hard-stop distance (`em - x*ea`), and the denominator of the entry sizing z.
    """

    def __init__(self, span):
        self.span = int(span)
        self._std = EWMSpanStdAdjust(self.span)
        self.prev_close = float("nan")
        self.atr_eq_deque = deque(maxlen=3)
        self.atr_eq_last = float("nan")

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        log_r = _log_return(c, self.prev_close)
        self.prev_close = c
        sd = self._std.update(log_r)
        self.atr_eq_last = c * sd          # NaN * c stays NaN, matching pandas
        self.atr_eq_deque.append(self.atr_eq_last)

    @property
    def get_logging_dict(self):
        return {"atr_eq_last": self.atr_eq_last}


class RvAllocCalc:
    """`rv_alloc = clip(log_r.ewm(span=span).std() * scale, q_lo, q_hi)`.

    The SIZING vol: `allocation = clip(VT / rv_alloc / annf, 0, 1)`.

    q_lo/q_hi are the IS-frozen expanding-quantile bounds for this (pair, TF) cell,
    loaded from the frozen artifact. They are constant for every post-IS bar, i.e.
    every bar production will ever process — never recompute them into live data.
    """

    def __init__(self, span, scale, q_lo, q_hi):
        self.span = int(span)
        self.scale = float(scale)
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self._std = EWMSpanStdAdjust(self.span)
        self.prev_close = float("nan")
        self.rv_alloc_deque = deque(maxlen=3)
        self.rv_alloc_last = float("nan")

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        log_r = _log_return(c, self.prev_close)
        self.prev_close = c
        sd = self._std.update(log_r)
        if math.isnan(sd):
            # pandas .clip() leaves NaN untouched — do the same rather than
            # clamping an unknown vol to a bound.
            self.rv_alloc_last = float("nan")
        else:
            rv = sd * self.scale
            self.rv_alloc_last = min(max(rv, self.q_lo), self.q_hi)
        self.rv_alloc_deque.append(self.rv_alloc_last)

    @property
    def get_logging_dict(self):
        return {"rv_alloc_last": self.rv_alloc_last}
